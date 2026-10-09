"""LLM-as-judge pass: score pending responses with a judge model, stored apart (PRD §18).

Runs after the measured runs (or alone with `llb judge`). The judge answers in JSON
constrained by a schema (`response_format`), one 1-5 score per criterion plus a short
explanation, so its output is always parseable.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llmbench.config.experiment import Experiment, ModelSpec
from llmbench.config.matrix import apply_variant
from llmbench.data.requests import Request
from llmbench.engine.runtime import Events
from llmbench.hub.store import ModelNotFoundError, ModelStore
from llmbench.llamacpp import builds
from llmbench.llamacpp.argv import server_argv
from llmbench.llamacpp.client import LlamaClient
from llmbench.llamacpp.server import LlamaServer, ServerError, next_log_path
from llmbench.results.store import Store

CRITERIA = {
    "grammar": "spelling, grammar and punctuation are correct",
    "clarity": "the text is clear, direct and easy to read",
    "naturalness": "it reads like a fluent native speaker wrote it",
    "instruction_following": "it does exactly what the task asks, respecting every constraint",
    "faithfulness": "it adds no information that is not in the task or input",
    "style_preservation": "it keeps the tone and style of the original text",
    "conciseness": "it has no unnecessary words or repetitions",
    "correctness": "the content or answer is correct",
}

SYSTEM = (
    "You are a strict, impartial evaluator of written responses. Score each criterion from "
    "1 (very poor) to 5 (excellent). Judge only what is asked; do not reward length."
)


@dataclass
class JudgeSummary:
    judged: int = 0
    failed: int = 0
    runs: int = 0
    error: str | None = None


def schema(criteria: list[str]) -> dict[str, Any]:
    props: dict[str, Any] = {c: {"type": "integer", "minimum": 1, "maximum": 5} for c in criteria}
    props["explanation"] = {"type": "string", "maxLength": 400}
    return {"type": "object", "properties": props, "required": [*criteria, "explanation"]}


def prompt(row: dict[str, Any], criteria: list[str], rubric: str | None) -> str:
    lines = ["Evaluate the RESPONSE to the TASK.", "", "Criteria:"]
    lines += [f"- {c}: {CRITERIA.get(c, c.replace('_', ' '))}" for c in criteria]
    if rubric:
        lines += ["", f"Additional rubric: {rubric}"]
    lines += ["", "TASK:", str(row.get("prompt_text") or "")]
    if row.get("reference"):
        lines += ["", "REFERENCE (a good answer, not the only one):", str(row["reference"])]
    lines += ["", "RESPONSE:", str(row.get("content") or ""), "", "Answer with JSON only."]
    return "\n".join(lines)


def run_judge(
    exp: Experiment,
    store: Store,
    events: Events,
    *,
    model_store: ModelStore | None = None,
) -> JudgeSummary:
    cfg = exp.phases.judge
    summary = JudgeSummary()
    pending = store.pending_judgements(exp.experiment.name)
    if not pending:
        return summary
    if cfg.model is None:
        summary.error = "no judge model configured (phases.judge.model)"
        return summary
    build_spec = next(
        (b for b in exp.llama_cpp.builds if b.id == cfg.build), exp.llama_cpp.builds[0]
    )
    try:
        rb = builds.resolve(build_spec)
        models = model_store or ModelStore(log=events.log)
        judge_model = models.resolve(ModelSpec(id="judge", source=cfg.model.source))
        sha = judge_model.sha256 or store.model_sha256(judge_model.path)
    except (builds.BuildError, ModelNotFoundError) as e:
        summary.error = str(e)
        return summary
    llama_cfg = apply_variant(exp.defaults, cfg.config)
    identity = {"source": cfg.model.source, "sha256": sha, "build_commit": rb.commit}
    events.log(f"judge: {len(pending)} response(s) with {judge_model.path.name}")
    server = LlamaServer(
        lambda port: server_argv(rb.tool("llama-server"), judge_model.path, llama_cfg, port=port),
        next_log_path(store.path.parent / "logs", "judge-server"),
        llama_cfg.env,
        cfg.startup_timeout_s,
    )
    try:
        server.start()
    except ServerError as e:
        summary.error = str(e)
        return summary
    client = LlamaClient(server.url)
    touched: set[int] = set()
    try:
        for row in pending:
            spec = json.loads(row["judge_spec"] or "{}")
            criteria: list[str] = spec.get("criteria") or list(CRITERIA)[:4]
            req = Request(
                id=row["request_id"],
                prompt=prompt(row, criteria, spec.get("rubric")),
                n_predict=cfg.max_tokens,
                system=SYSTEM,
            )
            body = {
                "temperature": 0,
                "seed": 42,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": schema(criteria)},
                },
            }
            result = client.complete(req, body, endpoint="chat")
            scores = _parse(result.content, criteria) if result.ok else None
            if scores is None:
                summary.failed += 1
                continue
            store.save_judgement(
                run_id=row["run_id"],
                idx=row["idx"],
                request_id=row["request_id"],
                scores=scores,
                judge=identity,
                judged_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            touched.add(row["run_id"])
            summary.judged += 1
    finally:
        client.close()
        server.stop()
    for run_id in touched:
        store.refresh_judge_metrics(run_id)
    summary.runs = len(touched)
    return summary


def _parse(text: str, criteria: list[str]) -> dict[str, tuple[float | None, str]] | None:
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None
    explanation = str(data.get("explanation", ""))[:400]
    out: dict[str, tuple[float | None, str]] = {}
    for c in criteria:
        v = data.get(c)
        out[c] = (float(v) if isinstance(v, int | float) and 1 <= v <= 5 else None, explanation)  # noqa: PLR2004
    return out

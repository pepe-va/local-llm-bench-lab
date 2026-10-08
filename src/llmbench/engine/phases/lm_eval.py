"""lm-evaluation-harness against a configuration's llama-server (OpenAI-compatible API).

llb only orchestrates: it starts (or reuses) the server, runs `lm_eval` as a subprocess with
the `local-chat-completions` / `local-completions` backend, and flattens the results JSON into
run metrics `lm_eval:<task>.<metric>[.<filter>]`.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmbench.config.experiment import LmEvalPhase
from llmbench.engine.phases.suites import ServerPool
from llmbench.engine.planner import RunSpec
from llmbench.engine.runtime import ModelHandle, RunContext, iso, now
from llmbench.llamacpp.argv import Argv, server_argv
from llmbench.llamacpp.builds import ResolvedBuild
from llmbench.llamacpp.server import ServerError
from llmbench.results.models import RunRecord, RunStatus
from llmbench.telemetry.sampler import Sampler


def lm_eval_command() -> list[str] | None:
    exe = shutil.which("lm_eval")
    if exe:
        return [exe]
    if importlib.util.find_spec("lm_eval") is not None:
        return [sys.executable, "-m", "lm_eval"]
    return None


def lm_eval_argv(cfg: LmEvalPhase, base_url: str, model_name: str, out_dir: Path) -> list[str]:
    cmd = lm_eval_command() or ["lm_eval"]
    chat = cfg.model == "local-chat-completions"
    endpoint = f"{base_url}/v1/{'chat/completions' if chat else 'completions'}"
    model_args = (
        f"base_url={endpoint},model={model_name},num_concurrent=1,max_retries=2,"
        "tokenized_requests=False"
    )
    argv = [*cmd, "--model", cfg.model, "--model_args", model_args]
    argv += ["--tasks", ",".join(cfg.tasks), "--output_path", str(out_dir)]
    if cfg.limit is not None:
        argv += ["--limit", str(cfg.limit)]
    if cfg.num_fewshot is not None:
        argv += ["--num_fewshot", str(cfg.num_fewshot)]
    if chat and cfg.apply_chat_template:
        argv.append("--apply_chat_template")
    return argv + cfg.extra_args


def flatten_results(data: dict[str, Any]) -> dict[str, float | None]:
    """`{"results": {"gsm8k": {"exact_match,strict-match": 0.4, ...}}}` -> run metrics."""
    out: dict[str, float | None] = {}
    results: dict[str, dict[str, Any]] = data.get("results", {})
    for task, metrics in results.items():
        for key, value in metrics.items():
            if key == "alias" or "_stderr" in key or not isinstance(value, int | float):
                continue
            metric, _, filt = key.partition(",")
            name = f"lm_eval:{task}.{metric}" + (f".{filt}" if filt and filt != "none" else "")
            out[name] = float(value)
    return out


def _fingerprint(spec: RunSpec, rb: ResolvedBuild, model: ModelHandle, cfg: LmEvalPhase) -> str:
    payload = {
        "phase": "lm_eval",
        "build": rb.identity(),
        "model_sha256": model.sha256,
        "config": spec.config.server_identity(),
        "lm_eval": cfg.model_dump(
            mode="json", exclude={"enabled", "timeout_s", "startup_timeout_s"}
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_lm_eval_spec(
    ctx: RunContext,
    spec: RunSpec,
    *,
    rb: ResolvedBuild,
    model: ModelHandle,
    pool: ServerPool,
    index: int,
    total: int,
) -> RunStatus | None:
    cfg = ctx.exp.phases.lm_eval
    fp = _fingerprint(spec, rb, model, cfg)
    if ctx.resume and ctx.store.is_completed(fp):
        ctx.events.run_skipped(index, total, spec, "already completed")
        return None
    ctx.events.run_start(index, total, spec)
    started = now()
    status: RunStatus = "completed"
    error: str | None = None
    metrics: dict[str, float | None] = {}
    argv: list[str] = []
    binary = rb.tool("llama-server")

    def make_argv(port: int) -> Argv:
        return server_argv(binary, model.path, spec.config, port=port)

    sampler = Sampler(ctx.exp.hardware.telemetry)
    telemetry = None
    if lm_eval_command() is None:
        status, error = "failed", "lm_eval is not installed: `uv sync --extra lm-eval`"
    else:
        try:
            server, _, _ = pool.acquire(
                spec.server_key(), make_argv, spec.config.env, cfg.startup_timeout_s
            )
            sampler.start()
            sampler.attach(server.pid)
            out_dir = ctx.out_dir / "lm_eval" / fp[:12]
            argv = lm_eval_argv(cfg, server.url, spec.model.id, out_dir)
            status, error, metrics = _execute(argv, out_dir, cfg.timeout_s)
        except ServerError as e:
            status, error = "failed", str(e)
        except KeyboardInterrupt:
            status, error = "interrupted", "interrupted by user"
            pool.release()
        finally:
            telemetry = sampler.stop() if sampler.started else None
    finished = now()
    record = RunRecord(
        fingerprint=fp,
        experiment=ctx.exp.experiment.name,
        phase="lm_eval",
        status=status,
        build={"id": rb.id, "ref": rb.ref, "version": rb.version, **rb.identity()},
        model=model.info,
        variant=spec.variant,
        config=spec.config.model_dump(mode="json"),
        argv=argv,
        ignored=[],
        env=ctx.machine,
        started_at=iso(started),
        finished_at=iso(finished),
        duration_s=(finished - started).total_seconds(),
        error=error,
        telemetry=telemetry.to_dict() if telemetry else None,
        suite="lm_eval:" + ",".join(cfg.tasks),
        metrics=metrics,
    )
    ctx.save(record, telemetry)
    ctx.events.run_end(index, total, record)
    return status


def _execute(
    argv: list[str], out_dir: Path, timeout_s: float
) -> tuple[RunStatus, str | None, dict[str, float | None]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "lm_eval.log"
    with log.open("wb") as f:
        try:
            proc = subprocess.run(
                argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return "failed", f"lm_eval timed out after {timeout_s:.0f}s (log: {log})", {}
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:])
        return "failed", f"lm_eval exit code {proc.returncode}\n{tail}", {}
    files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return "failed", f"lm_eval wrote no results under {out_dir}", {}
    return "completed", None, flatten_results(json.loads(files[-1].read_text()))

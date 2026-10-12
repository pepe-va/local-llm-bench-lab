"""The lab's suite editor: the request files in `suites/`, checked and tried as a run would.

A suite is the list of requests an experiment sends to llama-server, plus how to score the
answers. Files live in the repository's `suites/` (where experiments point), so the
terminal and the lab read the same thing. Validation goes through `load_suite` and
`materialize`, the path `llb validate` and `llb run` use; relative paths in a draft
(`file:`, images, a corpus) resolve as they will once it is saved, because the draft is
written next to the real files for the check.

"Try it" sends the first requests to the model loaded in the lab's chat and scores the
answers with the deterministic evaluators. It is a quick check while writing a suite, not
a measurement: nothing is stored, the chat server's settings are not an experiment's, and
there is no warm-up, cooldown or telemetry.
"""

import os
import re
import statistics
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter
from ruamel.yaml.scalarstring import LiteralScalarString

from llmbench.config.evaluators import All, AnyOf, EvaluatorSpec, Weighted
from llmbench.config.loader import ConfigError, load_suite
from llmbench.config.suite import Suite
from llmbench.data import synthetic
from llmbench.data.hf_datasets import DatasetError
from llmbench.data.requests import Request
from llmbench.data.sources import SuiteError, materialize
from llmbench.evaluation.deterministic import evaluate
from llmbench.lab.editor import EditorError, parse
from llmbench.llamacpp.client import LlamaClient

_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml$")
_LINE = re.compile(r"\(line (\d+)\)")
_EVAL = TypeAdapter[EvaluatorSpec](EvaluatorSpec)
PREVIEW = 20
"""Requests shown by a preview."""
TRY_MAX = 50
"""Most requests one "try it" may send (it holds the request open until done)."""
_TEXT = 4000
"""Longest prompt/answer text sent back to the page."""


def literal_blocks(value: Any) -> Any:
    """Multi-line strings become `|` blocks, like the prompts in `suites/`."""
    if isinstance(value, str) and "\n" in value:
        return LiteralScalarString(value)
    if isinstance(value, list):
        return [literal_blocks(v) for v in cast(list[Any], value)]
    if isinstance(value, dict):
        return {k: literal_blocks(v) for k, v in cast(dict[str, Any], value).items()}
    return value


# -- new suites ------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    key: str
    title: str
    help: str
    text: str
    """YAML with a `{name}` placeholder."""


KINDS = [
    Kind(
        "chat-es",
        "Spanish prompts with checks",
        "Hand-written prompts in Spanish (like es-writing.yaml): correcting, summarising, "
        "rules the answer must follow. Each item shows a different kind of check.",
        """\
# {name}: tareas en español con comprobaciones deterministas.
# Cada item usa un tipo de evaluador distinto: copia el que se parezca a lo que quieres medir.
version: 1
name: {name}
description: Qué miden estas peticiones.
endpoint: chat
system: >-
  Eres un asistente experto en español. Responde solo con lo que se pide, sin explicaciones.
max_tokens: 256
source:
  items:
    - id: capital
      category: datos
      prompt: ¿Cuál es la capital de Portugal? Responde con una sola palabra.
      reference: Lisboa
      evaluator: {{ type: exact, normalize: [strip, lower, strip_punct, strip_accents] }}
    - id: correccion
      category: correccion
      input: Haber si mañana podemos quedar.
      prompt: |
        Corrige las faltas de ortografía del texto.

        Texto: Haber si mañana podemos quedar.
      evaluator:
        type: all
        of:
          - {{ type: contains, values: [A ver si] }}
          - {{ type: not_contains, values: [Haber si] }}
    - id: resumen
      category: resumen
      input: >-
        La reunión del lunes se aplaza al miércoles porque el informe de ventas no estará
        listo a tiempo y el equipo de finanzas necesita revisarlo antes.
      prompt: |
        Resume el texto en una sola frase.

        Texto: La reunión del lunes se aplaza al miércoles porque el informe de ventas no
        estará listo a tiempo y el equipo de finanzas necesita revisarlo antes.
      evaluator:
        type: all
        of:
          - {{ type: contains, values: [miércoles] }}
          - {{ type: length, max_words: 25 }}
    - id: lista
      category: formato
      prompt: Escribe exactamente tres frutas, separadas por comas y sin nada más.
      evaluator: {{ type: regex, pattern: '^\\s*[^,]+,[^,]+,[^,]+\\s*$', mode: fullmatch }}
""",
    ),
    Kind(
        "chat",
        "chat items with checks",
        "Hand-written prompts sent to the chat endpoint, each with deterministic checks "
        "(like es-writing.yaml). For the quality and latency phases.",
        """\
# {name}: hand-written prompts with deterministic checks.
version: 1
name: {name}
description: What these requests test.
endpoint: chat
system: Answer briefly and precisely.
max_tokens: 256
source:
  items:
    - id: capital
      category: facts
      prompt: What is the capital of France? Answer with one word.
      reference: Paris
      evaluator: {{ type: exact, normalize: [strip, lower, strip_punct] }}
    - id: summary
      category: writing
      input: The meeting moved from Monday to Wednesday because the report was not ready.
      prompt: |
        Summarize in one short sentence.

        Text: The meeting moved from Monday to Wednesday because the report was not ready.
      evaluator:
        type: all
        of:
          - {{ type: contains, values: [Wednesday] }}
          - {{ type: length, max_words: 20 }}
""",
    ),
    Kind(
        "synthetic",
        "synthetic (exact token counts)",
        "Prompts with an exact number of input tokens, cut from a corpus with each model's "
        "own tokenizer. For speed and serving measurements; no scoring.",
        """\
# {name}: synthetic prompts with exact token counts (performance, no scoring).
version: 1
name: {name}
description: 512 prompt tokens, 128 generated tokens.
source:
  synthetic:
    prompt_tokens: {{ dist: fixed, value: 512 }}
    output_tokens: {{ dist: fixed, value: 128 }}
    count: 20
    seed: 42
""",
    ),
    Kind(
        "file",
        "rows from a file",
        "Requests read from a .jsonl, .json, .csv or .parquet file next to the suite; "
        "`columns` maps the file's columns to prompt, reference and so on.",
        """\
# {name}: one request per row of a local file (path relative to this suite).
version: 1
name: {name}
endpoint: chat
max_tokens: 256
source:
  file:
    path: data/{name}.jsonl
columns: {{ prompt: question, reference: answer }}
evaluator: {{ type: exact, normalize: [strip, lower] }}
""",
    ),
    Kind(
        "hf_dataset",
        "Hugging Face dataset",
        "Rows of a dataset on the Hub, read from its parquet conversion (needs the "
        "hf-datasets extra). Pin `revision` for a repeatable series.",
        """\
# {name}: questions from a Hugging Face dataset (its parquet conversion).
version: 1
name: {name}
endpoint: chat
system: You are a careful math tutor. Think step by step, then give the final answer.
template: "{{prompt}}\\n\\nEnd your answer with a final line of the form: #### <number>"
max_tokens: 512
source:
  hf_dataset:
    repo: openai/gsm8k
    config: main
    split: test
    limit: 20
    shuffle_seed: 42
columns: {{ prompt: question, reference: answer }}
evaluator:
  type: regex_extract
  pattern: "####\\\\s*\\\\$?(-?[\\\\d,]*\\\\.?\\\\d+)"
  reference_pattern: "####\\\\s*(-?[\\\\d,]*\\\\.?\\\\d+)"
  compare: numeric
""",
    ),
    Kind(
        "embeddings",
        "embedding pairs",
        "Sentence pairs with a similarity label, for embedding models.",
        """\
# {name}: sentence pairs and their similarity (0-1), for embedding models.
version: 1
name: {name}
endpoint: embeddings
source:
  items:
    - {{ a: The cat sleeps on the sofa., b: A cat is napping on the couch., label: 0.9 }}
    - {{ a: The cat sleeps on the sofa., b: Interest rates rose again., label: 0.0 }}
""",
    ),
    Kind(
        "rerank",
        "reranking",
        "Queries with candidate documents and the indices of the relevant ones.",
        """\
# {name}: for each query, which documents are relevant (indices into `documents`).
version: 1
name: {name}
endpoint: rerank
k: 3
source:
  items:
    - query: What is a panda?
      documents:
        - Pandas is a Python library for data analysis.
        - The giant panda is a bear native to China that eats bamboo.
        - Bamboo grows very fast.
      relevant: [1]
""",
    ),
]


@dataclass(frozen=True)
class EvaluatorTip:
    """One row of the evaluator cheat sheet: when to use a type, and a valid example."""

    type: str
    use_when: str
    example: dict[str, Any]


EVALUATOR_GUIDE = [
    EvaluatorTip(
        "exact",
        "There is one right answer, and normalizing (case, accents, punctuation) "
        "is enough to compare it. Expected text: `value`, or the item's `reference`.",
        {"type": "exact", "normalize": ["strip", "lower", "strip_punct", "strip_accents"]},
    ),
    EvaluatorTip(
        "contains",
        "The answer must mention some words or phrases, wherever they are. "
        "`mode: any` passes with one of them; the score is the share found.",
        {"type": "contains", "values": ["miércoles", "informe"], "mode": "all"},
    ),
    EvaluatorTip(
        "not_contains",
        "Some text must not appear: the error you asked to fix, a forbidden "
        "word, an English answer to a Spanish prompt.",
        {"type": "not_contains", "values": ["Haber si", "aver"]},
    ),
    EvaluatorTip(
        "regex",
        "The answer must follow a format: a date, a list of three, one word. "
        "`fullmatch` checks the whole answer, `search` any part of it.",
        {"type": "regex", "pattern": "^\\d{4}-\\d{2}-\\d{2}$", "mode": "fullmatch"},
    ),
    EvaluatorTip(
        "regex_extract",
        "The answer reasons first and gives a result at the end (maths): "
        "extract it and compare it with the reference, as text or as a number.",
        {"type": "regex_extract", "pattern": "####\\s*(-?[\\d.,]+)", "compare": "numeric"},
    ),
    EvaluatorTip(
        "length",
        "Summaries and rewrites: limit words or characters, or compare the length "
        "with the item's `input` (`ratio_to`), e.g. at most half of it.",
        {"type": "length", "ratio_to": "input", "max_ratio": 0.5},
    ),
    EvaluatorTip(
        "json_schema",
        "The model must answer with JSON (or a ```json block) of a given "
        "shape: extraction, tool-like outputs.",
        {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "required": ["nombre", "edad"],
                "properties": {"nombre": {"type": "string"}, "edad": {"type": "integer"}},
            },
        },
    ),
    EvaluatorTip(
        "llm_judge",
        "Quality no rule can check (grammar, naturalness, faithfulness). "
        "A stronger model rates it 1-5 later, with `llb judge`; stored apart.",
        {"type": "llm_judge", "criteria": ["grammar", "clarity", "faithfulness"]},
    ),
    EvaluatorTip(
        "all",
        "Several checks at once, all required: e.g. mentions the key fact and stays "
        "short. The score is the lowest part's.",
        {
            "type": "all",
            "of": [
                {"type": "contains", "values": ["miércoles"]},
                {"type": "length", "max_words": 25},
            ],
        },
    ),
    EvaluatorTip(
        "any",
        "Several acceptable answers in different forms: passes if one check passes.",
        {
            "type": "any",
            "of": [
                {"type": "exact", "value": "Lisboa"},
                {"type": "contains", "values": ["Lisbon"]},
            ],
        },
    ),
    EvaluatorTip(
        "weighted",
        "Partial credit: some checks matter more than others. Passes when the "
        "weighted mean reaches `pass_threshold`.",
        {
            "type": "weighted",
            "pass_threshold": 0.7,
            "of": [
                {"weight": 3, "evaluator": {"type": "contains", "values": ["A ver si"]}},
                {"weight": 1, "evaluator": {"type": "length", "max_words": 30}},
            ],
        },
    ),
]
"""Every example is validated against `EvaluatorSpec` in the tests."""


def new_suite(kind: str, name: str) -> str:
    found = next((k for k in KINDS if k.key == kind), None)
    if found is None:
        raise EditorError(f"unknown kind of suite {kind!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise EditorError(f"invalid suite name {name!r}: use letters, digits, . _ -")
    return found.text.format(name=name)


# -- files -----------------------------------------------------------------------------


class Suites:
    """The YAML files of `suites/`, and which experiments use each of them."""

    def __init__(self, folder: Path, experiments: Path) -> None:
        self.folder = folder
        self.experiments = experiments

    def files(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(p.name for p in self.folder.glob("*.y*ml") if _FILE.match(p.name))

    def path(self, name: str) -> Path:
        if not _FILE.match(name):
            raise EditorError(f"invalid file name {name!r}: use letters, digits, . _ - and .yaml")
        return self.folder / name

    def read(self, name: str) -> str:
        p = self.path(name)
        if not p.is_file():
            raise EditorError(f"no such suite: {name}")
        return p.read_text(encoding="utf-8")

    def save(self, name: str, text: str, *, overwrite: bool) -> Path:
        _mapping(text)  # never save something that is not even YAML
        p = self.path(name)
        if p.exists() and not overwrite:
            raise EditorError(f"{name} already exists")
        self.folder.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def used_by(self) -> dict[str, list[str]]:
        """Suite file name -> experiment files whose phases list it."""
        out: dict[str, list[str]] = {}
        if not self.experiments.is_dir():
            return out
        folder = self.folder.resolve()
        for exp in sorted(self.experiments.glob("*.y*ml")):
            if not _FILE.match(exp.name):
                continue
            try:
                doc = parse(exp.read_text(encoding="utf-8"))
            except (EditorError, OSError, UnicodeDecodeError):
                continue
            for suite in _suite_refs(doc):
                # anchored like the loader does (lexically, then resolved)
                p = Path(os.path.normpath(self.experiments / suite)).resolve()
                if p.parent == folder and exp.name not in out.setdefault(p.name, []):
                    out[p.name].append(exp.name)
        return out

    def load(self, text: str) -> Suite:
        """Validate unsaved text as `llb` would load it, from a hidden draft in `suites/`."""
        _mapping(text)
        self.folder.mkdir(parents=True, exist_ok=True)
        draft = self.folder / f".llb-lab-{uuid.uuid4().hex[:8]}.yaml"
        draft.write_text(text, encoding="utf-8")
        try:
            return load_suite(draft)
        except ConfigError as e:
            message = str(e).replace(str(draft.resolve()), "<draft>").replace(str(draft), "<draft>")
            m = _LINE.search(message)
            raise EditorError(message, int(m.group(1)) if m else None) from None
        finally:
            draft.unlink(missing_ok=True)


def _mapping(text: str) -> dict[str, Any]:
    doc = parse(text)
    if not doc:
        raise EditorError("the suite is empty")
    return doc


def _suite_refs(doc: dict[str, Any]) -> list[str]:
    phases = doc.get("phases")
    if not isinstance(phases, dict):
        return []
    out: list[str] = []
    for phase in ("latency", "quality", "serving"):
        node = cast(dict[str, Any], phases).get(phase)
        suites = cast(dict[str, Any], node).get("suites") if isinstance(node, dict) else None
        if isinstance(suites, list):
            out += [str(s) for s in cast(list[Any], suites)]
    return out


# -- requests --------------------------------------------------------------------------


def source_kind(suite: Suite) -> str:
    src = suite.source
    return next(k for k in ("synthetic", "items", "file", "hf_dataset") if getattr(src, k))


def requests_of(
    suite: Suite, *, fetch: bool = True, client: LlamaClient | None = None
) -> list[Request]:
    """The requests a run would send (synthetic ones need a live tokenizer)."""
    if suite.is_synthetic:
        if client is None:
            raise EditorError("synthetic prompts are cut with a model's tokenizer: load one first")
        assert suite.source.synthetic is not None
        try:
            return synthetic.build(
                suite.source.synthetic, tokenize=client.tokenize, bos=client.bos(),
                suite=suite.name,
            )  # fmt: skip
        except ValueError as e:
            raise EditorError(str(e)) from None
    if suite.source.hf_dataset is not None and not fetch:
        raise EditorError("the dataset is downloaded only when asked")
    try:
        return materialize(suite).requests
    except (SuiteError, DatasetError, OSError, ValueError) as e:
        raise EditorError(str(e)) from None
    except ImportError as e:  # pyarrow, for parquet files and datasets
        raise EditorError(f"{e}: install the extra with `uv sync --extra hf-datasets`") from None


def describe_evaluator(spec: EvaluatorSpec | None) -> str | None:
    """`all(contains, length)`: what an evaluator checks, in one line."""
    match spec:
        case None:
            return None
        case All() | AnyOf():
            return f"{spec.type}({', '.join(describe_evaluator(p) or '' for p in spec.of)})"
        case Weighted():
            parts = ", ".join(f"{w.weight:g}*{describe_evaluator(w.evaluator)}" for w in spec.of)
            return f"weighted({parts})"
        case _:
            return spec.type


def request_json(req: Request) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": req.id,
        "category": req.category,
        "prompt": _clip(req.prompt) if isinstance(req.prompt, str) else None,
        "prompt_tokens": req.expected_prompt_tokens,
        "system": req.system,
        "max_tokens": req.n_predict,
        "input": _clip(req.input),
        "reference": _clip(req.reference),
        "evaluator": describe_evaluator(req.evaluator),
        "evaluator_spec": _EVAL.dump_python(req.evaluator, mode="json", by_alias=True)
        if req.evaluator
        else None,
        "image": Path(req.image).name if req.image and not req.image.startswith("data:") else None,
        "a": req.a,
        "b": req.b,
        "label": req.label,
        "query": req.query,
        "documents": list(req.documents) or None,
        "relevant": list(req.relevant) or None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _clip(text: str | None) -> str | None:
    if text is None or len(text) <= _TEXT:
        return text
    return text[:_TEXT] + f"… ({len(text) - _TEXT} more characters)"


def preview(
    suites: Suites, text: str, *, fetch: bool = False, limit: int = PREVIEW
) -> dict[str, Any]:
    """Validate, then show the requests a run would send (as far as possible offline)."""
    suite = suites.load(text)
    kind = source_kind(suite)
    out: dict[str, Any] = {
        "name": suite.name,
        "endpoint": suite.endpoint,
        "source": kind,
        "max_tokens": suite.max_tokens,
        "evaluator": describe_evaluator(suite.evaluator),
        "sampling": suite.sampling.body(),
        "notes": [],
    }
    notes: list[str] = out["notes"]
    if suite.source.synthetic is not None:
        sizes = synthetic.shapes(suite.source.synthetic)
        prompts, outputs = [p for p, _ in sizes], [o for _, o in sizes]
        out["count"] = len(sizes)
        out["shapes"] = {
            "prompt_tokens": _stats(prompts),
            "output_tokens": _stats(outputs),
        }
        out["requests"] = [
            {"id": f"{suite.name}/{i:05d}", "prompt_tokens": p, "max_tokens": o}
            for i, (p, o) in enumerate(sizes[:limit])
        ]
        notes.append(
            "Synthetic prompts are cut from the corpus with each model's own tokenizer when the "
            "run starts, so their token counts are exact; the text depends on the model. "
            '"Try it" builds them with the chat model\'s tokenizer.'
        )
        notes.append("Runs send them with ignore_eos unless the suite's sampling says otherwise.")
        return out
    if suite.source.hf_dataset is not None and not fetch:
        ds = suite.source.hf_dataset
        out["dataset"] = ds.model_dump(exclude_none=True)
        notes.append(
            f"Valid. The rows of {ds.repo} are downloaded on the first run (or with "
            '"fetch dataset"), into the Hugging Face cache.'
        )
        out["requests"] = []
        return out
    requests = requests_of(suite, fetch=fetch)
    out["count"] = len(requests)
    out["requests"] = [request_json(r) for r in requests[:limit]]
    if suite.endpoint in ("embeddings", "rerank"):
        notes.append(
            f"{suite.endpoint} suites are scored by the run itself (similarity, nDCG@k); "
            "the chat server cannot try them."
        )
    elif not any(r.evaluator for r in requests):
        notes.append("No evaluator: answers are timed but not scored.")
    return out


def _stats(values: list[int]) -> dict[str, float]:
    return {"min": min(values), "mean": round(statistics.fmean(values), 1), "max": max(values)}


# -- try it ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TryOptions:
    limit: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    """Cap on the answer length (the suite's own `max_tokens` otherwise)."""


def try_events(
    suite: Suite,
    url: str,
    opts: TryOptions,
    *,
    alive: Callable[[], str | None] = lambda: None,
) -> Iterator[dict[str, Any]]:
    """`{start}`, one `{result}` per request, then `{done}` (or `{error}`).

    `alive` returns why to stop (the chat was unloaded, a job started) or None.
    """
    if suite.endpoint in ("embeddings", "rerank"):
        yield {"error": f"{suite.endpoint} needs llama-server started for it: run an experiment"}
        return
    client = LlamaClient(url, timeout_s=600)
    try:
        try:
            requests = requests_of(suite, fetch=True, client=client)
        except EditorError as e:
            yield {"error": str(e)}
            return
        sampling: dict[str, Any] = {"temperature": opts.temperature} | suite.sampling.body()
        if suite.is_synthetic:
            sampling.setdefault("ignore_eos", True)  # as a run sends them
        chosen = requests[: max(1, min(opts.limit, TRY_MAX))]
        yield {"start": {"count": len(chosen), "of": len(requests), "sampling": sampling}}
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for i, original in enumerate(chosen):
            reason = alive()
            if reason:
                yield {"error": f"stopped: {reason}", "totals": _totals(rows, t0)}
                return
            req = original
            if opts.max_tokens is not None and opts.max_tokens < req.n_predict:
                req = replace(req, n_predict=opts.max_tokens)
            row = _try_one(client, req, sampling, suite.endpoint) | {"i": i}
            rows.append(row)
            yield {"result": row}
        yield {"done": _totals(rows, t0)}
    finally:
        client.close()


def _try_one(
    client: LlamaClient, req: Request, sampling: dict[str, Any], endpoint: str
) -> dict[str, Any]:
    try:
        r = client.complete(req, sampling, endpoint=endpoint)
    except (OSError, ValueError) as e:  # e.g. an image file that is gone
        return request_json(req) | {"ok": False, "error": f"{type(e).__name__}: {e}"}
    row = request_json(req) | {
        "ok": r.ok,
        "error": r.error,
        "answer": _clip(r.content),
        "reasoning": _clip(r.reasoning) or None,
        "ttft_ms": _round(r.ttft_ms),
        "e2e_ms": _round(r.e2e_ms),
        "prompt_n": r.prompt_n,
        "predicted_n": r.predicted_n,
        "gen_tps": _round(r.gen_tps, 1),
        "stop": r.stop_type,
    }
    if r.ok and req.evaluator is not None:
        s = evaluate(req.evaluator, r.content, req)
        row |= {
            "score": round(s.value, 3) if s.has_objective else None,
            "passed": s.passed if s.has_objective else None,
            "detail": s.detail or None,
            "judge": s.pending_judge,
        }
    return row


def _round(v: float | None, digits: int = 0) -> float | None:
    return None if v is None else round(v, digits)


def _totals(rows: list[dict[str, Any]], t0: float) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    scored = [r for r in ok if r.get("score") is not None]
    e2e = [float(r["e2e_ms"]) for r in ok if r.get("e2e_ms") is not None]
    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    return {
        "sent": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "scored": len(scored),
        "passed": sum(1 for r in scored if r.get("passed")),
        "mean_score": round(statistics.fmean(r["score"] for r in scored), 3) if scored else None,
        "judge_pending": sum(1 for r in ok if r.get("judge")),
        "mean_e2e_ms": round(statistics.fmean(e2e)) if e2e else None,
        "mean_ttft_ms": round(statistics.fmean(ttft)) if ttft else None,
        "wall_s": round(time.perf_counter() - t0, 2),
    }

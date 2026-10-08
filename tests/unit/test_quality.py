"""Evaluators, suite sources (items, files, HF datasets) and quality aggregates."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import TypeAdapter, ValidationError

from llmbench.config.evaluators import EvaluatorSpec
from llmbench.config.loader import load_suite
from llmbench.config.suite import HfDataset, Suite
from llmbench.data import hf_datasets
from llmbench.data.requests import Request, requests_digest
from llmbench.data.sources import SuiteError, materialize
from llmbench.evaluation.deterministic import evaluate
from llmbench.results.models import RequestResult
from llmbench.results.stats import quality_metrics

SPEC = TypeAdapter[EvaluatorSpec](EvaluatorSpec)


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


ROOT = Path(__file__).parents[2]


def ev(spec: dict[str, Any], output: str, **req: Any) -> tuple[float, bool, str]:
    s = evaluate(SPEC.validate_python(spec), output, Request("r", "p", 8, **req))
    return s.value, s.passed, s.detail


# -- evaluators ---------------------------------------------------------------------------


def test_exact_with_normalization() -> None:
    assert ev({"type": "exact"}, "  Hola   mundo ", reference="Hola mundo")[1]
    assert not ev({"type": "exact"}, "hola mundo", reference="Hola mundo")[1]
    norm = ["strip", "lower", "strip_punct", "strip_accents"]
    assert ev({"type": "exact", "normalize": norm}, "¡Canción!", reference="cancion")[1]
    assert ev({"type": "exact"}, "x")[2] == "exact: no value and no reference"


def test_contains_and_not_contains() -> None:
    assert ev({"type": "contains", "values": ["Madrid", "2024"]}, "madrid en 2024") == (1, True, "")
    value, passed, detail = ev({"type": "contains", "values": ["a", "zz"]}, "abc")
    assert (value, passed) == (0.5, False)
    assert "zz" in detail
    assert ev({"type": "contains", "values": ["a", "zz"], "mode": "any"}, "abc")[1]
    assert not ev({"type": "not_contains", "values": ["Lunes"]}, "el lunes")[1]


def test_regex() -> None:
    assert ev({"type": "regex", "pattern": r"^.{0,10}$"}, "corto")[1]
    assert not ev({"type": "regex", "pattern": r"\d", "mode": "fullmatch"}, "12")[1]
    assert ev({"type": "regex", "pattern": "HOLA", "ignore_case": True}, "hola")[1]


def test_regex_extract_numeric_against_reference_pattern() -> None:
    spec = {
        "type": "regex_extract",
        "pattern": r"####\s*\$?(-?[\d,]*\.?\d+)",
        "reference_pattern": r"####\s*(-?[\d,]*\.?\d+)",
        "compare": "numeric",
    }
    ref = "She sells 16 - 3 - 4 = 9 eggs.\n#### 18"
    assert ev(spec, "steps...\n#### $18.00", reference=ref)[1]
    assert ev(spec, "#### 1,000", reference="#### 1000")[1]
    assert not ev(spec, "#### 17", reference=ref)[1]
    assert "not found" in ev(spec, "eighteen", reference=ref)[2]


def test_length_and_ratio() -> None:
    assert ev({"type": "length", "max_words": 3}, "uno dos tres")[1]
    assert "> 3" in ev({"type": "length", "max_words": 3}, "uno dos tres cuatro")[2]
    spec = {"type": "length", "ratio_to": "input", "max_ratio": 0.5}
    assert ev(spec, "dos palabras", input="una dos tres cuatro cinco")[1]
    assert not ev(spec, "una dos tres", input="una dos tres cuatro cinco")[1]
    assert "no input" in ev(spec, "x")[2]


def test_json_schema_accepts_fenced_json() -> None:
    spec = {
        "type": "json_schema",
        "schema": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}},
    }
    assert ev(spec, '```json\n{"n": 3}\n```')[1]
    assert "schema" in ev(spec, '{"n": "3"}')[2]
    assert "not JSON" in ev(spec, "nope")[2]


def test_composites_and_judge_parts() -> None:
    both = {"type": "all", "of": [{"type": "contains", "values": ["a"]}, {"type": "llm_judge"}]}
    s = evaluate(SPEC.validate_python(both), "a", Request("r", "p", 8))
    assert (s.value, s.passed, s.pending_judge, s.has_objective) == (1, True, True, True)
    judge_only = evaluate(SPEC.validate_python({"type": "llm_judge"}), "a", Request("r", "p", 8))
    assert (judge_only.has_objective, judge_only.pending_judge) == (False, True)
    weighted = {
        "type": "weighted",
        "pass_threshold": 0.7,
        "of": [
            {"weight": 3, "evaluator": {"type": "contains", "values": ["x"]}},
            {"weight": 1, "evaluator": {"type": "contains", "values": ["y"]}},
        ],
    }
    assert ev(weighted, "x") == (0.75, True, "missing ['y']")
    assert ev(
        {"type": "any", "of": [{"type": "regex", "pattern": "q"}, {"type": "exact", "value": "z"}]},
        "z",
    )[1]


def test_unknown_evaluator_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "vibes"})


# -- sources --------------------------------------------------------------------------------


def suite(**kw: Any) -> Suite:
    return Suite.model_validate({"name": "s"} | kw)


def test_exactly_one_source() -> None:
    with pytest.raises(ValidationError, match="exactly one source"):
        suite(source={})
    with pytest.raises(ValidationError, match="exactly one source"):
        suite(source={"items": [{"prompt": "a"}], "file": {"path": "x.jsonl"}})


def test_items_with_template_and_defaults() -> None:
    s = suite(
        endpoint="chat",
        system="sys",
        template="Q: {prompt} ({category})",
        max_tokens=64,
        evaluator={"type": "contains", "values": ["x"]},
        source={
            "items": [
                {"id": "a", "category": "c1", "prompt": "uno", "reference": "1"},
                {"prompt": "dos", "category": "c2", "max_tokens": 8, "system": "own",
                 "evaluator": {"type": "exact"}},
            ]
        },
    )  # fmt: skip
    reqs = materialize(s).requests
    assert [r.id for r in reqs] == ["s/a", "s/00001"]
    assert reqs[0].prompt == "Q: uno (c1)"
    assert (reqs[0].n_predict, reqs[1].n_predict) == (64, 8)
    assert reqs[0].messages() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q: uno (c1)"},
    ]
    assert reqs[1].system == "own"
    assert reqs[1].evaluator is not None
    assert reqs[1].evaluator.type == "exact"
    assert requests_digest(reqs) == requests_digest(materialize(s).requests)


def test_template_errors_are_explained() -> None:
    s = suite(template="{nope}", source={"items": [{"prompt": "a"}]})
    with pytest.raises(SuiteError, match="template error"):
        materialize(s)


@pytest.mark.parametrize("fmt", ["jsonl", "json", "csv", "parquet"])
def test_file_sources(tmp_path: Path, fmt: str) -> None:
    rows = [{"q": "uno", "a": "1", "cat": "x"}, {"q": "dos", "a": "2", "cat": "y"}]
    path = tmp_path / f"d.{fmt}"
    if fmt == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in rows))
    elif fmt == "json":
        path.write_text(json.dumps(rows))
    elif fmt == "csv":
        path.write_text("q,a,cat\nuno,1,x\ndos,2,y\n")
    else:
        write_parquet(rows, path)
    s = suite(
        source={"file": {"path": str(path)}},
        columns={"prompt": "q", "reference": "a", "category": "cat"},
    )
    reqs = materialize(s).requests
    assert [(r.prompt, r.reference, r.category) for r in reqs] == [
        ("uno", "1", "x"),
        ("dos", "2", "y"),
    ]


def test_missing_column(tmp_path: Path) -> None:
    (tmp_path / "d.jsonl").write_text('{"question": "a"}\n')
    s = suite(source={"file": {"path": str(tmp_path / "d.jsonl")}})
    with pytest.raises(SuiteError, match="column 'prompt' not found; columns: question"):
        materialize(s)


def test_suite_file_paths_are_relative_to_the_suite(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "d.jsonl").write_text('{"prompt": "hola"}\n')
    (tmp_path / "s.yaml").write_text("name: s\nsource: {file: {path: data/d.jsonl}}\n")
    assert materialize(load_suite(tmp_path / "s.yaml")).requests[0].prompt == "hola"


def test_hf_dataset_from_parquet_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        "main/test/0000.parquet": [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)],
        "main/train/0000.parquet": [{"question": "t", "answer": "#### 0"}],
    }
    for name, rows in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        write_parquet(rows, tmp_path / name)

    class Ref:
        ref = "refs/convert/parquet"
        target_commit = "f" * 40

    class Api:
        def list_repo_refs(self, repo: str, repo_type: str) -> Any:
            return type("Refs", (), {"converts": [Ref()]})()

        def list_repo_files(self, repo: str, repo_type: str, revision: str) -> list[str]:
            assert revision == "f" * 40
            return [".gitattributes", *files]

    monkeypatch.setattr(hf_datasets, "HfApi", Api)

    def download(repo: str, f: str, **_: object) -> str:
        return str(tmp_path / f)

    monkeypatch.setattr(hf_datasets, "hf_hub_download", download)
    ds = hf_datasets.load_hf_dataset(
        HfDataset(repo="o/gsm", config="main", split="test", limit=3, shuffle_seed=1)
    )
    assert ds.commit == "f" * 40
    assert len(ds.rows) == 3
    assert {r["question"] for r in ds.rows} <= {f"q{i}" for i in range(10)}
    with pytest.raises(hf_datasets.DatasetError, match="available: main/test, main/train"):
        hf_datasets.load_hf_dataset(HfDataset(repo="o/gsm", config="main", split="dev"))


def test_repo_suites_are_valid() -> None:
    for path in (ROOT / "suites").glob("*.yaml"):
        s = load_suite(path)
        if s.source.items is not None:
            assert materialize(s).requests


# -- aggregates --------------------------------------------------------------------------------


def test_quality_metrics() -> None:
    results = [
        RequestResult(
            "a", True, e2e_ms=1000, content="uno dos", category="x", score=1, passed=True
        ),
        RequestResult(
            "b", True, e2e_ms=1000, content="tres", category="x", score=0.5, passed=False
        ),
        RequestResult("c", True, e2e_ms=500, content="", category="y", judge_pending=True),
        RequestResult("d", False, error="boom"),
    ]
    m = quality_metrics(results)
    assert (m["scored"], m["score_mean"], m["pass_rate"], m["judge_pending"]) == (2, 0.75, 0.5, 1)
    assert m["score_mean:x"] == 0.75
    assert "score_mean:y" not in m
    assert m["words_per_s_mean"] == pytest.approx((2 + 1 + 0) / 3)

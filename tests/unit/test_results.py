from pathlib import Path

import pytest

from llmbench.config.experiment import BenchPhase
from llmbench.config.llama import LlamaConfig
from llmbench.engine import planner
from llmbench.engine.planner import fingerprint
from llmbench.llamacpp.bench import parse_jsonl
from llmbench.results.models import RunRecord
from llmbench.results.stats import annotate, cv
from llmbench.results.store import Store, sha256_file
from tests.conftest import LLAMACPP

BUILD: dict[str, object] = {"commit": "c811cb8", "backend": "cpu", "cmake": {}}


def fp(**overrides: object) -> str:
    args: dict[str, object] = {
        "build": BUILD,
        "model_sha256": "a" * 64,
        "config": LlamaConfig(),
        "bench": BenchPhase(),
    } | overrides
    return fingerprint(**args)  # type: ignore[arg-type]


def test_fingerprint_is_stable_and_sensitive() -> None:
    assert fp() == fp()
    assert fp() != fp(config=LlamaConfig.model_validate({"load": {"threads": 2}}))
    assert fp() != fp(build=BUILD | {"commit": "deadbeef"})
    assert fp() != fp(bench=BenchPhase(repetitions=3))


def test_fingerprint_ignores_non_measurement_settings() -> None:
    assert fp() == fp(bench=BenchPhase(timeout_s=10, unstable_cv=0.5))


def test_tests_per_run() -> None:
    assert planner.tests_per_run(BenchPhase()) == 2
    assert planner.tests_per_run(BenchPhase(n_prompt=[0, 512], n_gen=[128], n_depth=[0, 4096])) == 4
    assert planner.tests_per_run(BenchPhase(n_gen=[0], pg=[(512, 128)])) == 2


def test_cv_and_unstable_flag() -> None:
    assert cv([10.0]) == 0
    assert cv([10.0, 10.0, 10.0]) == 0
    m = parse_jsonl((LLAMACPP / "llama-bench.jsonl").read_text())[0]
    a = annotate(m, unstable_cv=0.05, throttled=False)
    # our CV matches llama-bench's own stddev/avg for the same samples
    assert a.cv == pytest.approx(m.stddev_ts / m.avg_ts, rel=0.01)
    assert annotate(m, unstable_cv=a.cv / 2, throttled=False).unstable
    assert not annotate(m, unstable_cv=a.cv * 2, throttled=False).unstable
    assert annotate(m, unstable_cv=a.cv * 2, throttled=True).unstable


def record(fingerprint: str, status: str = "completed") -> RunRecord:
    ms = [
        annotate(m, unstable_cv=0.1, throttled=False)
        for m in parse_jsonl((LLAMACPP / "llama-bench.jsonl").read_text())
    ]
    return RunRecord(
        fingerprint=fingerprint,
        experiment="exp",
        phase="bench",
        status=status,  # type: ignore[arg-type]
        build={"id": "b", **BUILD},
        model={"id": "m", "sha256": "a" * 64},
        variant={"load.threads": 2},
        config=LlamaConfig().model_dump(mode="json"),
        argv=["llama-bench"],
        ignored=[],
        env={},
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_s=1.0,
        telemetry={"temp_peak_c": 60.0, "max_rss_mb": 41.0},
        measurements=ms if status == "completed" else [],
    )


def test_store_roundtrip_and_resume(tmp_path: Path) -> None:
    store = Store(tmp_path / "b.sqlite")
    assert store.schema_version == 1
    store.save_run(record("f1", "failed"))
    assert not store.is_completed("f1")
    store.save_run(record("f1"))
    assert store.is_completed("f1")
    rows = store.summary_rows("exp")
    assert len(rows) == 4  # latest attempt only, one row per measurement
    assert {r["label"] for r in rows} == {"pp64", "tg16", "pp64@d32", "tg16@d32"}


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    Store(tmp_path / "b.sqlite").close()
    assert Store(tmp_path / "b.sqlite").schema_version == 1


def test_model_hash_cache(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 100)
    store = Store(tmp_path / "b.sqlite")
    assert store.model_sha256(model) == sha256_file(model)
    model.write_bytes(b"y" * 101)
    assert store.model_sha256(model) == sha256_file(model)

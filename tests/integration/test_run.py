"""`llb validate / plan / run` end to end, against a fake llama-bench that mimics c811cb8."""

import csv
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from llmbench.cli.app import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def out(r: Result) -> str:
    """CLI output with rich's line wrapping undone."""
    return " ".join(r.output.split())


def experiment(tmp_path: Path, bin_dir: Path, model: Path, extra: str = "") -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(
        f"""
version: 1
experiment: {{name: it}}
llama_cpp:
  builds: [{{id: fake, bin_dir: {bin_dir}}}]
models:
  - {{id: tiny, source: {model}}}
matrix:
  load.threads: [1, 2]
  kv.cache_type_k: [f16, q4_0]
phases:
  bench: {{n_prompt: [64], n_gen: [16], n_depth: [0, 32], repetitions: 3}}
hardware:
  cooldown: {{enabled: false}}
  telemetry: {{hz: 20, sources: [linux]}}
output: {{directory: {tmp_path / "out"}}}
{extra}"""
    )
    return p


def test_validate_and_plan(tmp_path: Path, fake_bin_dir: Path, tiny_model: Path) -> None:
    exp = experiment(tmp_path, fake_bin_dir, tiny_model)
    r = runner.invoke(app, ["validate", str(exp)])
    assert r.exit_code == 0, out(r)
    assert "all llama-bench flags supported" in out(r)

    r = runner.invoke(app, ["plan", str(exp)])
    assert r.exit_code == 0, out(r)
    assert "Total timed executions" in out(r)
    assert "48" in out(r)  # 4 variants x 4 tests x 3 repetitions


def test_validate_catches_unsupported_flag(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path
) -> None:
    exp = experiment(
        tmp_path, fake_bin_dir, tiny_model, "defaults: {extra_args: {bench: [-mmp, '0']}}"
    )
    r = runner.invoke(app, ["validate", str(exp)])
    assert r.exit_code == 1
    assert "lacks: -mmp" in out(r)


def test_run_records_failures_and_resumes(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path
) -> None:
    exp = experiment(tmp_path, fake_bin_dir, tiny_model)
    r = runner.invoke(app, ["run", "-y", str(exp)])
    # the fake fails like the real tool for q4_0 K cache: 2 of 4 runs fail, suite continues
    assert r.exit_code == 1, out(r)
    assert "2 completed, 2 failed" in out(r)
    assert "does not divide n_embd_head_k" in out(r)

    out_dir = tmp_path / "out"
    db = sqlite3.connect(out_dir / "benchmark.sqlite")
    statuses = [s for (s,) in db.execute("SELECT status FROM runs ORDER BY id")]
    assert statuses.count("completed") == 2
    assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 2 * 4
    tel = json.loads(
        db.execute("SELECT telemetry FROM runs WHERE status='completed'").fetchone()[0]
    )
    assert tel["max_rss_mb"] > 0

    rows = list(csv.DictReader((out_dir / "summary.csv").open()))
    assert {r["test"] for r in rows if r["status"] == "completed"} == {
        "pp64", "tg16", "pp64@d32", "tg16@d32",
    }  # fmt: skip
    assert len((out_dir / "runs.jsonl").read_text().splitlines()) == 4

    r = runner.invoke(app, ["run", "-y", "--resume", str(exp)])
    assert "0 completed, 2 failed, 2 skipped" in out(r)


def test_oom_kill_is_reported(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_BENCH_OOM", "1")
    exp = experiment(tmp_path, fake_bin_dir, tiny_model)
    r = runner.invoke(app, ["run", "-y", "--model", "tiny", str(exp)])
    assert "likely out of memory" in out(r)


def test_missing_model_does_not_crash(tmp_path: Path, fake_bin_dir: Path) -> None:
    exp = experiment(tmp_path, fake_bin_dir, tmp_path / "missing.gguf")
    r = runner.invoke(app, ["run", "-y", str(exp)])
    assert r.exit_code == 1
    assert "file not found" in out(r)


def test_hf_source_is_explained(tmp_path: Path, fake_bin_dir: Path) -> None:
    exp = experiment(tmp_path, fake_bin_dir, Path("x"))
    exp.write_text(exp.read_text().replace("source: x", "source: hf://org/repo:Q4_K_M"))
    r = runner.invoke(app, ["validate", str(exp)])
    assert "Phase 2" in out(r)

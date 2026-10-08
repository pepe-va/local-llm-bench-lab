"""`llb validate / plan / run` end to end, against a fake llama-bench that mimics c811cb8."""

import csv
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from llmbench.cli.app import app
from tests.helpers import FakeHub

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
    assert "total timed executions" in out(r)
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


def test_hf_model_is_downloaded_once_and_released(
    tmp_path: Path, fake_bin_dir: Path, hub: FakeHub
) -> None:
    exp = experiment(tmp_path, fake_bin_dir, Path("x"))
    exp.write_text(
        exp.read_text()
        .replace(
            "source: x",
            "source: 'hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M', retention: delete_after_run",
        )
        .replace("kv.cache_type_k: [f16, q4_0]", "kv.cache_type_k: [f16]")
    )
    r = runner.invoke(app, ["validate", str(exp)])
    assert "is not downloaded" in out(r)  # validate never downloads

    r = runner.invoke(app, ["run", "-y", str(exp)])
    assert r.exit_code == 0, out(r)
    assert hub.downloads == ["unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"]
    assert "removed from the HF cache" in out(r)
    assert hub.cached_files() == []

    db = sqlite3.connect(tmp_path / "out" / "benchmark.sqlite")
    model = json.loads(db.execute("SELECT model FROM runs LIMIT 1").fetchone()[0])
    assert model["repo"] == "unsloth/Qwen3-1.7B-GGUF"
    assert model["commit"] == hub.repos["unsloth/Qwen3-1.7B-GGUF"].commit
    # identity comes from the Hub's LFS sha256, not from hashing the file
    assert (
        model["sha256"]
        == hub.repos["unsloth/Qwen3-1.7B-GGUF"]
        .files[
            [f.name for f in hub.repos["unsloth/Qwen3-1.7B-GGUF"].files].index(
                "Qwen3-1.7B-Q4_K_M.gguf"
            )
        ]
        .sha256
    )
    assert model["gguf"]["architecture"] == "llama"


def test_models_cli(tmp_path: Path, hub: FakeHub) -> None:
    ref = "hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M"
    r = runner.invoke(app, ["models", "pull", ref])
    assert r.exit_code == 0, out(r)
    assert "(download" in out(r)

    r = runner.invoke(app, ["models", "inspect", ref])
    assert r.exit_code == 0, out(r)
    assert "architecture llama" in out(r)

    r = runner.invoke(app, ["models", "favorite", ref])
    assert r.exit_code == 0, out(r)
    r = runner.invoke(app, ["models", "ls"])
    assert "★" in out(r)
    assert "Qwen3-1.7B-Q4_K_M.gguf" in r.output.replace("\n", "").replace("│", "").replace(" ", "")

    r = runner.invoke(app, ["models", "unfavorite", "-y", ref])
    assert r.exit_code == 0, out(r)
    r = runner.invoke(app, ["models", "prune", "--dry-run"])
    assert "nothing to prune" in out(r)

    r = runner.invoke(app, ["models", "pull", "hf://unsloth/Qwen3-1.7B-GGUF:Q9_X"])
    assert r.exit_code == 1
    assert "no GGUF matches" in out(r)


# -- latency phase (fake llama-server) ------------------------------------------------------


def latency_experiment(tmp_path: Path, bin_dir: Path, model: Path, extra: str = "") -> Path:
    (tmp_path / "suites").mkdir(exist_ok=True)
    (tmp_path / "suites" / "syn.yaml").write_text(
        "name: syn\nsource:\n  synthetic: {prompt_tokens: {dist: uniform, min: 20, max: 90},"
        " output_tokens: 6, count: 4}\n"
    )
    p = tmp_path / "lat.yaml"
    p.write_text(
        f"""
version: 1
experiment: {{name: lat}}
llama_cpp: {{builds: [{{id: fake, bin_dir: {bin_dir}}}]}}
models: [{{id: tiny, source: {model}}}]
matrix:
  load.threads: [1, 2]
  sampling.temperature: [0, 0.7]
phases:
  bench: {{enabled: false}}
  latency: {{enabled: true, suites: [suites/syn.yaml], passes: 2, warmup_requests: 1}}
hardware: {{cooldown: {{enabled: false}}, telemetry: {{hz: 20, sources: [linux]}}}}
output: {{directory: {tmp_path / "out"}}}
{extra}"""
    )
    return p


def test_latency_run_reuses_servers_and_records_requests(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = tmp_path / "starts.txt"
    monkeypatch.setenv("FAKE_SERVER_STARTS", str(starts))
    exp = latency_experiment(tmp_path, fake_bin_dir, tiny_model)

    r = runner.invoke(app, ["plan", str(exp)])
    assert "llama-server starts 2" in out(r)
    assert "timed requests 32" in out(r)

    r = runner.invoke(app, ["run", "-y", str(exp)])
    assert r.exit_code == 0, out(r)
    assert "4 completed, 0 failed" in out(r)
    assert "TTFT p50" in out(r)
    assert len(starts.read_text().splitlines()) == 2  # temperature variants reuse the server
    assert all("--host 127.0.0.1" in line for line in starts.read_text().splitlines())

    db = sqlite3.connect(tmp_path / "out" / "benchmark.sqlite")
    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4 * 4 * 2
    assert db.execute("SELECT COUNT(*) FROM requests WHERE cache_n != 0").fetchone()[0] == 0
    prompt_ns = {n for (n,) in db.execute("SELECT prompt_n FROM requests")}
    assert all(20 <= n <= 90 for n in prompt_ns)  # exact synthetic sizes, BOS included
    rows = list(csv.DictReader((tmp_path / "out" / "latency.csv").open()))
    assert len(rows) == 4
    assert float(rows[0]["ttft_ms_p50"]) > 0
    assert rows[0]["requests"] == "8.0"

    r = runner.invoke(app, ["run", "-y", "--resume", str(exp)])
    assert "0 completed, 0 failed, 4 skipped" in out(r)
    assert len(starts.read_text().splitlines()) == 2  # nothing restarted


def test_server_start_failure_is_recorded_once_per_server(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = tmp_path / "starts.txt"
    monkeypatch.setenv("FAKE_SERVER_STARTS", str(starts))
    monkeypatch.setenv("FAKE_SERVER_FAIL_START", "1")
    r = runner.invoke(
        app, ["run", "-y", str(latency_experiment(tmp_path, fake_bin_dir, tiny_model))]
    )
    assert r.exit_code == 1
    assert "0 completed, 4 failed" in out(r)
    assert "tensor data is not within file bounds" in out(r)
    assert len(starts.read_text().splitlines()) == 2  # one attempt per server key, no retries


def test_validate_checks_server_flags(tmp_path: Path, fake_bin_dir: Path, tiny_model: Path) -> None:
    exp = latency_experiment(
        tmp_path, fake_bin_dir, tiny_model, "defaults: {extra_args: {server: [--draft-max, '8']}}"
    )
    r = runner.invoke(app, ["validate", str(exp)])
    assert r.exit_code == 1
    assert "llama-server lacks: --draft-max" in out(r)
    assert "suite syn" in out(r)


def test_quality_run_scores_chat_answers(
    tmp_path: Path, fake_bin_dir: Path, tiny_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = tmp_path / "starts.txt"
    monkeypatch.setenv("FAKE_SERVER_STARTS", str(starts))
    monkeypatch.setenv("FAKE_CHAT_ANSWER", "La capital es Madrid.")
    (tmp_path / "q.yaml").write_text(
        """
name: q
endpoint: chat
system: Responde breve.
source:
  items:
    - {id: ok, category: geo, prompt: "¿España?", evaluator: {type: contains, values: [Madrid]}}
    - {id: ko, category: geo, prompt: "¿Francia?", evaluator: {type: contains, values: [París]}}
    - {id: judged, category: estilo, prompt: "Escribe algo", evaluator: {type: llm_judge}}
"""
    )
    exp = latency_experiment(tmp_path, fake_bin_dir, tiny_model)
    exp.write_text(
        exp.read_text()
        .replace("sampling.temperature: [0, 0.7]", "sampling.temperature: [0]")
        .replace(
            "passes: 2, warmup_requests: 1}",
            "passes: 1, warmup_requests: 1}\n  quality: {enabled: true, suites: [q.yaml]}",
        )
    )
    r = runner.invoke(app, ["run", "-y", str(exp)])
    assert r.exit_code == 0, out(r)
    assert "score 0.500 · pass 50% · 2 scored · 1 awaiting judge" in out(r)
    # latency and quality runs of the same config share one server
    assert len(starts.read_text().splitlines()) == 2

    db = sqlite3.connect(tmp_path / "out" / "benchmark.sqlite")
    rows = db.execute(
        "SELECT request_id, score, passed, judge_pending, eval_detail FROM requests "
        "WHERE run_id = (SELECT MIN(id) FROM runs WHERE phase = 'quality') ORDER BY request_id"
    ).fetchall()
    assert rows[0][:4] == ("q/judged#0", None, None, 1)
    assert rows[1][:3] == ("q/ko#0", 0.0, 0)
    assert "París" in rows[1][4]
    assert rows[2][:3] == ("q/ok#0", 1.0, 1)
    quality = list(csv.DictReader((tmp_path / "out" / "quality.csv").open()))
    assert quality[0]["score_mean:geo"] == "0.5"

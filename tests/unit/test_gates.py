"""Analysis view, gates, winner selection, deployment command and report rendering."""

from pathlib import Path
from typing import Any

import pytest

from llmbench.analysis.view import ConfigRow, build_view
from llmbench.config.experiment import Gate, Select
from llmbench.config.llama import LlamaConfig
from llmbench.gates.deploy import deployment
from llmbench.gates.rules import check, select
from llmbench.llamacpp.argv import sampling_flags
from llmbench.llamacpp.introspect import parse_help, unknown_flags
from llmbench.report.data import collect
from llmbench.report.html import render_html, render_png
from llmbench.report.markdown import render_guidance, render_markdown
from llmbench.results.models import Measurement, RunRecord
from llmbench.results.store import Store
from tests.conftest import LLAMACPP


def row(model: str, **values: list[float]) -> ConfigRow:
    r = ConfigRow("b", "c" * 7, model, {"load.threads": 4}, {}, {})
    r.values = dict(values)
    return r


def test_gates_and_selection() -> None:
    rows = [
        row("fast", **{"tps:tg128": [10.0, 12.0], "ttft_ms_p95": [900.0]}),
        row("slow", **{"tps:tg128": [4.0], "ttft_ms_p95": [300.0]}),
        row("nodata"),
    ]
    gates = [
        Gate(metric="tps:tg128", op=">=", value=5),
        Gate(metric="ttft_ms_p95", op="<=", value=1000, agg="max"),
    ]
    verdicts = check(gates, rows)
    assert [v.passed for v in verdicts] == [True, False, False]
    assert verdicts[1].reasons == ["tps:tg128 (mean) >= 5: 4"]
    assert "missing" in verdicts[2].reasons[0]
    assert select(verdicts, Select(metric="tps:tg128")) is verdicts[0]
    assert (
        select(verdicts, Select(metric="ttft_ms_p95", objective="minimize", among="all"))
        is verdicts[1]
    )
    assert select(verdicts, None) is verdicts[0]  # the only one that passed


def test_metric_aggregations() -> None:
    r = row("m", x=[1.0, 2.0, 6.0])
    assert (r.metric("x"), r.metric("x", "median"), r.metric("x", "min"), r.metric("x", "max")) == (
        3,
        2,
        1,
        6,
    )
    assert r.metric("missing") is None


def test_sampling_flags_exist_in_the_real_server() -> None:
    body = {
        "temperature": 0.2, "top_k": 20, "top_p": 0.9, "min_p": 0.05, "typical_p": 1.0,
        "top_n_sigma": 1.0, "xtc_probability": 0.1, "xtc_threshold": 0.2, "repeat_penalty": 1.1,
        "repeat_last_n": 64, "presence_penalty": 0.1, "frequency_penalty": 0.1,
        "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64,
        "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "dynatemp_range": 0.1,
        "dynatemp_exponent": 1.0, "seed": 42, "samplers": ["top_k", "temperature"],
        "ignore_eos": True, "grammar": "root ::= x",
    }  # fmt: skip
    argv = sampling_flags(body)
    supported = parse_help((LLAMACPP / "llama-server.help.txt").read_text())
    assert unknown_flags(["x", *argv.args], supported) == []
    assert argv.ignored == ["sampling.grammar"]


def record(fp: str, variant: dict[str, Any], phase: str, **kw: Any) -> RunRecord:
    cfg = LlamaConfig.model_validate(
        {"load": {"threads": variant.get("load.threads", 4)},
         "sampling": {"temperature": variant.get("sampling.temperature", 0)}}
    ).model_dump(mode="json")  # fmt: skip
    return RunRecord(
        fingerprint=fp, experiment="exp", phase=phase, status="completed",
        build={"id": "b", "commit": "c" * 40, "backend": "cpu", "cmake": {}},
        model={"id": "m", "sha256": "s", "path": "/models/m.gguf", "gguf": {"quant": "Q4_K_M"}},
        variant=variant, config=cfg, argv=["/opt/llama/llama-bench"], ignored=[],
        env={"cpu_model": "Cortex-A76", "os": "Pi OS", "cpu_count": 4}, started_at="t",
        finished_at="t", duration_s=1.0, telemetry={"max_rss_mb": 900.0, "temp_peak_c": 61.0},
        **kw,
    )  # fmt: skip


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "benchmark.sqlite")
    m = Measurement(
        "tg", 0, 128, 0, 9.5, 0.2, 1, 1, [9.4, 9.6], {}, 9.5, 0.02, False,
        iqr_ts=0.1, ci_low_ts=9.4, ci_high_ts=9.6,
    )  # fmt: skip
    for threads in (2, 4):
        # the planner keeps the first variant of each bench-identical group
        bench_variant = {"load.threads": threads, "sampling.temperature": 0}
        s.save_run(record(f"b{threads}", bench_variant, "bench", measurements=[m]))
        for temp in (0, 0.7):
            v = {"load.threads": threads, "sampling.temperature": temp}
            s.save_run(
                record(
                    f"l{threads}{temp}", v, "latency", suite="syn",
                    metrics={"ttft_ms_p95": 100.0 * threads, "gen_tps_mean": 10.0 - threads},
                )
            )  # fmt: skip
    return s


def test_view_shares_bench_results_with_sampling_variants(store: Store) -> None:
    rows = {
        (r.variant["load.threads"], r.variant.get("sampling.temperature")): r
        for r in build_view(store, "exp")
    }
    assert rows[(2, 0.7)].metric("tps:tg128") == 9.5  # bench ran once for threads=2
    assert rows[(2, 0.7)].metric("tps_ci_low:tg128") == 9.4
    assert rows[(2, 0.7)].metric("ttft_ms_p95@syn") == 200
    assert rows[(4, 0)].metric("peak_rss_mb", "max") == 900


def test_reports_and_guidance(store: Store, tmp_path: Path) -> None:
    gates = [Gate(metric="ttft_ms_p95", op="<=", value=300, agg="max")]
    d = collect(store, "exp", gates, Select(metric="gen_tps_mean"))
    assert sum(v.passed for v in d.verdicts) == 2
    assert d.winner is not None
    assert d.winner.row.variant["load.threads"] == 2
    assert d.deploy is not None
    assert d.deploy.argv[:3] == ["/opt/llama/llama-server", "-m", "/models/m.gguf"]
    assert "-t 2" in d.deploy.command
    assert "ExecStart=/opt/llama/llama-server" in d.deploy.systemd_unit()

    md = render_markdown(d)
    assert "## Gates" in md
    assert "| m | Q4_K_M |" in md
    assert "| 9.50 | 0.20 | [9.40, 9.60] | 9.50 | 0.10 |" in md
    guidance = render_guidance(d)
    assert "**2 of 4** configurations pass" in guidance
    assert "127.0.0.1" in guidance

    html = render_html(d)
    assert "Plotly" in html  # library embedded: works offline
    assert "Time to first token" in html
    assert html.count("<table") >= 2
    assert "[9.40, 9.60]" in html
    assert "whiskers: 95% CI" in html
    pngs = render_png(d, tmp_path / "charts")
    assert {p.name for p in pngs} == {"bench.png", "latency.png"}
    assert all(p.stat().st_size > 1000 for p in pngs)


def test_deployment_without_server_runs_uses_bench_binary_dir(store: Store) -> None:
    rows = build_view(store, "exp")
    bench_only = ConfigRow(
        "b", "c", "m", {}, {"path": "/m.gguf"}, rows[0].config, runs=[rows[0].runs[0]]
    )
    assert deployment(bench_only).argv[0] == "/opt/llama/llama-server"

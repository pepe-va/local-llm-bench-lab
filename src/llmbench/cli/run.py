from pathlib import Path
from typing import Annotated

import typer

from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.cli.plan import ModelOpt, print_plan
from llmbench.config.experiment import Experiment
from llmbench.config.matrix import variant_label
from llmbench.engine.env import PreflightError
from llmbench.engine.executor import execute
from llmbench.engine.planner import RunSpec, make_plan
from llmbench.engine.runtime import output_dir
from llmbench.report.data import collect
from llmbench.report.html import MissingExtraError
from llmbench.report.write import write_guidance, write_reports
from llmbench.results.models import RunRecord
from llmbench.results.store import Store


class RichEvents:
    def log(self, message: str) -> None:
        console.print(f"[dim]{message}[/]", highlight=False)

    def run_start(self, index: int, total: int, spec: RunSpec) -> None:
        console.print(
            f"\n[bold][{index}/{total}][/] {spec.phase} · {spec.model.id} · {spec.build.id} · "
            f"{variant_label(spec.variant)}" + (f" · {spec.suite.name}" if spec.suite else ""),
            highlight=False,
        )

    def cooling(self, temp_c: float, elapsed_s: float) -> None:
        console.print(f"  cooling down: {temp_c:.1f} °C ({elapsed_s:.0f}s)", highlight=False)

    def run_end(self, index: int, total: int, record: RunRecord) -> None:
        if record.status != "completed":
            lines = (record.error or "").strip().splitlines()
            console.print(
                f"  [red]✗ {record.status}[/]: {lines[0] if lines else ''}", highlight=False
            )
            for line in lines[1:3]:
                console.print(f"    [dim]{line}[/]", highlight=False)
            return
        if record.phase == "quality" and (
            record.metrics.get("scored") or record.metrics.get("judge_pending")
        ):
            m = record.metrics
            cats = [
                f"{k.split(':', 1)[1]} {v:.2f}"
                for k, v in m.items()
                if k.startswith("score_mean:") and v is not None
            ]
            console.print(
                f"  score {_f(m.get('score_mean'), 3)} · pass {_pct(m.get('pass_rate'))}"
                f" · {int(m.get('scored') or 0)} scored"
                + (
                    f" · {int(m['judge_pending'] or 0)} awaiting judge"
                    if m.get("judge_pending")
                    else ""
                ),
                highlight=False,
            )
            if cats:
                console.print(f"  [dim]{' · '.join(cats)}[/]", highlight=False)
        m = record.metrics
        if m.get("spearman") is not None or m.get("cosine_mean") is not None:
            console.print(
                f"  spearman {_f(m.get('spearman'), 3)} · pearson {_f(m.get('pearson'), 3)}"
                f" · {_f(m.get('embed_tps_mean'))} tokens/s embedded",
                highlight=False,
            )
        ndcg = next((v for k, v in m.items() if k.startswith("ndcg@")), None)
        if ndcg is not None:
            console.print(
                f"  nDCG {_f(ndcg, 3)} · MRR {_f(m.get('mrr'), 3)}"
                f" · top-1 {_pct(m.get('top1_accuracy'))}"
                f" · {_f(m.get('rerank_tps_mean'))} tokens/s",
                highlight=False,
            )
        if record.phase in ("latency", "quality") and m.get("ttft_ms_p50") is not None:
            console.print(
                f"  TTFT p50 {_f(m.get('ttft_ms_p50'))} ms · p95 {_f(m.get('ttft_ms_p95'))} ms"
                f"   ITL p50 {_f(m.get('itl_ms_p50'))} ms · p95 {_f(m.get('itl_ms_p95'))} ms",
                highlight=False,
            )
            console.print(
                f"  prompt {_f(m.get('prompt_tps_mean'))} t/s · gen {_f(m.get('gen_tps_mean'))} t/s"
                f" · e2e p50 {_f(m.get('e2e_ms_p50'))} ms · {int(m.get('requests') or 0)} requests",
                highlight=False,
            )
        for m in record.measurements:
            flag = " [yellow](unstable)[/]" if m.unstable else ""
            console.print(
                f"  {m.label:<14} {m.avg_ts:10.2f} t/s ± {m.stddev_ts:.2f}  cv {m.cv:.1%}{flag}",
                highlight=False,
            )
        console.print(f"  [dim]{_telemetry_line(record)}[/]", highlight=False)

    def run_skipped(self, index: int, total: int, spec: RunSpec, reason: str) -> None:
        console.print(f"[dim][{index}/{total}] skip {spec.model.id} · {reason}[/]")


def run(
    experiment: ExperimentArg,
    model: ModelOpt = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip runs already completed (by fingerprint).")
    ] = False,
    no_build: Annotated[
        bool, typer.Option("--no-build", help="Fail instead of compiling missing builds.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Run an experiment."""
    exp = load(experiment)
    try:
        p = make_plan(exp, model_ids=model)
    except ValueError as e:
        fail(str(e))
    print_plan(p, exp.experiment.name, str(output_dir(exp)))
    if not p.runs:
        fail("nothing to run (no enabled models or bench phase disabled)")
    if not yes and not typer.confirm("Start?", default=True):
        raise typer.Exit(1)

    try:
        s = execute(exp, p, RichEvents(), resume=resume, auto_build=not no_build)
    except PreflightError as e:
        fail(f"preflight: {e}")
    console.print(
        f"\n[bold]done[/]: {s.completed} completed, {s.failed} failed, {s.skipped} skipped"
        + (" · [yellow]interrupted (use --resume)[/]" if s.interrupted else "")
    )
    console.print(f"results: {s.output_dir}")
    gates_failed = _reports(exp, s.output_dir)
    if s.interrupted:
        raise typer.Exit(130)
    if s.failed or gates_failed:
        raise typer.Exit(1)


def _reports(exp: Experiment, out: Path) -> bool:
    """Write the configured reports and, with gates, guidance.md. True if no config passes."""
    db = out / "benchmark.sqlite"
    if not db.is_file():
        return False
    store = Store(db)
    try:
        d = collect(store, exp.experiment.name, exp.gates, exp.select)
    finally:
        store.close()
    try:
        for path in write_reports(d, out, exp.output.report):
            console.print(f"report: {path}", highlight=False)
    except MissingExtraError as e:
        console.print(f"[yellow]report skipped[/]: {e}", highlight=False)
    if not exp.gates:
        return False
    path = write_guidance(d, out)
    passed = sum(1 for v in d.verdicts if v.passed)
    winner = f" · winner: {d.winner.row.label}" if d.winner else ""
    console.print(f"gates: {passed}/{len(d.verdicts)} configurations pass{winner} → {path}")
    return passed == 0


def _f(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _telemetry_line(record: RunRecord) -> str:
    tel = record.telemetry or {}
    parts: list[str] = []
    if tel.get("temp_peak_c") is not None:
        parts.append(f"temp {tel.get('temp_before_c')} → peak {tel['temp_peak_c']} °C")
    if tel.get("max_rss_mb") is not None:
        parts.append(f"peak RSS {tel['max_rss_mb']:.0f} MB")
    elif record.metrics.get("server_peak_rss_mb") is not None:
        parts.append(f"server peak RSS {record.metrics['server_peak_rss_mb']:.0f} MB")
    parts.append(f"{record.duration_s:.2f}s")
    return " · ".join(parts)

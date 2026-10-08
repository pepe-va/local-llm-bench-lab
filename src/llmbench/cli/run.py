from typing import Annotated

import typer

from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.cli.plan import ModelOpt, print_plan
from llmbench.config.matrix import variant_label
from llmbench.engine.env import PreflightError
from llmbench.engine.executor import execute, output_dir
from llmbench.engine.planner import RunSpec, make_plan
from llmbench.results.models import RunRecord


class RichEvents:
    def log(self, message: str) -> None:
        console.print(f"[dim]{message}[/]", highlight=False)

    def run_start(self, index: int, total: int, spec: RunSpec) -> None:
        console.print(
            f"\n[bold][{index}/{total}][/] {spec.model.id} · {spec.build.id} · "
            f"{variant_label(spec.variant)}",
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
    if s.interrupted:
        raise typer.Exit(130)
    if s.failed:
        raise typer.Exit(1)


def _telemetry_line(record: RunRecord) -> str:
    tel = record.telemetry or {}
    parts: list[str] = []
    if tel.get("temp_peak_c") is not None:
        parts.append(f"temp {tel.get('temp_before_c')} → peak {tel['temp_peak_c']} °C")
    if tel.get("max_rss_mb") is not None:
        parts.append(f"peak RSS {tel['max_rss_mb']:.0f} MB")
    parts.append(f"{record.duration_s:.2f}s")
    return " · ".join(parts)

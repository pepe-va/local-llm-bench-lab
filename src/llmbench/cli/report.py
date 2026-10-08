from pathlib import Path
from typing import Annotated

import typer

from llmbench.cli.common import console, fail, load
from llmbench.config.experiment import Experiment
from llmbench.engine.runtime import output_dir
from llmbench.report.data import collect
from llmbench.report.html import MissingExtraError
from llmbench.report.markdown import gates_table
from llmbench.report.write import write_guidance, write_reports
from llmbench.results.store import Store

TargetArg = Annotated[
    Path, typer.Argument(help="Experiment YAML, a results directory, or a benchmark.sqlite.")
]


def _open(target: Path) -> tuple[Store, Experiment | None, Path]:
    """(store, experiment if a YAML was given, default output directory)."""
    if target.suffix in (".yaml", ".yml"):
        exp = load(target)
        out = output_dir(exp)
        db = out / "benchmark.sqlite"
    else:
        exp = None
        db = target / "benchmark.sqlite" if target.is_dir() else target
        out = db.parent
    if not db.is_file():
        fail(f"no results at {db} (run the experiment first)")
    return Store(db), exp, out


def report(
    target: TargetArg,
    fmt: Annotated[
        list[str] | None, typer.Option("--format", "-f", help="html, md, png (repeatable).")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory.")] = None,
    experiment: Annotated[
        str | None, typer.Option(help="Experiment name (when the store has several).")
    ] = None,
) -> None:
    """Build reports (HTML with charts, Markdown, PNG) from stored results."""
    store, exp, default_out = _open(target)
    names = [exp.experiment.name] if exp else ([experiment] if experiment else store.experiments())
    formats = fmt or ["html", "md"]
    try:
        for name in names:
            d = collect(store, name, exp.gates if exp else None, exp.select if exp else None)
            dest = (out or default_out) if len(names) == 1 else (out or default_out) / name
            for path in write_reports(d, dest, formats):
                console.print(f"[green]✓[/] {path}", highlight=False)
    except MissingExtraError as e:
        fail(str(e))
    finally:
        store.close()


def guide(
    experiment: Annotated[Path, typer.Argument(help="Experiment YAML (with gates).")],
) -> None:
    """Apply the experiment's gates, pick the winner and write guidance.md (exit 1 if none pass)."""
    store, exp, out = _open(experiment)
    assert exp is not None
    try:
        d = collect(store, exp.experiment.name, exp.gates, exp.select)
    finally:
        store.close()
    if not d.rows:
        fail("no results for this experiment yet")
    path = write_guidance(d, out)
    console.print(gates_table(d), highlight=False)
    if d.winner and d.deploy:
        console.print(f"\n[bold green]winner[/]: {d.winner.row.label}", highlight=False)
        console.print(d.deploy.command, highlight=False, soft_wrap=True)
    console.print(f"\n[green]✓[/] {path}", highlight=False)
    if not any(v.passed for v in d.verdicts):
        raise typer.Exit(1)

from typing import Annotated

import typer
from rich.table import Table

from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.config.matrix import variant_label
from llmbench.engine.planner import Plan, make_plan
from llmbench.engine.runtime import output_dir

ModelOpt = Annotated[
    list[str] | None, typer.Option("--model", "-m", help="Only these model ids (repeatable).")
]


def plan(experiment: ExperimentArg, model: ModelOpt = None) -> None:
    """Show what `run` would execute, without running anything."""
    exp = load(experiment)
    try:
        p = make_plan(exp, model_ids=model)
    except ValueError as e:
        fail(str(e))
    print_plan(p, exp.experiment.name, str(output_dir(exp)))
    if p.variants > 1:
        t = Table("#", "variant", title="Matrix", title_justify="left", box=None)
        seen: list[str] = []
        for r in p.runs:
            label = variant_label(r.variant) + (f" · {r.phase}" if p.latency_runs else "")
            if label not in seen:
                seen.append(label)
                t.add_row(str(len(seen)), label)
        console.print(t)


def print_plan(p: Plan, name: str, out: str) -> None:
    t = Table(box=None, show_header=False, title=f"Experiment {name}", title_justify="left")
    t.add_row("Builds", str(p.builds))
    t.add_row("Models", str(p.models))
    t.add_row("Configurations", str(p.variants))
    if p.bench_runs:
        t.add_row("[bold]bench[/] runs (llama-bench calls)", f"[bold]{len(p.bench_runs)}[/]")
        t.add_row("  tests per run x repetitions", f"{p.tests_per_run} x {p.repetitions}")
        t.add_row("  total timed executions", f"[bold]{p.total_executions}[/]")
    if p.latency_runs:
        t.add_row("[bold]latency[/] runs", f"[bold]{len(p.latency_runs)}[/]")
        t.add_row("  suites", ", ".join(s.name for s in p.suites))
        t.add_row("  llama-server starts", str(p.servers))
        t.add_row("  timed requests", f"[bold]{p.total_requests}[/]")
    t.add_row("Output", out)
    console.print(t)

from typing import Annotated

import typer
from rich.table import Table

from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.config.matrix import variant_label
from llmbench.engine.executor import output_dir
from llmbench.engine.planner import Plan, make_plan

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
            label = variant_label(r.variant)
            if label not in seen:
                seen.append(label)
                t.add_row(str(len(seen)), label)
        console.print(t)


def print_plan(p: Plan, name: str, out: str) -> None:
    t = Table(box=None, show_header=False, title=f"Experiment {name}", title_justify="left")
    t.add_row("Builds", str(p.builds))
    t.add_row("Models", str(p.models))
    t.add_row("Configurations", str(p.variants))
    t.add_row("Runs (llama-bench calls)", f"[bold]{len(p.runs)}[/]")
    t.add_row("Tests per run", str(p.tests_per_run))
    t.add_row("Repetitions", str(p.repetitions))
    t.add_row("Total timed executions", f"[bold]{p.total_executions}[/]")
    t.add_row("Output", out)
    console.print(t)

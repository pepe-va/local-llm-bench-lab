from pathlib import Path
from typing import Annotated

import typer

from llmbench.cli.common import console, fail, load
from llmbench.cli.run import RichEvents
from llmbench.engine.judge import run_judge
from llmbench.engine.runtime import output_dir
from llmbench.results import export
from llmbench.results.store import Store


def judge(
    experiment: Annotated[Path, typer.Argument(help="Experiment YAML with `phases.judge`.")],
    force: Annotated[bool, typer.Option(help="Judge every judgeable response again.")] = False,
) -> None:
    """Run the LLM-judge pass over responses awaiting it (scores are stored apart)."""
    exp = load(experiment)
    if exp.phases.judge.model is None:
        fail("set phases.judge.model (and enabled: true) in the experiment")
    out = output_dir(exp)
    db = out / "benchmark.sqlite"
    if not db.is_file():
        fail(f"no results at {db}")
    store = Store(db)
    try:
        if force:
            console.print(f"reset {store.reset_judgements(exp.experiment.name)} response(s)")
        s = run_judge(exp, store, RichEvents())
        if "csv" in exp.output.formats:
            export.write_quality_csv(out / "quality.csv", store, exp.experiment.name)
    finally:
        store.close()
    if s.error:
        fail(f"judge: {s.error}", code=1)
    console.print(f"judged {s.judged} response(s) in {s.runs} run(s); {s.failed} unparseable")

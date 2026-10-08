"""Helpers shared by CLI commands."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from llmbench.config.experiment import Experiment
from llmbench.config.loader import ConfigError, load_experiment

console = Console()
err = Console(stderr=True)

ExperimentArg = Annotated[
    Path, typer.Argument(help="Experiment YAML.", exists=False, dir_okay=False)
]


def fail(message: str, code: int = 2) -> NoReturn:
    err.print(f"[red]{message}[/]", highlight=False)
    raise typer.Exit(code)


def load(path: Path) -> Experiment:
    try:
        return load_experiment(path)
    except ConfigError as e:
        fail(str(e))

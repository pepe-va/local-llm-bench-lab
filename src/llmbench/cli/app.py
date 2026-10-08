"""Root `llb` command. Subcommands are registered here as they land."""

import platform
from typing import Annotated

import typer
from rich.console import Console

from llmbench import __version__

app = typer.Typer(
    name="llb",
    help="Reproducible benchmarking lab for local LLMs on llama.cpp.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"llb {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    """local-llm-bench-lab."""


@app.command()
def doctor() -> None:
    """Show the environment llb would benchmark on."""
    console.print(f"[bold]llb[/] {__version__}")
    console.print(f"python    {platform.python_version()}")
    console.print(f"machine   {platform.machine()}")
    console.print(f"system    {platform.system()} {platform.release()}")

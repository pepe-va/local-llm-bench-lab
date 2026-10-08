from pathlib import Path
from typing import Annotated

import typer

from llmbench.cli.common import console, fail


def serve(
    results: Annotated[
        list[Path] | None,
        typer.Option("--results", "-r", help="Results dir / sqlite / parent dir (repeatable)."),
    ] = None,
    experiment: Annotated[
        list[Path] | None,
        typer.Option(
            "--experiment", "-e", help="Experiment YAML whose gates to show (repeatable)."
        ),
    ] = None,
    host: Annotated[str, typer.Option(help="Bind address (keep it local).")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
) -> None:
    """Read-only web dashboard over stored results (extra `serve`)."""
    try:
        import uvicorn  # noqa: PLC0415

        from llmbench.serve.app import create_app  # noqa: PLC0415
    except ImportError:
        fail("the dashboard needs the `serve` extra: `uv sync --extra serve`")
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]warning[/]: listening on {host}. The dashboard has no authentication: "
            "use only on a trusted network, never exposed to the Internet."
        )
    console.print(f"llb dashboard on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(create_app(results, experiment), host=host, port=port, log_level="warning")

from pathlib import Path
from typing import Annotated

import typer

from llmbench.cli.common import console, fail


def lab(
    results: Annotated[
        list[Path] | None,
        typer.Option("--results", "-r", help="Results dir / sqlite / parent dir (repeatable)."),
    ] = None,
    llama_server: Annotated[
        Path | None,
        typer.Option(help="A llama-server binary for the chat, besides the managed builds."),
    ] = None,
    experiments: Annotated[
        Path,
        typer.Option(help="Folder of experiment YAMLs the test tab edits (with `_profiles/`)."),
    ] = Path("experiments"),
    host: Annotated[str, typer.Option(help="Bind address (keep it local).")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
) -> None:
    """Web lab: download models from Hugging Face, chat with them and test them (extra `serve`).

    Run it from the repository root: commands run there, like `llb` in a terminal.
    Includes the results dashboard of `llb serve`.
    """
    try:
        import uvicorn  # noqa: PLC0415

        from llmbench.lab.app import create_lab_app  # noqa: PLC0415
        from llmbench.lab.state import Lab  # noqa: PLC0415
    except ImportError:
        fail("the lab needs the `serve` extra: `uv sync --extra serve`")
    if llama_server is not None and not llama_server.is_file():
        fail(f"no such file: {llama_server}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]warning[/]: listening on {host}. The lab has no authentication and can "
            "download models and run commands: never expose it beyond a network you trust. "
            "Prefer 127.0.0.1 behind an SSH tunnel or `tailscale serve`."
        )
    root = Path.cwd()
    folder = experiments if experiments.is_absolute() else root / experiments
    if not (folder / "_profiles").is_dir():
        console.print(
            f"[yellow]warning[/]: no {folder}/_profiles: new experiments start without a "
            "machine profile. Start `llb lab` from the repository root."
        )
    the_lab = Lab(
        root=root,
        llama_server=llama_server.resolve() if llama_server else None,
        experiments=folder,
    )
    console.print(f"llb lab on http://{host}:{port}/lab  (Ctrl-C to stop)")
    uvicorn.run(create_lab_app(the_lab, results), host=host, port=port, log_level="warning")

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.table import Table

from llmbench.cli.common import console, fail
from llmbench.config.experiment import BuildSpec
from llmbench.llamacpp import builds
from llmbench.llamacpp.introspect import IntrospectionError, binary_flags

app = typer.Typer(help="Manage llama.cpp builds.", no_args_is_help=True)

Backend = Literal["cpu", "cuda", "vulkan", "metal", "blas"]


@app.command("build")
def build_cmd(
    ref: Annotated[str, typer.Option(help="Tag, branch or commit (e.g. b6500, master).")],
    backend: Annotated[Backend, typer.Option(help="ggml backend.")] = "cpu",
    cmake: Annotated[
        list[str] | None, typer.Option("--cmake", "-D", help="Extra CMake define KEY=VALUE.")
    ] = None,
    jobs: Annotated[int | None, typer.Option("--jobs", "-j")] = None,
    force: Annotated[bool, typer.Option(help="Rebuild even if it exists.")] = False,
) -> None:
    """Fetch and compile llama.cpp at a ref (cached per commit + backend + flags)."""
    defines: dict[str, str | int | bool] = {}
    for d in cmake or []:
        key, sep, value = d.partition("=")
        if not sep:
            fail(f"--cmake expects KEY=VALUE, got {d!r}")
        defines[key] = value
    spec = BuildSpec(id="cli", ref=ref, backend=backend, cmake=defines)
    try:
        rb = builds.build(spec, jobs=jobs, force=force, log=lambda m: console.print(f"[dim]{m}[/]"))
    except builds.BuildError as e:
        fail(str(e), code=1)
    console.print(f"[green]built[/] {rb.version} @ {rb.commit[:12]} → {rb.bin_dir}")


@app.command("ls")
def ls_cmd() -> None:
    """List managed builds."""
    t = Table("dir", "ref", "commit", "version", "backend")
    for rb in builds.list_builds():
        t.add_row(rb.id, rb.ref or "", rb.commit[:12], rb.version, rb.backend)
    console.print(t)


@app.command("flags")
def flags_cmd(
    bin_dir: Annotated[Path, typer.Argument(help="Directory with llama-* binaries.")],
    tool: Annotated[str, typer.Option(help="Binary name.")] = "llama-bench",
) -> None:
    """Print the flags a binary supports (as llb sees them)."""
    try:
        flags = binary_flags((bin_dir / tool).resolve())
    except IntrospectionError as e:
        fail(str(e))
    console.print(" ".join(sorted(flags)))

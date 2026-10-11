from typing import Annotated

import typer
from rich.table import Table

from llmbench.cli.common import console, fail
from llmbench.config.experiment import ModelSpec
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.refs import RefError, is_hf, parse_hf
from llmbench.hub.store import ModelNotFoundError, ModelStore, ResolvedModel

app = typer.Typer(help="Manage GGUF models (Hugging Face cache and models/ favorites).")

RefArg = Annotated[str, typer.Argument(help="hf://org/repo[@rev][:QUANT] or a local path.")]


def _store() -> ModelStore:
    return ModelStore(log=lambda m: console.print(f"[dim]{m}[/]", highlight=False))


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _show(r: ResolvedModel) -> None:
    console.print(f"[green]✓[/] {r.path}  [dim]({r.origin}, {_gb(r.size)})[/]", highlight=False)
    if r.repo:
        console.print(f"  {r.repo}@{r.commit}  {', '.join(r.files)}", highlight=False)
    if r.sha256:
        console.print(f"  sha256 {r.sha256}", highlight=False)


@app.command("pull")
def pull(
    ref: RefArg,
    favorite: Annotated[bool, typer.Option(help="Keep it in models/ (not the cache).")] = False,
) -> None:
    """Download a model from Hugging Face (into the HF cache, or models/ with --favorite)."""
    if not is_hf(ref):
        fail("pull expects an hf:// reference")
    try:
        r = _store().pull(ref, favorite=favorite)
    except (ModelNotFoundError, RefError) as e:
        fail(str(e), code=1)
    _show(r)


@app.command("ls")
def ls() -> None:
    """List favorites (models/) and GGUF files in the Hugging Face cache."""
    store = _store()
    t = Table("", "model", "size", "revision", "last used")
    t.add_column("file", overflow="fold")
    favs = store.favorites.entries()
    indexed = {f.path for f in favs}
    for f in favs:
        size = (store.favorites.root / f.path).stat().st_size if f.size is None else f.size
        t.add_row("★", f.repo, _gb(size), f.commit[:12], f.pulled_at[:10], f.file)
    root = store.favorites.root
    if root.is_dir():
        for p in sorted(root.rglob("*.gguf")):
            rel = p.relative_to(root).as_posix()
            if rel not in indexed:
                t.add_row("★", "(local)", _gb(p.stat().st_size), "", "", rel)
    for c in sorted(store.client.cached_files(), key=lambda c: (c.repo, c.name)):
        if c.name.endswith(".gguf"):
            t.add_row("", c.repo, _gb(c.size), c.commit[:12], f"{c.last_accessed:%Y-%m-%d}", c.name)
    console.print(t)
    console.print(f"[dim]★ = models/ ({root}); others live in the Hugging Face cache[/]")


@app.command("inspect")
def inspect(
    ref: RefArg,
    raw: Annotated[bool, typer.Option("--raw", help="Print every metadata key.")] = False,
) -> None:
    """Show GGUF metadata (architecture, params, quant...) of a model on disk. Never downloads."""
    try:
        r = ModelStore(allow_download=False).resolve(ModelSpec(id="inspect", source=ref))
        info = read_gguf(r.shards)
    except (ModelNotFoundError, RefError, GGUFError) as e:
        fail(str(e), code=1)
    _show(r)
    t = Table(box=None, show_header=False)
    for k, v in info.summary().items():
        if v is not None:
            t.add_row(k, f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v))
    t.add_row("tensors", f"{info.tensor_count:,}")
    t.add_row(
        "tensor types",
        ", ".join(f"{k} {v / info.n_params:.0%}" for k, v in sorted(info.tensor_types.items())),
    )
    console.print(t)
    if raw:
        for k, v in info.metadata.items():
            console.print(f"[dim]{k}[/] = {v}", highlight=False)


@app.command("favorite")
def favorite(ref: RefArg) -> None:
    """Move a model into models/ (downloading it first if needed)."""
    try:
        r = _store().pull(ref, favorite=True)
    except (ModelNotFoundError, RefError) as e:
        fail(str(e), code=1)
    _show(r)


@app.command("unfavorite")
def unfavorite(ref: RefArg, yes: Annotated[bool, typer.Option("--yes", "-y")] = False) -> None:
    """Delete a favorite from models/ (it can be pulled again later)."""
    store = _store()
    try:
        favs = store.favorites.find(parse_hf(ref))
    except RefError as e:
        fail(str(e))
    if not favs:
        fail(f"{ref} is not a favorite", code=1)
    for f in favs:
        console.print(f"  {store.favorites.root / f.path}", highlight=False)
    if not yes and not typer.confirm("Delete these files?", default=False):
        raise typer.Exit(1)
    for p in store.unfavorite(ref):
        console.print(f"[dim]deleted {p}[/]", highlight=False)


@app.command("prune")
def prune(
    older_than: Annotated[
        float | None, typer.Option("--older-than", help="Only files unused for N days.")
    ] = None,
    repo: Annotated[str | None, typer.Option(help="Only this repo (org/name).")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only show what would go.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Delete GGUF files from the Hugging Face cache. Never touches models/."""
    store = _store()
    targets, size = store.prune(older_than_days=older_than, repo=repo, dry_run=True)
    if not targets:
        console.print("nothing to prune")
        return
    for c in targets:
        console.print(f"  {c.repo}/{c.name}  {_gb(c.size)}", highlight=False)
    console.print(f"{len(targets)} file(s), {_gb(size)}")
    if dry_run:
        return
    if not yes and not typer.confirm("Delete them?", default=False):
        raise typer.Exit(1)
    _, freed = store.prune(older_than_days=older_than, repo=repo, dry_run=False)
    console.print(f"[green]freed {_gb(freed)}[/]")

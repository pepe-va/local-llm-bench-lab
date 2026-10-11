"""What the lab shows about models: the GGUF choices of a Hub repo and the models on disk."""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from llmbench.hub.client import RemoteFile, RepoListing
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.refs import shard_info
from llmbench.hub.store import ModelStore

Fit = Literal["ok", "tight", "no"]
Where = Literal["favorite", "cache", "local"]

_RUNTIME_BYTES = 1_000_000_000
"""Headroom on top of the weights: KV cache at a few thousand tokens, compute buffers, OS."""
_QUANT = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4))(?=[-_.]|$)"
)


def fit(size: int, ram_total: int) -> Fit:
    """Rough answer to "will it load?": weights plus runtime headroom against total RAM."""
    need = size + _RUNTIME_BYTES
    if need <= 0.75 * ram_total:
        return "ok"
    if need <= 0.95 * ram_total:
        return "tight"
    return "no"


def quant_label(filename: str) -> str:
    base = filename.rsplit("/", 1)[-1].removesuffix(".gguf")
    m = _QUANT.search(base)
    return m.group(1) if m else base


def _is_projector(name: str) -> bool:
    return name.rsplit("/", 1)[-1].lower().startswith("mmproj")


def _group_key(name: str) -> str:
    info = shard_info(name)
    return (
        f"{name.rsplit('/', 1)[0]}/{info[0]}"
        if info and "/" in name
        else (info[0] if info else name)
    )


@dataclass(frozen=True)
class RemoteChoice:
    """One downloadable model of a repo (all shards of a split model together)."""

    file: str
    """First shard, or the only file: what `hf://repo/<file>` points to."""
    quant: str
    size: int
    shards: int
    fit: Fit


def repo_choices(listing: RepoListing, ram_total: int) -> list[RemoteChoice]:
    """Every model file of a repo (projectors left out), smallest first."""
    groups: dict[str, list[RemoteFile]] = {}
    for f in listing.files:
        if f.name.endswith(".gguf") and not _is_projector(f.name):
            groups.setdefault(_group_key(f.name), []).append(f)
    out: list[RemoteChoice] = []
    for files in groups.values():
        files.sort(key=lambda f: f.name)
        size = sum(f.size or 0 for f in files)
        out.append(
            RemoteChoice(
                files[0].name, quant_label(files[0].name), size, len(files), fit(size, ram_total)
            )
        )
    return sorted(out, key=lambda c: (c.size, c.file))


@dataclass(frozen=True)
class LocalModel:
    key: str
    """Stable id for the UI: a short hash of the first file's path."""
    name: str
    """Repo (`org/name`) or, for a loose file, its path under `models/`."""
    file: str
    path: Path
    size: int
    shards: int
    where: Where
    source: str
    """Model reference for an experiment (`hf://repo@commit/file` or an absolute path)."""
    repo: str | None = None
    commit: str | None = None
    also: tuple[str, ...] = field(default_factory=tuple[str, ...])
    """Cache only: other GGUFs in the same revision, deleted with it (deletion is per revision)."""

    @property
    def quant(self) -> str:
        return quant_label(self.file)


def _key(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:12]


def local_models(store: ModelStore) -> list[LocalModel]:
    """Favorites in `models/`, loose `.gguf` files there, and GGUFs in the Hugging Face cache."""
    out: list[LocalModel] = []
    root = store.favorites.root
    favs = [f for f in store.favorites.entries() if (root / f.path).is_file()]
    indexed = {f.path for f in favs}
    fav_groups: dict[tuple[str, str], list[Any]] = {}
    for f in favs:
        fav_groups.setdefault((f.repo, _group_key(f.file)), []).append(f)
    for (repo, _), group in fav_groups.items():
        group.sort(key=lambda f: f.file)
        first = group[0]
        path = root / first.path
        out.append(
            LocalModel(
                _key(path), repo, first.file, path,
                sum((root / f.path).stat().st_size for f in group), len(group), "favorite",
                f"hf://{repo}@{first.commit}/{first.file}", repo, first.commit,
            )
        )  # fmt: skip
    if root.is_dir():
        loose: dict[str, list[Path]] = {}
        for p in sorted(root.rglob("*.gguf")):
            rel = p.relative_to(root).as_posix()
            if rel not in indexed and not _is_projector(rel):
                loose.setdefault(_group_key(rel), []).append(p)
        for paths in loose.values():
            path = paths[0]
            rel = path.relative_to(root).as_posix()
            out.append(
                LocalModel(
                    _key(path), rel, rel, path, sum(p.stat().st_size for p in paths),
                    len(paths), "local", str(path.resolve()),
                )
            )  # fmt: skip
    cached = [c for c in store.client.cached_files() if c.name.endswith(".gguf")]
    by_revision: dict[tuple[str, str], list[Any]] = {}
    for c in cached:
        if not _is_projector(c.name):
            by_revision.setdefault((c.repo, c.commit), []).append(c)
    for (repo, commit), files in by_revision.items():
        groups: dict[str, list[Any]] = {}
        for c in files:
            groups.setdefault(_group_key(c.name), []).append(c)
        for group in groups.values():
            group.sort(key=lambda c: c.name)
            first = group[0]
            others = tuple(sorted(c.name for c in files if c not in group))
            out.append(
                LocalModel(
                    _key(first.path), repo, first.name, first.path, sum(c.size for c in group),
                    len(group), "cache", f"hf://{repo}@{commit}/{first.name}", repo, commit,
                    others,
                )
            )  # fmt: skip
    return sorted(out, key=lambda m: (m.name.lower(), m.size))


def describe(path: Path) -> dict[str, Any]:
    """What a model is, from its GGUF header (cached per file version)."""
    try:
        st = path.stat()
    except OSError:
        return {}
    return _describe(path, st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=128)
def _describe(path: Path, _mtime: int, _size: int) -> dict[str, Any]:
    try:
        info = read_gguf(path)
    except (GGUFError, OSError):
        return {}
    template = str(info.metadata.get("tokenizer.chat_template") or "")
    return {
        "architecture": info.architecture,
        "name": info.metadata.get("general.name"),
        "params_b": round(info.n_params / 1e9, 2) if info.n_params else None,
        "context_length": info.arch("context_length"),
        "quant": info.quant,
        "chat": bool(template),
        "reasoning": "<think>" in template or "enable_thinking" in template,
        "embedding": info.arch("pooling_type") is not None,
    }

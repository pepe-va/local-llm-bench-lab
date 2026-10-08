"""Where a model's GGUF comes from: local path, `models/` favorites, HF cache or download.

Resolution order for `hf://` sources (ADR-0008):
    1. favorites in `models/` (indexed in `models/.llb.json`; works offline)
    2. the Hub listing -> pick files -> HF cache hit, or download into the cache
    3. if the Hub is unreachable: the HF cache alone (offline)
"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from llmbench.config.experiment import ModelSpec
from llmbench.hub.client import (
    CachedFile,
    HubClient,
    HubNotFoundError,
    HubUnavailableError,
    HuggingFaceClient,
    RemoteFile,
    RepoListing,
)
from llmbench.hub.refs import DEFAULT_QUANT, HfRef, RefError, parse_hf, quant_matches, shard_info
from llmbench.paths import models_dir

type Origin = Literal["local", "favorite", "cache", "download"]
type Log = Callable[[str], None]

INDEX_NAME = ".llb.json"
_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


class ModelNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    id: str
    source: str
    shards: tuple[Path, ...]
    """All files of the model; llama.cpp is given the first one."""
    origin: Origin
    sha256: str | None = None
    """Known content identity (combined over shards); None -> hash locally."""
    repo: str | None = None
    commit: str | None = None
    files: tuple[str, ...] = ()
    file_sha256: tuple[str | None, ...] = ()
    """Per-file LFS sha256 from the Hub listing (parallel to `files`)."""

    @property
    def path(self) -> Path:
        return self.shards[0]

    @property
    def size(self) -> int:
        return sum(p.stat().st_size for p in self.shards)


@dataclass(frozen=True)
class Favorite:
    path: str
    """Relative to the models directory."""
    repo: str
    commit: str
    file: str
    sha256: str | None
    size: int | None
    revision: str | None = None
    """The revision as requested (branch/tag), if any."""
    pulled_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def combined_sha256(shas: list[str]) -> str:
    """Identity of a (possibly sharded) model: the file sha256, or a hash over shard hashes."""
    if len(shas) == 1:
        return shas[0]
    return hashlib.sha256("\n".join(shas).encode()).hexdigest()


def pick_files(listing: RepoListing, ref: HfRef) -> list[RemoteFile]:
    """Select the GGUF file(s) a reference points to, including every shard of a split model."""
    ggufs = [
        f
        for f in listing.files
        if f.name.endswith(".gguf") and not f.name.rsplit("/", 1)[-1].startswith("mmproj")
    ]
    if ref.file is not None:
        hit = [f for f in ggufs if f.name == ref.file]
        if not hit:
            raise ModelNotFoundError(
                f"{ref}: no file {ref.file!r} in {ref.repo}@{listing.commit[:12]}"
                + _available(ggufs)
            )
        candidates = hit
    else:
        quant = ref.quant or DEFAULT_QUANT
        candidates = [f for f in ggufs if quant_matches(f.name, quant)]
        if not candidates and ref.quant is None and len(_groups(ggufs)) == 1:
            candidates = ggufs  # single-model repo without a Q4_K_M: take it, like llama.cpp
        if not candidates:
            raise ModelNotFoundError(f"{ref}: no GGUF matches quant {quant!r}" + _available(ggufs))

    groups = _groups(candidates)
    if len(groups) > 1:
        names = "\n  ".join(sorted(groups))
        raise ModelNotFoundError(
            f"{ref} is ambiguous; pick one with hf://{ref.repo}/<file>:\n  {names}"
        )
    key, _ = next(iter(groups.items()))
    files = sorted(_groups(ggufs)[key], key=lambda f: f.name)
    first = shard_info(files[0].name)
    if first is not None and len(files) != first[2]:
        raise ModelNotFoundError(
            f"{ref}: split model is incomplete ({len(files)}/{first[2]} shards)"
        )
    return files


def _groups(files: list[RemoteFile]) -> dict[str, list[RemoteFile]]:
    """Group shards of the same split model under one key (its first-shard-less prefix)."""
    out: dict[str, list[RemoteFile]] = {}
    for f in files:
        out.setdefault(_SHARD_SUFFIX.sub("", f.name), []).append(f)
    return out


def _available(files: list[RemoteFile]) -> str:
    if not files:
        return " (the repo has no GGUF files)"
    return "; available:\n  " + "\n  ".join(sorted(f.name for f in files))


class Favorites:
    """`models/` plus its index (`models/.llb.json`) of where each favorite came from."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or models_dir()

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def entries(self) -> list[Favorite]:
        if not self.index_path.is_file():
            return []
        return [Favorite(**e) for e in json.loads(self.index_path.read_text())]

    def _save(self, entries: list[Favorite]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in sorted(entries, key=lambda e: e.path)]
        self.index_path.write_text(json.dumps(data, indent=2) + "\n")

    def add(self, new: list[Favorite]) -> None:
        paths = {e.path for e in new}
        self._save([e for e in self.entries() if e.path not in paths] + new)

    def remove(self, paths: set[str]) -> None:
        self._save([e for e in self.entries() if e.path not in paths])

    def find(self, ref: HfRef) -> list[Favorite]:
        same = [
            e
            for e in self.entries()
            if e.repo == ref.repo
            and (
                ref.revision is None
                or ref.revision in (e.revision, e.commit)
                or e.commit.startswith(ref.revision)
            )
            and (self.root / e.path).is_file()
        ]
        listing = RepoListing(
            ref.repo, "favorites", [RemoteFile(e.file, e.size, e.sha256) for e in same]
        )
        try:
            files = {f.name for f in pick_files(listing, ref)}
        except ModelNotFoundError:
            return []
        return sorted((e for e in same if e.file in files), key=lambda e: e.file)


class ModelStore:
    def __init__(
        self,
        client: HubClient | None = None,
        *,
        favorites: Favorites | None = None,
        allow_download: bool = True,
        log: Log = lambda _: None,
    ) -> None:
        self._client = client
        self.favorites = favorites or Favorites()
        self.allow_download = allow_download
        self.log = log

    @property
    def client(self) -> HubClient:
        if self._client is None:
            self._client = HuggingFaceClient()
        return self._client

    # -- resolution ----------------------------------------------------------

    def resolve(self, spec: ModelSpec) -> ResolvedModel:
        if spec.scheme == "hf":
            try:
                ref = parse_hf(spec.source)
            except RefError as e:
                raise ModelNotFoundError(f"model {spec.id!r}: {e}") from None
            return self._resolve_hf(spec.id, spec.source, ref)
        if spec.scheme != "file":
            raise ModelNotFoundError(
                f"model {spec.id!r}: unsupported source scheme {spec.scheme!r}"
            )
        return self._resolve_local(spec.id, spec.source)

    def _resolve_local(self, model_id: str, source: str) -> ResolvedModel:
        path = Path(source.removeprefix("file://")).expanduser()
        root = self.favorites.root
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [Path.cwd() / path, root / path]
            if root.is_dir() and len(path.parts) == 1:
                candidates += sorted(root.rglob(path.name))
        for c in candidates:
            if c.is_file():
                return ResolvedModel(model_id, source, _local_shards(c.resolve()), "local")
        tried = ", ".join(str(c) for c in candidates[:2])
        raise ModelNotFoundError(f"model {model_id!r}: file not found (tried {tried}, {root}/**)")

    def _resolve_hf(self, model_id: str, source: str, ref: HfRef) -> ResolvedModel:
        if favs := self.favorites.find(ref):
            return ResolvedModel(
                model_id,
                source,
                tuple(self.favorites.root / f.path for f in favs),
                "favorite",
                sha256=_combined_or_none([f.sha256 for f in favs]),
                repo=ref.repo,
                commit=favs[0].commit,
                files=tuple(f.file for f in favs),
            )
        try:
            listing = self.client.list_repo(ref.repo, ref.revision)
        except HubNotFoundError as e:
            raise ModelNotFoundError(f"model {model_id!r}: {e}") from None
        except HubUnavailableError as e:
            self.log(f"Hugging Face unreachable ({e}); trying the local cache")
            return self._resolve_offline(model_id, source, ref)

        files = pick_files(listing, ref)
        paths: list[Path] = []
        origin: Origin = "cache"
        for f in files:
            cached = self.client.cached_path(ref.repo, f.name, listing.commit)
            if cached is None:
                if not self.allow_download:
                    raise ModelNotFoundError(
                        f"model {model_id!r}: {f.name} is not downloaded "
                        f"(run `llb models pull {ref}`)"
                    )
                self.log(f"downloading {ref.repo}/{f.name} ({_gb(f.size)})...")
                cached = self.client.download(ref.repo, f.name, listing.commit)
                origin = "download"
            paths.append(cached)
        return ResolvedModel(
            model_id,
            source,
            tuple(paths),
            origin,
            sha256=_combined_or_none([f.sha256 for f in files]),
            repo=ref.repo,
            commit=listing.commit,
            files=tuple(f.name for f in files),
            file_sha256=tuple(f.sha256 for f in files),
        )

    def _resolve_offline(self, model_id: str, source: str, ref: HfRef) -> ResolvedModel:
        cached = [c for c in self.client.cached_files() if c.repo == ref.repo]
        by_commit: dict[str, list[CachedFile]] = {}
        for c in cached:
            if ref.revision is None or c.commit.startswith(ref.revision):
                by_commit.setdefault(c.commit, []).append(c)
        # most recently used revision first
        for commit, files in sorted(
            by_commit.items(), key=lambda kv: max(f.last_accessed for f in kv[1]), reverse=True
        ):
            # offline: no LFS metadata, so the identity is hashed locally later
            listing = RepoListing(
                ref.repo, commit, [RemoteFile(f.name, f.size, None) for f in files]
            )
            try:
                picked = pick_files(listing, ref)
            except ModelNotFoundError:
                continue
            paths = {f.name: f.path for f in files}
            return ResolvedModel(
                model_id,
                source,
                tuple(paths[f.name] for f in picked),
                "cache",
                repo=ref.repo,
                commit=commit,
                files=tuple(f.name for f in picked),
            )
        raise ModelNotFoundError(f"model {model_id!r}: {ref} is not available offline")

    # -- management ------------------------------------------------------------

    def pull(self, source: str, *, favorite: bool = False) -> ResolvedModel:
        spec = ModelSpec(id="pull", source=source)
        resolved = self.resolve(spec)
        if favorite and resolved.origin in ("cache", "download"):
            resolved = self.make_favorite(resolved, parse_hf(source))
        return resolved

    def make_favorite(self, resolved: ResolvedModel, ref: HfRef) -> ResolvedModel:
        """Move a cached model into `models/<org>--<repo>/` and index it."""
        assert resolved.repo is not None
        assert resolved.commit is not None
        dest_dir = self.favorites.root / ref.folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        shas = resolved.file_sha256 or (None,) * len(resolved.files)
        new: list[Favorite] = []
        dests: list[Path] = []
        for name, src, sha in zip(resolved.files, resolved.shards, shas, strict=True):
            dest = dest_dir / name.rsplit("/", 1)[-1]
            self.log(f"copying {name} -> {dest}")
            shutil.copyfile(src.resolve(), dest)
            new.append(
                Favorite(
                    path=dest.relative_to(self.favorites.root).as_posix(),
                    repo=resolved.repo,
                    commit=resolved.commit,
                    file=name,
                    sha256=sha,
                    size=dest.stat().st_size,
                    revision=ref.revision,
                )
            )
            dests.append(dest)
        self.favorites.add(new)
        self.release(resolved)
        return ResolvedModel(
            resolved.id,
            resolved.source,
            tuple(dests),
            "favorite",
            sha256=resolved.sha256,
            repo=resolved.repo,
            commit=resolved.commit,
            files=resolved.files,
            file_sha256=resolved.file_sha256,
        )

    def unfavorite(self, source: str) -> list[Path]:
        ref = parse_hf(source)
        favs = self.favorites.find(ref)
        if not favs:
            raise ModelNotFoundError(f"{ref} is not a favorite")
        removed = [self.favorites.root / f.path for f in favs]
        for p in removed:
            p.unlink(missing_ok=True)
        self.favorites.remove({f.path for f in favs})
        return removed

    def release(self, model: ResolvedModel) -> int:
        """Drop a model's cached revision (retention / favorite). Never touches favorites.

        Deletion is per revision (see `HubClient.delete_revisions`), so a revision that
        also holds other GGUF files (e.g. other quants of the same repo) is kept.
        """
        if model.origin not in ("cache", "download") or model.repo is None or model.commit is None:
            return 0
        in_revision = {
            c.name
            for c in self.client.cached_files()
            if c.repo == model.repo and c.commit == model.commit and c.name.endswith(".gguf")
        }
        others = sorted(in_revision - set(model.files))
        if others:
            self.log(
                f"kept {model.repo}@{model.commit[:12]} in the HF cache: it also holds "
                + ", ".join(others)
            )
            return 0
        return self.client.delete_revisions(model.repo, {model.commit})

    def prune(
        self, *, older_than_days: float | None, repo: str | None, dry_run: bool
    ) -> tuple[list[CachedFile], int]:
        """Delete cached revisions holding GGUF files (only the HF cache, never `models/`).

        A revision is pruned as a whole once none of its files was used for `older_than_days`.
        """
        now = datetime.now(UTC)
        revisions: dict[tuple[str, str], list[CachedFile]] = {}
        for c in self.client.cached_files():
            if repo is None or c.repo == repo:
                revisions.setdefault((c.repo, c.commit), []).append(c)
        targets: list[CachedFile] = []
        selected: dict[str, set[str]] = {}
        for (r, commit), files in revisions.items():
            if not any(f.name.endswith(".gguf") for f in files):
                continue
            idle_days = (now - max(f.last_accessed for f in files)).days
            if older_than_days is None or idle_days >= older_than_days:
                targets += files
                selected.setdefault(r, set()).add(commit)
        if dry_run:
            return targets, sum(c.size for c in targets)
        freed = sum(self.client.delete_revisions(r, commits) for r, commits in selected.items())
        return targets, freed


def model_info(resolved: ResolvedModel, sha256: str) -> dict[str, Any]:
    """What gets stored with each run about the model."""
    return {
        "id": resolved.id,
        "source": resolved.source,
        "origin": resolved.origin,
        "path": str(resolved.path),
        "shards": len(resolved.shards),
        "repo": resolved.repo,
        "commit": resolved.commit,
        "files": list(resolved.files),
        "sha256": sha256,
        "size": resolved.size,
    }


def _local_shards(path: Path) -> tuple[Path, ...]:
    info = shard_info(path.name)
    if info is None:
        return (path,)
    prefix, _, total = info
    shards = tuple(
        path.parent / f"{prefix}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)
    )
    missing = [s.name for s in shards if not s.is_file()]
    if missing:
        raise ModelNotFoundError(f"{path.name}: missing shards {', '.join(missing)}")
    return shards


def _combined_or_none(shas: list[str | None]) -> str | None:
    known = [s for s in shas if s]
    return combined_sha256(known) if len(known) == len(shas) else None


def _gb(size: int | None) -> str:
    return f"{size / 1e9:.2f} GB" if size else "? GB"

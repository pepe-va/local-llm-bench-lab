"""Test helpers: a tiny GGUF writer and an in-memory Hugging Face Hub."""

import hashlib
import json
import struct
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from llmbench.hub.client import (
    CachedFile,
    HubNotFoundError,
    HubUnavailableError,
    RemoteFile,
    RepoHit,
    RepoListing,
)

HF_FIXTURES = Path(__file__).parent / "fixtures" / "hf" / "repos.json"


def write_gguf(
    path: Path,
    metadata: dict[str, Any] | None = None,
    tensors: list[tuple[str, tuple[int, ...], int]] | None = None,
) -> Path:
    """Write a valid GGUF v3 header (no tensor data, which readers never touch)."""
    metadata = {"general.architecture": "llama", "llama.context_length": 2048} | (metadata or {})
    tensors = tensors or [("tok_embd.weight", (64, 512), 0), ("blk.0.attn_q.weight", (64, 64), 12)]

    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    def value(v: Any) -> bytes:
        if isinstance(v, bool):
            return struct.pack("<I?", 7, v)
        if isinstance(v, int):
            return struct.pack("<Iq", 11, v) if v < 0 else struct.pack("<IQ", 10, v)
        if isinstance(v, float):
            return struct.pack("<Id", 12, v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + s(v)
        if isinstance(v, list):
            items: list[Any] = v  # pyright: ignore[reportUnknownVariableType]
            body = b"".join(struct.pack("<Q", len(x.encode())) + x.encode() for x in items)
            return struct.pack("<IIQ", 9, 8, len(items)) + body
        raise TypeError(v)

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        out += s(k) + value(v)
    for name, dims, ttype in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", ttype, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


@dataclass
class FakeHub:
    """Implements `HubClient` in memory. Downloads write small valid GGUFs into `cache_dir`."""

    cache_dir: Path
    repos: dict[str, RepoListing] = field(default_factory=dict[str, RepoListing])
    online: bool = True
    downloads: list[str] = field(default_factory=list[str])
    _cached: dict[tuple[str, str, str], CachedFile] = field(default_factory=dict)

    @classmethod
    def from_fixture(cls, cache_dir: Path) -> "FakeHub":
        data = json.loads(HF_FIXTURES.read_text())
        repos = {
            repo: RepoListing(
                repo, d["sha"], [RemoteFile(f["name"], f["size"], f["sha256"]) for f in d["files"]]
            )
            for repo, d in data.items()
        }
        return cls(cache_dir, repos)

    def add_repo(self, repo: str, files: list[str], commit: str = "c" * 40) -> None:
        self.repos[repo] = RepoListing(
            repo,
            commit,
            [RemoteFile(f, 1000, hashlib.sha256(f.encode()).hexdigest()) for f in files],
        )

    def list_repo(self, repo: str, revision: str | None) -> RepoListing:
        if not self.online:
            raise HubUnavailableError("offline")
        if repo not in self.repos:
            raise HubNotFoundError(f"{repo}: repository not found")
        listing = self.repos[repo]
        if revision is not None and not listing.commit.startswith(revision) and revision != "main":
            raise HubNotFoundError(f"{repo}: revision {revision!r} not found")
        return listing

    def cached_path(self, repo: str, file: str, commit: str) -> Path | None:
        hit = self._cached.get((repo, commit, file))
        return hit.path if hit else None

    def download(self, repo: str, file: str, commit: str) -> Path:
        if not self.online:
            raise HubUnavailableError("offline")
        blob = write_gguf(self.cache_dir / "blobs" / f"{len(self.downloads):04d}")
        link = self.cache_dir / repo.replace("/", "--") / "snapshots" / commit / file
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        self._cached[(repo, commit, file)] = CachedFile(
            repo, commit, file, link, blob.stat().st_size, datetime.now(UTC)
        )
        self.downloads.append(f"{repo}/{file}")
        return link

    def cached_files(self) -> list[CachedFile]:
        return list(self._cached.values())

    def delete_revisions(self, repo: str, commits: set[str]) -> int:
        freed = 0
        for key, f in list(self._cached.items()):
            if f.repo == repo and f.commit in commits:
                freed += f.size
                f.path.resolve().unlink(missing_ok=True)
                f.path.unlink(missing_ok=True)
                del self._cached[key]
        return freed

    def search(self, query: str, limit: int) -> list[RepoHit]:
        if not self.online:
            raise HubUnavailableError("offline")
        words = query.lower().split()
        return [
            RepoHit(repo, 1000 - i, 10, "text-generation", False)
            for i, repo in enumerate(sorted(self.repos))
            if all(w in repo.lower() for w in words)
        ][:limit]

    def downloading_bytes(self, repo: str) -> int:
        return 0

    def age(self, days: int) -> None:
        """Pretend every cached file was last used `days` ago."""
        old = datetime.now(UTC) - timedelta(days=days)
        self._cached = {
            k: CachedFile(c.repo, c.commit, c.name, c.path, c.size, old)
            for k, c in self._cached.items()
        }

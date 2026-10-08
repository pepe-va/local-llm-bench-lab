"""Thin seam over `huggingface_hub`, so resolution logic can be tested without network."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from huggingface_hub import (
    HfApi,
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    scan_cache_dir,
    try_to_load_from_cache,
)
from huggingface_hub.errors import (
    CacheNotFound,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

try:  # huggingface_hub >= 2 is built on httpx2; older releases on httpx/requests
    from httpx2 import HTTPError as _TransportFamily  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    from httpx import HTTPError as _TransportFamily

# Network failures (proxy refusal, DNS, connection reset...) as raised by the Hub client.
NETWORK_ERRORS: tuple[type[BaseException], ...] = (_TransportFamily, OSError, OfflineModeIsEnabled)


class HubUnavailableError(Exception):
    """Network/offline problem: the caller may fall back to the local cache."""


class HubNotFoundError(Exception):
    """The Hub answered: the repo or revision does not exist (or is gated/private)."""


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None
    sha256: str | None
    """LFS sha256 of the content; GGUF files are always LFS on the Hub."""


@dataclass(frozen=True)
class RepoListing:
    repo: str
    commit: str
    files: list[RemoteFile]


@dataclass(frozen=True)
class CachedFile:
    repo: str
    commit: str
    name: str
    path: Path
    size: int
    last_accessed: datetime


class HubClient(Protocol):
    def list_repo(self, repo: str, revision: str | None) -> RepoListing: ...
    def cached_path(self, repo: str, file: str, commit: str) -> Path | None: ...
    def download(self, repo: str, file: str, commit: str) -> Path: ...
    def cached_files(self) -> list[CachedFile]: ...
    def delete_revisions(self, repo: str, commits: set[str]) -> int: ...


class HuggingFaceClient:
    def __init__(self) -> None:
        self.api = HfApi()

    def list_repo(self, repo: str, revision: str | None) -> RepoListing:
        try:
            info = self.api.model_info(repo, revision=revision, files_metadata=True)
        except GatedRepoError:
            raise HubNotFoundError(
                f"{repo} is gated: accept its license on huggingface.co and set HF_TOKEN"
            ) from None
        except RepositoryNotFoundError:
            raise HubNotFoundError(
                f"{repo}: repository not found (private? set HF_TOKEN)"
            ) from None
        except RevisionNotFoundError:
            raise HubNotFoundError(f"{repo}: revision {revision!r} not found") from None
        except (*NETWORK_ERRORS, HfHubHTTPError) as e:
            # huggingface_hub 2.x is built on httpx: network failures arrive as
            # httpx.TransportError (e.g. ProxyError, ConnectError), not OSError
            raise HubUnavailableError(f"{repo}: {e}") from None
        files = [
            RemoteFile(s.rfilename, s.size, s.lfs.sha256 if s.lfs else None)
            for s in info.siblings or []
        ]
        assert info.sha is not None
        return RepoListing(repo, info.sha, files)

    def cached_path(self, repo: str, file: str, commit: str) -> Path | None:
        hit = try_to_load_from_cache(repo, file, revision=commit)
        return Path(hit) if isinstance(hit, str) else None

    def download(self, repo: str, file: str, commit: str) -> Path:
        try:
            return Path(str(hf_hub_download(repo, file, revision=commit)))
        except (LocalEntryNotFoundError, *NETWORK_ERRORS) as e:
            raise HubUnavailableError(str(e)) from None

    def cached_files(self) -> list[CachedFile]:
        try:
            info = scan_cache_dir()
        except CacheNotFound:
            return []
        out: list[CachedFile] = []
        for repo in info.repos:
            if repo.repo_type != "model":
                continue
            for rev in repo.revisions:
                for f in rev.files:
                    rel = f.file_path.relative_to(rev.snapshot_path).as_posix()
                    out.append(
                        CachedFile(
                            repo=repo.repo_id,
                            commit=rev.commit_hash,
                            name=rel,
                            path=f.file_path,
                            size=f.size_on_disk,
                            last_accessed=datetime.fromtimestamp(f.blob_last_accessed, UTC),
                        )
                    )
        return out

    def delete_revisions(self, repo: str, commits: set[str]) -> int:
        """Delete whole cached revisions through huggingface_hub (it owns the blob refcounts).

        huggingface_hub 2.x stores content in a shared, refcounted blob store; removing
        files by hand would corrupt it, so deletion goes through the official strategy.
        """
        try:
            info = scan_cache_dir()
        except CacheNotFound:
            return 0
        hashes = [
            rev.commit_hash
            for r in info.repos
            if r.repo_id == repo
            for rev in r.revisions
            if rev.commit_hash in commits
        ]
        if not hashes:
            return 0
        strategy = info.delete_revisions(*hashes)
        strategy.execute()
        return strategy.expected_freed_size

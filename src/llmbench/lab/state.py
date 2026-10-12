"""The lab's state: downloads, the chat server and the background job, kept from clashing.

A small machine holds one model at a time and any measurement must run alone, so:
a run, judge or build stops the chat first; no chat, download or deletion starts while a job
runs; and no run, judge or build starts while a download is still writing to disk.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llmbench.hub.client import HubNotFoundError, HubUnavailableError
from llmbench.hub.refs import RefError
from llmbench.hub.store import ModelNotFoundError, ModelStore
from llmbench.lab import editor
from llmbench.lab.chat import ChatServer, ChatSettings
from llmbench.lab.editor import Experiments
from llmbench.lab.experiment import PRESETS, default_profile, default_ref
from llmbench.lab.jobs import JobKind, Jobs, LabBusyError
from llmbench.lab.models import LocalModel, describe, fit, local_models, repo_choices
from llmbench.llamacpp import builds
from llmbench.paths import llb_home
from llmbench.telemetry.sources import (
    cpu_freq_mhz,
    cpu_governor,
    ram_total_mb,
    ram_used_mb,
    temperature_c,
    throttled_raw,
)

_MB = 1024 * 1024


class LabError(Exception):
    """A request the lab cannot honour as asked (HTTP 400)."""


@dataclass
class Download:
    id: str
    repo: str
    file: str
    commit: str
    files: list[tuple[str, int]]
    """Every shard with its size."""
    started: float
    state: Literal["running", "done", "error"] = "running"
    error: str | None = None
    favorite: bool = False
    """Keep it in `models/` once downloaded (like `llb models pull --favorite`)."""

    @property
    def size(self) -> int:
        return sum(s for _, s in self.files)


class Lab:
    def __init__(
        self,
        *,
        root: Path,
        home: Path | None = None,
        store: ModelStore | None = None,
        llama_server: Path | None = None,
        experiments: Path | None = None,
    ) -> None:
        self.root = root
        """Repository root: commands run here, and `suites/` lives here."""
        self.experiments = Experiments(experiments or root / "experiments", root)
        self.home = home or llb_home() / "lab"
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = store or ModelStore()
        self.llama_server = llama_server
        self.jobs = Jobs(self.home)
        self.chat: ChatServer | None = None
        self.downloads: dict[str, Download] = {}
        self._lock = threading.RLock()

    # -- models --------------------------------------------------------------------

    def ram_total(self) -> int:
        return int(ram_total_mb() * _MB)

    def search(self, query: str) -> list[dict[str, Any]]:
        query = query.strip()
        if "/" in query and " " not in query:  # a repo id, URL or hf:// reference
            return [{"repo": _repo_id(query), "downloads": None, "likes": None, "direct": True}]
        try:
            hits = self.store.client.search(query, 25)
        except HubUnavailableError as e:
            raise LabError(f"Hugging Face is not reachable: {e}") from None
        return [
            {"repo": h.repo, "downloads": h.downloads, "likes": h.likes,
             "pipeline": h.pipeline, "gated": h.gated}
            for h in hits
        ]  # fmt: skip

    def repo(self, repo: str) -> dict[str, Any]:
        repo = _repo_id(repo)
        try:
            listing = self.store.client.list_repo(repo, None)
        except (HubNotFoundError, HubUnavailableError) as e:
            raise LabError(str(e)) from None
        choices = repo_choices(listing, self.ram_total())
        return {
            "repo": repo,
            "commit": listing.commit,
            "choices": [c.__dict__ for c in choices],
            "projectors": [f.name for f in listing.files if "mmproj" in f.name.lower()],
            "other_files": len(listing.files),
        }

    def models(self) -> list[dict[str, Any]]:
        ram = self.ram_total()
        chat_key = self.chat.model.key if self.chat and self.chat.state != "stopped" else None
        return [
            {
                "key": m.key, "name": m.name, "file": m.file, "quant": m.quant, "size": m.size,
                "shards": m.shards, "where": m.where, "source": m.source, "also": list(m.also),
                "fit": fit(m.size, ram), "info": describe(m.path), "in_chat": m.key == chat_key,
            }
            for m in local_models(self.store)
        ]  # fmt: skip

    def model(self, key: str) -> LocalModel:
        for m in local_models(self.store):
            if m.key == key:
                return m
        raise LabError("that model is no longer on disk")

    def download(self, repo: str, file: str, *, favorite: bool = False) -> Download:
        repo = _repo_id(repo)
        with self._lock:
            if self.jobs.busy:
                raise LabBusyError("a job is running: a download now would skew its numbers")
            if any(d.repo == repo and d.state == "running" for d in self.downloads.values()):
                raise LabBusyError(f"{repo} is already downloading")
            try:
                listing = self.store.client.list_repo(repo, None)
            except (HubNotFoundError, HubUnavailableError) as e:
                raise LabError(str(e)) from None
            choice = next((c for c in repo_choices(listing, 1) if c.file == file), None)
            if choice is None:
                raise LabError(f"{file} is not a model file of {repo}")
            group = [
                (f.name, f.size or 0)
                for f in listing.files
                if f.name == file or _same_split(f.name, file, choice.shards)
            ]
            d = Download(
                f"{repo}/{file}", repo, file, listing.commit, group, time.time(), favorite=favorite
            )
            self.downloads[d.id] = d
        threading.Thread(target=self._download, args=(d,), daemon=True).start()
        return d

    def _download(self, d: Download) -> None:
        try:
            self.store.pull(f"hf://{d.repo}@{d.commit}/{d.file}", favorite=d.favorite)
        except (ModelNotFoundError, RefError, HubUnavailableError, OSError) as e:
            d.state, d.error = "error", str(e)
        else:
            d.state = "done"

    def download_progress(self, d: Download) -> dict[str, Any]:
        done = 0
        if d.state == "running":
            for name, size in d.files:
                if self.store.client.cached_path(d.repo, name, d.commit) is not None:
                    done += size
            done += self.store.client.downloading_bytes(d.repo)
        else:
            done = d.size if d.state == "done" else 0
        return {
            "id": d.id, "repo": d.repo, "file": d.file, "size": d.size, "done": min(done, d.size),
            "state": d.state, "error": d.error, "elapsed_s": round(time.time() - d.started),
            "favorite": d.favorite,
        }  # fmt: skip

    def favorite(self, key: str) -> None:
        m = self.model(key)
        if m.where != "cache":
            raise LabError("only models in the Hugging Face cache can be moved to models/")
        self._not_while_busy(m, "moving it")
        try:
            self.store.pull(m.source, favorite=True)
        except (ModelNotFoundError, RefError) as e:
            raise LabError(str(e)) from None

    def delete(self, key: str) -> None:
        m = self.model(key)
        self._not_while_busy(m, "deleting it")
        if m.where == "favorite":
            self.store.unfavorite(m.source)
        elif m.where == "cache":
            assert m.repo is not None
            assert m.commit is not None
            self.store.client.delete_revisions(m.repo, {m.commit})  # ADR-0010: through the API
        else:
            for p in sorted(m.path.parent.glob("*.gguf")):
                if p == m.path or (m.shards > 1 and _same_split(p.name, m.path.name, m.shards)):
                    p.unlink()

    def _not_while_busy(self, m: LocalModel, what: str) -> None:
        if self.jobs.busy:
            raise LabBusyError(f"a job is running: wait before {what}")
        if self.chat and self.chat.model.key == m.key and self.chat.state != "stopped":
            raise LabBusyError(f"the chat is using this model: stop it before {what}")

    # -- chat ----------------------------------------------------------------------

    def builds(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.llama_server is not None:
            out.append(
                {"id": "given", "label": str(self.llama_server), "path": str(self.llama_server)}
            )
        found = [b for b in builds.list_builds() if b.tool("llama-server").is_file()]
        found.sort(key=lambda b: b.bin_dir.stat().st_mtime, reverse=True)
        out += [
            {
                "id": b.id,
                "label": f"{b.ref or b.version} · {b.commit[:8]} · {b.backend}",
                "path": str(b.tool("llama-server")),
            }
            for b in found
        ]
        return out

    def start_chat(self, key: str, settings: ChatSettings, build: str | None = None) -> ChatServer:
        with self._lock:
            if self.jobs.busy:
                raise LabBusyError("a job is running: chat when it is done")
            model = self.model(key)
            info = describe(model.path)
            if info.get("embedding"):
                raise LabError("this is an embedding model: it does not chat")
            available = self.builds()
            if not available:
                raise LabError("no llama.cpp build with llama-server: build one first")
            chosen = next((b for b in available if b["id"] == build), available[0])
            self.stop_chat()
            log = self.home / "logs" / "chat-server.log"
            self.chat = ChatServer(model, Path(chosen["path"]), settings, log)
            return self.chat

    def stop_chat(self) -> None:
        with self._lock:
            if self.chat is not None:
                self.chat.stop()

    # -- experiments (the test tab) ------------------------------------------------

    def editor_meta(self) -> dict[str, Any]:
        profiles = self.experiments.profiles()
        default = default_profile(profiles)
        return {
            "schema": editor.schema(),
            "help": editor.field_help(),
            "folder": str(self.experiments.folder),
            "files": self.experiments.files(),
            "profiles": profiles,
            "default_profile": default,
            "suites": self.experiments.suites(),
            "presets": [p.__dict__ for p in PRESETS],
            "default_ref": default_ref(self.experiments.folder / default) if default else "master",
        }

    def new_experiment(
        self,
        keys: list[str],
        profile: str | None,
        presets: list[str],
        suites: list[str] | None = None,
    ) -> str:
        models = [self.model(k) for k in keys]
        known = set(self.experiments.suites())
        if unknown := [s for s in suites or [] if s not in known]:
            raise LabError(f"unknown suite(s): {', '.join(unknown)}")
        doc = editor.new_experiment(
            models, folder=self.experiments.folder, root=self.root, profile=profile,
            presets=presets, suites=suites,
        )  # fmt: skip
        return editor.dump(doc)

    def command(
        self,
        name: str,
        command: str,
        *,
        models: list[str] | None = None,
        resume: bool = False,
        no_build: bool = False,
        force: bool = False,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run an `llb` command on a saved experiment file, as the terminal would."""
        path = self.experiments.path(name)
        text = self.experiments.read(name)
        experiment = str(editor.parse(text).get("experiment", {}).get("name") or "") or None
        if command == "guide":
            return {"sync": self.experiments.run_sync(["guide", str(path)])}
        if command == "run":
            args = ["run", "-y", str(path)]
            args += ["--resume"] if resume else []
            args += ["--no-build"] if no_build else []
            for m in models or []:
                args += ["-m", m]
        elif command == "judge":
            args = ["judge", str(path)] + (["--force"] if force else [])
        elif command == "report":
            args = ["report", str(path)]
            for f in formats or []:
                args += ["-f", f]
        else:
            raise LabError(f"unknown command {command!r}")
        # run and judge load models and measure; a report only reads results
        return self.start_job(
            command, f"{command} {name}", args, heavy=command != "report", experiment=experiment
        )

    def start_job(
        self,
        kind: JobKind,
        title: str,
        args: list[str],
        *,
        heavy: bool,
        experiment: str | None = None,
    ) -> dict[str, Any]:
        """Run `llb <args>` as the lab's one background job.

        `heavy` jobs (they load models, compile or measure) unload the chat first and wait
        for downloads to finish. Returns the job and whether the chat was unloaded.
        """
        with self._lock:
            if heavy:
                self._before_job()
                stopped_chat = self.chat is not None and self.chat.state != "stopped"
                self.stop_chat()
            else:
                if self.jobs.busy:
                    raise LabBusyError("another job is already running")
                stopped_chat = False
            job = self.jobs.start(kind, title, args, cwd=self.root, experiment=experiment)
        return {"job": job.to_json(), "stopped_chat": stopped_chat}

    def _before_job(self) -> None:
        if self.jobs.busy:
            raise LabBusyError("another job is already running")
        if any(d.state == "running" for d in self.downloads.values()):
            raise LabBusyError("a download is in progress: start when it is done")

    def cancel_job(self) -> None:
        if self.jobs.current is not None:
            self.jobs.current.cancel()

    # -- status --------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        job = self.jobs.current
        return {
            "machine": {
                "temp_c": temperature_c(),
                "ram_used_mb": round(ram_used_mb()),
                "ram_total_mb": round(ram_total_mb()),
                "cpu_mhz": cpu_freq_mhz(),
                "governor": cpu_governor(),
                "throttled": throttled_raw(),
            },
            "chat": self.chat.to_json() if self.chat else None,
            "job": job.to_json() if job else None,
            "downloads": [
                self.download_progress(d)
                for d in sorted(self.downloads.values(), key=lambda d: -d.started)
            ],
        }

    def close(self) -> None:
        """On shutdown: free the chat's memory. A running job carries on by itself."""
        self.stop_chat()


def _repo_id(text: str) -> str:
    """`org/name` from a repo id, a huggingface.co URL or an hf:// reference."""
    t = text.strip().removeprefix("hf://")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        t = t.removeprefix(prefix)
    parts = [p for p in t.replace(":", "/").replace("@", "/").split("/") if p]
    if len(parts) < 2:  # noqa: PLR2004
        raise LabError(f"not a Hugging Face repo: {text!r} (expected org/name)")
    return f"{parts[0]}/{parts[1]}"


def _same_split(name: str, first: str, shards: int) -> bool:
    """Whether `name` is a shard of the split model whose first shard is `first`."""
    if shards < 2:  # noqa: PLR2004
        return False
    stem = first.rsplit("-00001-of-", 1)[0]
    return name.startswith(stem + "-") and name.endswith(f"-of-{shards:05d}.gguf")

"""What the system tab shows and does: the environment, llama.cpp builds and their flags,
plus the models tab's extras (inspect a GGUF, prune the cache, pull by reference)."""

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from huggingface_hub import constants
from pydantic import ValidationError

from llmbench import __version__
from llmbench.config.experiment import BuildSpec
from llmbench.hub.client import HubNotFoundError, HubUnavailableError
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.refs import RefError, is_hf, parse_hf, shard_info
from llmbench.hub.store import ModelNotFoundError, pick_files
from llmbench.lab.experiment import default_profile, default_ref
from llmbench.lab.jobs import LabBusyError
from llmbench.lab.models import LocalModel
from llmbench.lab.state import Download, Lab, LabError
from llmbench.llamacpp import builds
from llmbench.llamacpp.introspect import IntrospectionError, binary_flags, binary_version
from llmbench.paths import builds_dir, llama_src_dir, llb_home, runs_dir

TOOLS = ("llama-bench", "llama-server")
"""The binaries llb drives, so the ones whose flags it checks."""
EXTRAS: dict[str, tuple[str, ...]] = {
    "report": ("jinja2", "plotly", "matplotlib"),
    "serve": ("fastapi", "uvicorn", "jinja2"),
    "hf-datasets": ("pyarrow",),
    "lm-eval": ("lm_eval",),
}
"""Optional extras and the modules each one brings."""
BUILD_PROGRAMS = ("git", "cmake", "c++", "make", "ninja")

_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_CMAKE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
_SKIPPED_ARRAY = re.compile(r"^<array of (\d+)>$")
"""How `read_gguf` stands in for a long array (e.g. the tokenizer's tokens) it skipped."""
_MAX_STRING = 4000
_MAX_ITEMS = 32
_MAX_HELP = 200_000


# -- environment -------------------------------------------------------------------


def environment(lab: Lab) -> dict[str, Any]:
    """`llb doctor`, plus where the lab keeps things, how much they weigh and what is free."""
    cached = [c for c in lab.store.client.cached_files() if c.name.endswith(".gguf")]
    hf_cache = Path(constants.HF_HUB_CACHE)
    locations = [
        _location("LLB_HOME", llb_home(), None),
        _location("llama.cpp builds", builds_dir(), _tree_size(builds_dir())),
        _location("llama.cpp source", llama_src_dir(), _tree_size(llama_src_dir())),
        _location("runs", runs_dir(), _tree_size(runs_dir())),
        _location(
            "models/ (favorites)",
            lab.store.favorites.root.resolve(),
            _tree_size(lab.store.favorites.root, suffix=".gguf"),
        ),
        _location("Hugging Face cache (GGUF)", hf_cache, sum(c.size for c in cached)),
        _location("experiments", lab.experiments.folder, None),
        _location("repository (commands run here)", lab.root, None),
    ]
    return {
        "llb": __version__,
        "python": platform.python_version(),
        "python_path": sys.executable,
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpus": os.cpu_count(),
        "locations": locations,
        "extras": [
            {"name": name, "modules": list(mods), "installed": all(map(_importable, mods))}
            for name, mods in EXTRAS.items()
        ],
        "programs": {p: shutil.which(p) for p in BUILD_PROGRAMS},
    }


def _location(label: str, path: Path, used: int | None) -> dict[str, Any]:
    free = None
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        free, total = usage.free, usage.total
    except OSError:
        total = None
    return {
        "label": label, "path": str(path), "exists": path.exists(), "used": used,
        "free": free, "disk_total": total,
    }  # fmt: skip


def _tree_size(root: Path, suffix: str | None = None) -> int:
    """Bytes under `root` (symlinks not followed, so a cache snapshot is not counted twice)."""
    total = 0
    stack = [root]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False) and (
                    suffix is None or e.name.endswith(suffix)
                ):
                    total += e.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# -- builds ------------------------------------------------------------------------


def build_list(lab: Lab) -> list[dict[str, Any]]:
    """Managed builds (`llb llama ls`), plus the llama-server the lab was started with."""
    out: list[dict[str, Any]] = []
    for b in builds.list_builds():
        out.append(
            {
                "id": b.id, "ref": b.ref, "commit": b.commit, "version": b.version,
                "backend": b.backend, "cmake": b.cmake, "bin_dir": str(b.bin_dir),
                "built_at": _built_at(b.bin_dir.parent.parent / "manifest.json"),
                "tools": {t: b.tool(t).is_file() for t in builds.TARGETS},
                "managed": True,
            }
        )  # fmt: skip
    if lab.llama_server is not None:
        bin_dir = lab.llama_server.parent
        out.append(
            {
                "id": "given", "ref": None, "commit": None, "version": None, "backend": None,
                "cmake": {}, "bin_dir": str(bin_dir), "built_at": None,
                "tools": {t: (bin_dir / t).is_file() for t in builds.TARGETS}
                | {"llama-server": lab.llama_server.is_file()},
                "managed": False,
            }
        )  # fmt: skip
    return out


def _built_at(manifest: Path) -> str | None:
    try:
        value = json.loads(manifest.read_text()).get("built_at")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def build_options(lab: Lab) -> dict[str, Any]:
    """What the build form offers: backends, their CMake defines and a default ref."""
    profile = default_profile(lab.experiments.profiles())
    ref = default_ref(lab.experiments.folder / profile) if profile else "master"
    return {
        "backends": list(builds.BACKEND_CMAKE),
        "backend_cmake": builds.BACKEND_CMAKE,
        "base_cmake": builds.BASE_CMAKE,
        "default_ref": ref,
        "cpus": os.cpu_count(),
    }


def build_args(
    ref: str, *, backend: str, cmake: list[str], jobs: int | None, force: bool
) -> list[str]:
    """`llb llama build …` argv, every value checked first (it reaches git and cmake)."""
    ref = ref.strip()
    if not _REF.match(ref) or ".." in ref:
        raise LabError("give a llama.cpp tag, branch or commit (letters, digits and . _ - /)")
    if backend not in builds.BACKEND_CMAKE:
        raise LabError(f"unknown backend {backend!r}: one of {', '.join(builds.BACKEND_CMAKE)}")
    defines: dict[str, str | int | bool] = {}
    for d in cmake:
        key, sep, value = d.strip().partition("=")
        if not sep or not _CMAKE_KEY.match(key):
            raise LabError(f"a CMake define is KEY=VALUE (KEY: letters, digits, _), got {d!r}")
        if len(value) > 500 or any(ord(c) < 32 or ord(c) == 127 for c in value):  # noqa: PLR2004
            raise LabError(f"{key}: the value must be one line of at most 500 characters")
        defines[key] = value
    if jobs is not None and not 1 <= jobs <= 1024:  # noqa: PLR2004
        raise LabError("jobs must be between 1 and 1024")
    try:
        BuildSpec.model_validate({"id": "lab", "ref": ref, "backend": backend, "cmake": defines})
    except ValidationError as e:
        raise LabError(str(e)) from None
    args = ["llama", "build", "--ref", ref, "--backend", backend]
    for key, value in defines.items():
        args += ["--cmake", f"{key}={value}"]
    if jobs is not None:
        args += ["--jobs", str(jobs)]
    if force:
        args.append("--force")
    return args


def start_build(
    lab: Lab, ref: str, *, backend: str, cmake: list[str], jobs: int | None, force: bool
) -> dict[str, Any]:
    args = build_args(ref, backend=backend, cmake=cmake, jobs=jobs, force=force)
    title = f"build llama.cpp {ref.strip()} ({backend}{', forced' if force else ''})"
    return lab.start_job("build", title, args, heavy=True)


def _bin_dir(lab: Lab, build_id: str) -> Path:
    """Only binaries the lab knows of: a request never names a path to execute."""
    for b in build_list(lab):
        if b["id"] == build_id:
            return Path(b["bin_dir"])
    raise LabError(f"no build {build_id!r}")


def flags(lab: Lab, build_id: str, tool: str, *, raw: bool = False) -> dict[str, Any]:
    """`llb llama flags <bin_dir> --tool <tool>`: the flags llb validates argv against."""
    if tool not in TOOLS:
        raise LabError(f"tool must be one of {', '.join(TOOLS)}")
    binary = (_bin_dir(lab, build_id) / tool).resolve()
    try:
        found = binary_flags(binary)
    except IntrospectionError as e:
        raise LabError(str(e)) from None
    try:
        version = binary_version(binary).version
    except IntrospectionError:
        version = None
    out: dict[str, Any] = {
        "build": build_id, "tool": tool, "binary": str(binary), "version": version,
        "flags": sorted(found, key=lambda f: (f.lstrip("-").lower(), f)),
    }  # fmt: skip
    if raw:
        out["help"] = _help_text(binary)
    return out


def _help_text(binary: Path) -> str:
    try:
        p = subprocess.run(
            [str(binary), "--help"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LabError(f"cannot run {binary.name} --help: {e}") from None
    text = p.stdout + p.stderr
    return text if len(text) <= _MAX_HELP else text[:_MAX_HELP] + "\n… (truncated)"


# -- models: inspect, prune, pull by reference --------------------------------------


def _shards(m: LocalModel) -> list[Path]:
    info = shard_info(m.path.name)
    if m.shards < 2 or info is None:  # noqa: PLR2004
        return [m.path]
    prefix, _, total = info
    paths = [m.path.parent / f"{prefix}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    return [p for p in paths if p.is_file()]


def inspect(lab: Lab, key: str) -> dict[str, Any]:
    """`llb models inspect`: the GGUF summary, tensor types and (abridged) raw metadata."""
    m = lab.model(key)
    try:
        info = read_gguf(_shards(m))
    except (GGUFError, OSError) as e:
        raise LabError(f"cannot read {m.file}: {e}") from None
    summary = {k: v for k, v in info.summary().items() if v is not None}
    types = sorted(info.tensor_types.items())
    return {
        "key": m.key, "name": m.name, "file": m.file, "path": str(m.path), "size": m.size,
        "shards": m.shards, "where": m.where, "source": m.source,
        "summary": summary, "tensors": info.tensor_count,
        "tensor_types": [
            {"type": t, "params": n, "share": n / info.n_params if info.n_params else None}
            for t, n in types
        ],
        "metadata": [_meta(k, v) for k, v in info.metadata.items()],
    }  # fmt: skip


def _meta(key: str, value: Any) -> dict[str, Any]:
    """One metadata entry, small enough to send: long arrays and strings are abridged."""
    if isinstance(value, list):
        items = cast(list[Any], value)
        n = len(items)
        if n > _MAX_ITEMS:
            head = ", ".join(_short(x) for x in items[:8])
            return {"key": key, "type": "array", "value": f"[{n:,} items] {head}, …"}
        return {"key": key, "type": "array", "value": "[" + ", ".join(map(_short, items)) + "]"}
    if isinstance(value, str) and (m := _SKIPPED_ARRAY.match(value)):
        return {"key": key, "type": "array", "value": f"[{int(m[1]):,} items]"}
    if isinstance(value, str):
        text = value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "… (truncated)"
        return {"key": key, "type": "string", "value": text, "length": len(value)}
    return {"key": key, "type": type(value).__name__, "value": str(value)}


def _short(x: Any) -> str:
    s = repr(x) if isinstance(x, str) else str(x)
    return s if len(s) <= 80 else s[:80] + "…"  # noqa: PLR2004


def prune(
    lab: Lab,
    *,
    older_than_days: float | None,
    repo: str | None,
    dry_run: bool,
    expect: list[str] | None = None,
) -> dict[str, Any]:
    """`llb models prune`: whole cached revisions with GGUF files (never `models/`).

    Deleting refuses while a job or a download runs, when the model loaded in the chat
    would go, and when the preview the user confirmed (`expect`) no longer matches.
    """
    repo = (repo or "").strip() or None
    targets, size = lab.store.prune(older_than_days=older_than_days, repo=repo, dry_run=True)
    chat = lab.chat.model if lab.chat is not None and lab.chat.state != "stopped" else None
    chat_rev = (chat.repo, chat.commit) if chat is not None and chat.where == "cache" else None
    now = datetime.now(UTC)
    files = [
        {
            "repo": c.repo, "commit": c.commit, "name": c.name, "size": c.size,
            "last_used": c.last_accessed.isoformat(timespec="seconds"),
            "idle_days": (now - c.last_accessed).days, "in_chat": (c.repo, c.commit) == chat_rev,
        }
        for c in sorted(targets, key=lambda c: (c.repo, c.commit, c.name))
    ]  # fmt: skip
    revisions = sorted({f"{c.repo}@{c.commit}" for c in targets})
    if dry_run:
        return {"files": files, "revisions": revisions, "size": size, "freed": 0}
    if lab.jobs.busy:
        raise LabBusyError("a job is running: prune when it is done")
    if any(d.state == "running" for d in lab.downloads.values()):
        raise LabBusyError("a download is in progress: prune when it is done")
    if chat_rev is not None and any(f["in_chat"] for f in files):
        raise LabBusyError("the chat is using one of these models: unload it or narrow the filter")
    if expect is not None and sorted(expect) != revisions:
        raise LabBusyError("the cache changed since the preview: preview again")
    if not targets:
        return {"files": [], "revisions": [], "size": 0, "freed": 0}
    _, freed = lab.store.prune(older_than_days=older_than_days, repo=repo, dry_run=False)
    return {"files": files, "revisions": revisions, "size": size, "freed": freed}


def pull(lab: Lab, ref_text: str, *, favorite: bool) -> Download:
    """`llb models pull <ref> [--favorite]`, through the lab's download (with progress)."""
    text = ref_text.strip()
    if not is_hf(text):
        text = "hf://" + text.removeprefix("https://huggingface.co/")
    try:
        ref = parse_hf(text)
    except RefError as e:
        raise LabError(str(e)) from None
    try:
        listing = lab.store.client.list_repo(ref.repo, ref.revision)
        if ref.revision is not None:
            latest = lab.store.client.list_repo(ref.repo, None)
            if latest.commit != listing.commit:
                raise LabError(
                    f"the lab downloads the current revision of {ref.repo}: pull "
                    f"@{ref.revision} with `llb models pull {ref}`"
                )
        files = pick_files(listing, ref)
    except (HubNotFoundError, HubUnavailableError, ModelNotFoundError) as e:
        raise LabError(str(e)) from None
    return lab.download(ref.repo, files[0].name, favorite=favorite)

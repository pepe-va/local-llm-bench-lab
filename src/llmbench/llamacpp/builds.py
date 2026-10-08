"""Managed llama.cpp builds: one directory per (commit, backend, cmake flags).

Layout under `$LLB_HOME/builds/`:
    index.json                         ref key -> build dir name (so `run` works offline)
    <sha12>-<backend>[-<cmakehash>]/
        manifest.json                  ref, commit, backend, cmake, version, built_at
        cmake-build/bin/llama-*        binaries (shared libs found via rpath)
"""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llmbench.config.experiment import BuildSpec
from llmbench.llamacpp.introspect import IntrospectionError, binary_version
from llmbench.paths import builds_dir, llama_src_dir

REPO_URL = "https://github.com/ggml-org/llama.cpp"
TARGETS = ("llama-bench", "llama-server", "llama-cli")
BACKEND_CMAKE: dict[str, dict[str, str]] = {
    "cpu": {"GGML_NATIVE": "ON"},
    "cuda": {"GGML_CUDA": "ON"},
    "vulkan": {"GGML_VULKAN": "ON"},
    "metal": {"GGML_METAL": "ON"},
    "blas": {"GGML_BLAS": "ON", "GGML_BLAS_VENDOR": "OpenBLAS"},
}
BASE_CMAKE = {"CMAKE_BUILD_TYPE": "Release", "LLAMA_CURL": "OFF", "LLAMA_BUILD_TESTS": "OFF"}

type Log = Callable[[str], None]


class BuildError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedBuild:
    id: str
    bin_dir: Path
    commit: str
    version: str
    backend: str
    cmake: dict[str, str]
    managed: bool
    ref: str | None = None

    def tool(self, name: str) -> Path:
        return self.bin_dir / name

    def identity(self) -> dict[str, object]:
        """What makes two builds produce comparable numbers (used in fingerprints)."""
        return {"commit": self.commit, "backend": self.backend, "cmake": self.cmake}


def cmake_flags(backend: str, extra: dict[str, str | int | bool]) -> dict[str, str]:
    flags = BASE_CMAKE | BACKEND_CMAKE[backend]
    for k, v in extra.items():
        flags[k] = ("ON" if v else "OFF") if isinstance(v, bool) else str(v)
    return dict(sorted(flags.items()))


def _key(ref: str, backend: str, cmake: dict[str, str]) -> str:
    return f"{ref}|{backend}|{json.dumps(cmake, sort_keys=True)}"


def _dir_name(commit: str, backend: str, extra: dict[str, str | int | bool]) -> str:
    name = f"{commit[:12]}-{backend}"
    if extra:
        name += "-" + hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
    return name


def _index_path() -> Path:
    return builds_dir() / "index.json"


def _read_index() -> dict[str, str]:
    p = _index_path()
    return json.loads(p.read_text()) if p.is_file() else {}


def _write_index(index: dict[str, str]) -> None:
    _index_path().parent.mkdir(parents=True, exist_ok=True)
    _index_path().write_text(json.dumps(index, indent=2, sort_keys=True))


def resolve(spec: BuildSpec) -> ResolvedBuild:
    """Find an existing build for a spec. Never compiles (see `build`)."""
    if spec.bin_dir is not None:
        bin_dir = spec.bin_dir.expanduser().resolve()
        try:
            v = binary_version(bin_dir / "llama-bench")
        except IntrospectionError as e:
            raise BuildError(f"build {spec.id!r}: {e}") from None
        return ResolvedBuild(spec.id, bin_dir, v.commit, v.version, spec.backend, {}, False)

    assert spec.ref is not None  # guaranteed by BuildSpec validation
    flags = cmake_flags(spec.backend, spec.cmake)
    name = _read_index().get(_key(spec.ref, spec.backend, flags))
    manifest = builds_dir() / name / "manifest.json" if name else None
    if manifest is None or not manifest.is_file():
        raise BuildError(
            f"build {spec.id!r} ({spec.ref}, {spec.backend}) is not built yet: "
            f"run `llb llama build --ref {spec.ref} --backend {spec.backend}`"
        )
    return _from_manifest(spec.id, manifest)


def _from_manifest(build_id: str, manifest: Path) -> ResolvedBuild:
    m = json.loads(manifest.read_text())
    return ResolvedBuild(
        id=build_id,
        bin_dir=manifest.parent / "cmake-build" / "bin",
        commit=m["commit"],
        version=m["version"],
        backend=m["backend"],
        cmake=m["cmake"],
        managed=True,
        ref=m["ref"],
    )


def build(
    spec: BuildSpec, *, jobs: int | None = None, force: bool = False, log: Log = print
) -> ResolvedBuild:
    """Fetch `spec.ref` (shallow) and compile it, unless that exact build already exists."""
    if spec.ref is None:
        return resolve(spec)
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            raise BuildError(f"`{tool}` is required to build llama.cpp")

    src = llama_src_dir()
    commit = _checkout(src, spec.ref, log)
    flags = cmake_flags(spec.backend, spec.cmake)
    out = builds_dir() / _dir_name(commit, spec.backend, spec.cmake)
    manifest = out / "manifest.json"
    if manifest.is_file() and not force:
        log(f"already built: {out}")
    else:
        bdir = out / "cmake-build"
        defs = [f"-D{k}={v}" for k, v in flags.items()]
        _sh(["cmake", "-S", str(src), "-B", str(bdir), *defs], log)
        jobs = jobs or os.cpu_count() or 1
        _sh(["cmake", "--build", str(bdir), "-j", str(jobs), "--target", *TARGETS], log)
        version = binary_version(bdir / "bin" / "llama-bench").version
        manifest.write_text(
            json.dumps(
                {
                    "ref": spec.ref,
                    "commit": commit,
                    "backend": spec.backend,
                    "cmake": flags,
                    "version": version,
                    "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
    index = _read_index()
    index[_key(spec.ref, spec.backend, flags)] = out.name
    _write_index(index)
    return _from_manifest(spec.id, manifest)


def list_builds() -> list[ResolvedBuild]:
    root = builds_dir()
    if not root.is_dir():
        return []
    return [_from_manifest(p.parent.name, p) for p in sorted(root.glob("*/manifest.json"))]


def _checkout(src: Path, ref: str, log: Log) -> str:
    if not (src / ".git").is_dir():
        src.mkdir(parents=True, exist_ok=True)
        _sh(["git", "-C", str(src), "init", "-q"], log)
        _sh(["git", "-C", str(src), "remote", "add", "origin", REPO_URL], log)
    _sh(["git", "-C", str(src), "fetch", "-q", "--depth", "1", "origin", ref], log)
    _sh(["git", "-C", str(src), "checkout", "-q", "--detach", "FETCH_HEAD"], log)
    return _sh(["git", "-C", str(src), "rev-parse", "HEAD"], log, quiet=True).strip()


def _sh(cmd: list[str], log: Log, *, quiet: bool = False) -> str:
    if not quiet:
        log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-20:])
        raise BuildError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{tail}")
    return p.stdout


__all__ = ["BuildError", "ResolvedBuild", "asdict", "build", "list_builds", "resolve"]

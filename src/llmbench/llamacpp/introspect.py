"""What a given llama.cpp binary supports, read from its own `--help` / `--version`.

llama.cpp renames flags often (e.g. `-mmp` became `--load-mode`), so instead of
hard-coding a version table we validate every argv against the build it targets.
"""

import re
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_FLAG = re.compile(r"(?<![\w<|-])(--?[A-Za-z][\w+-]*)")
_MAX_DECL_INDENT = 8
_VERSION_NEW = re.compile(r"version:\s*(\S+)\s*\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)")
_VERSION_OLD = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)")


@dataclass(frozen=True)
class BinaryVersion:
    version: str
    build_number: int | None
    commit: str
    built_with: str = ""


class IntrospectionError(Exception):
    pass


def parse_help(text: str) -> frozenset[str]:
    """Flags declared in the left column of a llama.cpp `--help` output.

    Flags whose description says "the argument has been removed" are excluded: llama.cpp
    keeps them in `--help` only to print a migration hint.
    """
    flags: set[str] = set()
    current: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if "has been removed" in stripped:
            flags -= current
        # Declarations start near column 0; deeply indented lines continue a description
        # (and may mention other flags, e.g. "use --spec-draft-n-max instead").
        if not stripped.startswith("-") or len(line) - len(line.lstrip()) > _MAX_DECL_INDENT:
            continue
        # Columns are separated by 2+ spaces; aliases may be too (`-tb,   --threads-batch N`).
        # Keep leading segments that start with a dash; the first other one is the description.
        current = set()
        for segment in re.split(r"\s{2,}", stripped):
            if not segment.startswith("-"):
                break
            current.update(_FLAG.findall(segment))
        flags |= current
        if "has been removed" in stripped:
            flags -= current
    return frozenset(flags)


def parse_version(text: str) -> BinaryVersion:
    built_with = next((ln.strip() for ln in text.splitlines() if ln.startswith("built with")), "")
    if m := _VERSION_NEW.search(text):
        return BinaryVersion(m.group(1), int(m.group(2)), m.group(3), built_with)
    if m := _VERSION_OLD.search(text):
        return BinaryVersion(m.group(1), int(m.group(1)), m.group(2), built_with)
    raise IntrospectionError(f"unrecognized --version output: {text.strip()[:200]!r}")


def flags_in(argv: list[str]) -> list[str]:
    """Flag tokens of an argv (skips values such as `-1` or patterns)."""
    return [tok.split("=", 1)[0] for tok in argv[1:] if re.match(r"^--?[A-Za-z]", tok)]


def unknown_flags(argv: list[str], supported: frozenset[str]) -> list[str]:
    return [f for f in flags_in(argv) if f not in supported]


@cache
def binary_flags(binary: Path) -> frozenset[str]:
    return parse_help(_run(binary, "--help"))


@cache
def binary_version(binary: Path) -> BinaryVersion:
    return parse_version(_run(binary, "--version"))


def _run(binary: Path, flag: str) -> str:
    if not binary.is_file():
        raise IntrospectionError(f"binary not found: {binary}")
    try:
        p = subprocess.run(
            [str(binary), flag], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError as e:
        raise IntrospectionError(f"cannot execute {binary}: {e}") from None
    return p.stdout + p.stderr

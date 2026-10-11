"""The lab's experiment editor: the same YAML files and commands as the terminal.

Experiments live where the CLI expects them (`experiments/`), so relative paths
(`_profiles/pc.yaml`, `../suites/...`) mean the same thing in both. The YAML text is the
source of truth: the form sends one change at a time and `patch` applies it with a
round-trip load, so comments and layout survive. The form itself is generated in the
browser from `Experiment`'s JSON schema, so every field of the YAML is reachable.
"""

import os
import re
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from llmbench.config.experiment import BenchPhase, Experiment
from llmbench.config.llama import LlamaConfig
from llmbench.lab.models import LocalModel
from llmbench.llamacpp.argv import bench_argv, sampling_flags, server_argv
from llmbench.serve.details import SETTING_HELP

_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml$")
_YAML11_WORDS = {"on", "off", "yes", "no", "y", "n", "true", "false"}
"""Strings a YAML 1.1 reader would take for booleans: always written quoted."""
Path_ = list[str | int]
_FLOW_WIDTH = 80
"""A mapping of scalars is written on one line (`{ id: x, source: y }`) only if it fits."""


class EditorError(Exception):
    """Bad input for the editor (HTTP 400). `line` is 1-based when known."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


def _rt() -> YAML:
    """Round-trip YAML, written in the style of the files in `experiments/`."""
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 100
    return y


# -- reading and changing YAML -------------------------------------------------------


def parse(text: str) -> dict[str, Any]:
    try:
        data = YAML(typ="safe").load(text)  # pyright: ignore[reportUnknownMemberType]
    except YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        raise EditorError(f"YAML: {e}", mark.line + 1 if mark else None) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise EditorError("an experiment is a mapping (key: value) at the top level")
    return cast(dict[str, Any], data)


def patch(text: str, path: Path_, value: Any) -> str:
    """Set (or, with `None`, remove) one value, keeping comments and layout."""
    if not path:
        raise EditorError("empty path")
    try:
        doc: Any = _rt().load(text) or CommentedMap()  # pyright: ignore[reportUnknownMemberType]
    except YAMLError as e:
        raise EditorError(f"fix the YAML first: {e}") from None
    if value is None:
        _remove(doc, path)
    else:
        _set(doc, path, _to_yaml(value))
    out = StringIO()
    _rt().dump(doc, out)  # pyright: ignore[reportUnknownMemberType]
    return out.getvalue()


def _set(node: Any, path: Path_, value: Any) -> None:
    *parents, last = path
    for i, key in enumerate(parents):
        nxt = path[i + 1]
        child = _get(node, key)
        if child is None:
            child = CommentedSeq() if isinstance(nxt, int) else CommentedMap()
            _put(node, key, child)
        node = child
    _put(node, last, value)


def _get(node: Any, key: str | int) -> Any:
    if isinstance(key, int):
        seq = _as_list(node)
        return seq[key] if seq is not None and key < len(seq) else None
    mapping = _as_dict(node)
    return mapping.get(key) if mapping is not None else None


def _as_list(node: Any) -> list[Any] | None:
    return cast(list[Any], node) if isinstance(node, list) else None


def _as_dict(node: Any) -> dict[str, Any] | None:
    return cast(dict[str, Any], node) if isinstance(node, dict) else None


def _put(node: Any, key: str | int, value: Any) -> None:
    if isinstance(key, int):
        seq = _as_list(node)
        if seq is None:
            raise EditorError(f"not a list at index {key}")
        if key == len(seq):
            seq.append(value)
        else:
            seq[key] = value
        return
    mapping = _as_dict(node)
    if mapping is None:
        raise EditorError(f"not a mapping at {key!r}")
    mapping[key] = value


def _remove(node: Any, path: Path_) -> None:
    chain = [node]
    for key in path[:-1]:
        node = _get(node, key)
        if node is None:
            return
        chain.append(node)
    last = path[-1]
    seq, mapping = _as_list(chain[-1]), _as_dict(chain[-1])
    if isinstance(last, int) and seq is not None and last < len(seq):
        del seq[last]
    elif isinstance(last, str) and mapping is not None:
        mapping.pop(last, None)
    # drop mappings left empty by the removal (but never the document itself)
    for depth in range(len(chain) - 1, 0, -1):
        key, holder = path[depth - 1], _as_dict(chain[depth - 1])
        if _as_dict(chain[depth]) == {} and isinstance(key, str) and holder is not None:
            holder.pop(key, None)
        else:
            break


def _flow(node: Any) -> Any:
    """Write a collection of scalars on one line, like the files in `experiments/`."""
    node.fa.set_flow_style()
    return node


def _to_yaml(value: Any) -> Any:
    if isinstance(value, str):
        return DoubleQuotedScalarString(value) if value.lower() in _YAML11_WORDS else value
    if isinstance(value, list):
        items: list[Any] = [_to_yaml(v) for v in cast(list[Any], value)]
        seq = CommentedSeq(items)
        return _flow(seq) if all(not isinstance(v, dict | list) for v in items) else seq
    if isinstance(value, dict):
        entries = {str(k): _to_yaml(v) for k, v in cast(dict[str, Any], value).items()}
        m = CommentedMap(entries)
        short = sum(len(str(k)) + len(str(v)) + 4 for k, v in entries.items()) <= _FLOW_WIDTH
        scalars = all(not isinstance(v, dict | list) for v in entries.values())
        return _flow(m) if scalars and short else m
    return value


# -- schema and help -------------------------------------------------------------------


@lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    return Experiment.model_json_schema()


@lru_cache(maxsize=1)
def field_help() -> dict[str, str]:
    """`group.field` of the llama.cpp settings -> what it becomes on the command line.

    Derived by setting each field alone and diffing the argv that `argv.py` builds, so the
    flag spellings still live only there.
    """
    out: dict[str, str] = dict(SETTING_HELP)
    model, binary = Path("m.gguf"), Path("llama")
    base = LlamaConfig()
    base_server = _pairs(server_argv(binary, model, base, port=1).args)
    base_bench = _pairs(bench_argv(binary, model, base, BenchPhase()).args)
    defs = LlamaConfig.model_json_schema()["$defs"]
    for group in ("load", "kv", "gpu", "rope", "speculative", "server", "sampling"):
        cls = cast(type[BaseModel], type(getattr(base, group)))
        props: dict[str, Any] = defs[cls.__name__]["properties"]
        for name, prop in props.items():
            key = f"{group}.{name}"
            if group == "sampling":
                flags = _new_flags(sampling_flags({name: s}).args for s in _samples(prop))
                where = f"sent per request as `{name}`" + (f" (server: {flags})" if flags else "")
                out[key] = _join(out.get(key), where)
                continue
            server, bench = "", ""
            for sample in _samples(prop):
                try:
                    cfg = LlamaConfig.model_validate({group: {name: sample}})
                except ValidationError:
                    continue
                server = _diff(server_argv(binary, model, cfg, port=1).args, base_server)
                bench = _diff(bench_argv(binary, model, cfg, BenchPhase()).args, base_bench)
                break
            flags = _flags_text(server, bench)
            if flags:
                out[key] = _join(out.get(key), flags)
    return out


def _samples(prop: dict[str, Any]) -> list[Any]:
    """Valid-looking values to probe a field with."""
    options = [o for o in prop.get("anyOf", [prop]) if o.get("type") != "null"]
    for o in options:
        if "enum" in o:
            return list(o["enum"])[:1]
        kind = o.get("type")
        if kind == "boolean":
            return [True]
        if kind == "integer":
            low = max(1, int(o.get("minimum", 1)))
            default = prop.get("default")
            return [low + 1 if default == low else low]  # differ from the default
        if kind == "number":
            return [1.0]
        if kind == "string":
            return ["1", "f16", "x", "/tmp/x.gguf"]
    return []


def _pairs(args: list[str]) -> dict[str, str | None]:
    """`--flag value` / `--toggle` -> {flag: value or None}."""
    out: dict[str, str | None] = {}
    for i, a in enumerate(args):
        if a.startswith("-") and not _number(a):
            nxt = args[i + 1] if i + 1 < len(args) else None
            out[a] = (
                nxt if nxt is not None and not (nxt.startswith("-") and not _number(nxt)) else None
            )
    return out


def _diff(args: list[str], base: dict[str, str | None]) -> str:
    """Flags that are new or take another value than in the default argv."""
    now = _pairs(args)
    return " ".join(f for f, v in now.items() if f not in base or base[f] != v)


def _new_flags(argvs: Any) -> str:
    for args in argvs:
        flags = [a for a in cast(list[str], args) if a.startswith("-") and not _number(a)]
        if flags:
            return " ".join(flags)
    return ""


def _number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _flags_text(server: str, bench: str) -> str:
    if server and server == bench:
        return f"flag {server}"
    parts = [f"llama-server {server}" if server else "", f"llama-bench {bench}" if bench else ""]
    return " · ".join(p for p in parts if p)


def _join(first: str | None, second: str) -> str:
    return f"{first} ({second})" if first else second


# -- files and commands ----------------------------------------------------------------


class Experiments:
    """The YAML files of one experiments folder, and the CLI commands run on them."""

    def __init__(self, folder: Path, root: Path) -> None:
        self.folder = folder
        self.root = root

    def files(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(p.name for p in self.folder.glob("*.y*ml") if _FILE.match(p.name))

    def profiles(self) -> list[str]:
        """`extends` entries, relative to the experiments folder."""
        return sorted(f"_profiles/{p.name}" for p in (self.folder / "_profiles").glob("*.yaml"))

    def suites(self) -> list[str]:
        folder = self.root / "suites"
        return sorted(os.path.relpath(p, self.folder) for p in folder.glob("*.yaml"))

    def path(self, name: str) -> Path:
        if not _FILE.match(name):
            raise EditorError(f"invalid file name {name!r}: use letters, digits, . _ - and .yaml")
        return self.folder / name

    def read(self, name: str) -> str:
        p = self.path(name)
        if not p.is_file():
            raise EditorError(f"no such experiment: {name}")
        return p.read_text()

    def save(self, name: str, text: str, *, overwrite: bool) -> Path:
        parse(text)  # never save something that is not even YAML
        p = self.path(name)
        if p.exists() and not overwrite:
            raise EditorError(f"{name} already exists")
        self.folder.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def check(self, text: str, command: str, models: list[str]) -> dict[str, Any]:
        """`llb validate|plan` on unsaved text: a hidden draft next to the real files."""
        if command not in ("validate", "plan"):
            raise EditorError(f"unknown check {command!r}")
        parse(text)
        self.folder.mkdir(parents=True, exist_ok=True)
        draft = self.folder / f".llb-lab-{uuid.uuid4().hex[:8]}.yaml"
        draft.write_text(text)
        try:
            args = [command, str(draft)]
            if command == "plan":
                for m in models:
                    args += ["-m", m]
            out = self.run_sync(args)
        finally:
            draft.unlink(missing_ok=True)
        out["output"] = out["output"].replace(str(draft), "<draft>").replace(draft.name, "<draft>")
        return out

    def run_sync(self, args: list[str], timeout: float = 300) -> dict[str, Any]:
        """A quick llb command, with its output as the terminal would show it."""
        with tempfile.TemporaryFile() as out:
            try:
                p = subprocess.run(
                    [sys.executable, "-m", "llmbench", *args],
                    cwd=self.root, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=os.environ | {"COLUMNS": "120", "NO_COLOR": "1"}, timeout=timeout,
                    check=False,
                )  # fmt: skip
                code: int | None = p.returncode
            except subprocess.TimeoutExpired:
                code = None
            out.seek(0)
            text = out.read().decode(errors="replace")
        if code is None:
            text += f"\n(stopped after {timeout:.0f} s)"
        return {"ok": code == 0, "code": code, "output": text}


def new_experiment(
    models: list[LocalModel],
    *,
    folder: Path,
    root: Path,
    profile: str | None,
    presets: list[str],
    name: str | None = None,
) -> dict[str, Any]:
    """A starting point: the chosen models, this machine's profile and test presets."""
    from llmbench.lab.experiment import PRESETS, experiment_name  # noqa: PLC0415

    chosen = [p for p in PRESETS if p.key in presets]
    doc: dict[str, Any] = {"version": 1}
    if profile:
        doc["extends"] = [profile, "_profiles/deterministic.yaml"]
    doc["experiment"] = {
        "name": name or (experiment_name(models[0]) if len(models) == 1 else "lab-experiment"),
        "description": "Made in the lab"
        + (": " + ", ".join(p.title for p in chosen) if chosen else ""),
        "tags": ["lab"],
    }
    doc["models"] = [
        {"id": experiment_name(m).removeprefix("lab-"), "source": m.source, "retention": "keep"}
        for m in models
    ]
    doc["defaults"] = {"load": {"ctx_size": 4096}}
    phases: dict[str, Any] = {"bench": {"enabled": False}}
    for p in chosen:
        if p.phase == "bench":
            phases["bench"] = {"n_prompt": [512], "n_gen": [128], "repetitions": 3}
        else:
            suites = [os.path.relpath(root / "suites" / s, folder) for s in p.suites]
            phases[p.phase] = {"enabled": True, "suites": suites}
    doc["phases"] = phases
    return doc


def dump(doc: dict[str, Any]) -> str:
    """A new document in the style of `experiments/` (flow style for short leaves)."""
    out = StringIO()
    _rt().dump(_to_yaml(doc), out)  # pyright: ignore[reportUnknownMemberType]
    return out.getvalue()

"""YAML -> validated `Experiment`: `extends`, `${env:VAR}` interpolation, readable errors."""

import os
import re
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from llmbench.config.experiment import Experiment
from llmbench.config.suite import Suite

_ENV = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Configuration problem, already formatted for humans (one issue per line)."""


def load_experiment(path: Path) -> Experiment:
    path = path.expanduser().resolve()
    raw, root_doc = _load_tree(path, stack=())
    try:
        exp = Experiment.model_validate(_interpolate(raw))
    except ValidationError as e:
        raise ConfigError(_format_errors(e, path, root_doc)) from None
    for phase in (exp.phases.latency, exp.phases.quality):
        for suite in phase.suites if phase.enabled else []:
            load_suite(suite)  # fail early on a broken suite
    return exp


def _read(path: Path) -> CommentedMap:
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    try:
        doc: Any = YAML(typ="rt").load(path.read_text(encoding="utf-8"))  # pyright: ignore[reportUnknownMemberType]
    except YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML\n{e}") from None
    if doc is None:
        return CommentedMap()
    if not isinstance(doc, CommentedMap):
        raise ConfigError(f"{path}: top level must be a mapping")
    return doc


def _load_tree(path: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], CommentedMap]:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ConfigError(f"circular `extends`: {chain}")
    doc = _read(path)
    extends: Any = cast(dict[str, Any], doc).get("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ConfigError(f"{path}: `extends` must be a path or a list of paths")

    merged: dict[str, Any] = {}
    for base in extends:  # pyright: ignore[reportUnknownVariableType]
        base_path = (path.parent / str(base)).expanduser().resolve()  # pyright: ignore[reportUnknownArgumentType]
        base_data, _ = _load_tree(base_path, (*stack, path))
        merged = deep_merge(merged, base_data)
    own = _plain(doc)
    own.pop("extends", None)
    _anchor_paths(own, path.parent)
    return deep_merge(merged, own), doc


def _anchor_paths(data: dict[str, Any], base: Path) -> None:
    """Make file references relative to the YAML that declares them (also through `extends`)."""
    for phase in ("latency", "quality"):
        node = _node(data, "phases", phase)
        suites = node.get("suites") if node is not None else None
        if node is not None and isinstance(suites, list):
            node["suites"] = [_anchored(str(s), base) for s in cast(list[Any], suites)]


def _node(data: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    node: Any = data
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = cast(dict[str, Any], node).get(k)
    return cast(dict[str, Any], node) if isinstance(node, dict) else None


def _anchored(value: str, base: Path) -> str:
    p = Path(value).expanduser()
    return value if p.is_absolute() else os.path.normpath(base / p)


def load_suite(path: Path) -> Suite:
    path = path.expanduser().resolve()
    doc = _plain(_read(path))
    synthetic = _node(doc, "source", "synthetic")
    corpus = synthetic.get("corpus") if synthetic is not None else None
    if synthetic is not None and isinstance(corpus, str) and corpus != "words":
        synthetic["corpus"] = _anchored(corpus, path.parent)
    file = _node(doc, "source", "file")
    if file is not None and isinstance(file.get("path"), str):
        file["path"] = _anchored(str(file["path"]), path.parent)
    try:
        return Suite.model_validate(_interpolate(doc))
    except ValidationError as e:
        raise ConfigError(_format_errors(e, path, _read(path))) from None


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively; everything else (lists included) is replaced."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[k] = v
    return out


def _plain(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(k): _plain(v) for k, v in cast(dict[Any, Any], node).items()}
    if isinstance(node, list):
        return [_plain(v) for v in cast(list[Any], node)]
    return node


def _interpolate(node: Any) -> Any:
    if isinstance(node, str):
        return _ENV.sub(_env_value, node)
    if isinstance(node, dict):
        return {k: _interpolate(v) for k, v in node.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(node, list):
        return [_interpolate(v) for v in node]  # pyright: ignore[reportUnknownVariableType]
    return node


def _env_value(m: re.Match[str]) -> str:
    name: str = m.group(1)
    default: str | None = m.group(2)
    value = os.environ.get(name, default)
    if value is None:
        raise ConfigError(f"environment variable {name} is not set (use ${{env:{name}:-default}})")
    return value


def _format_errors(e: ValidationError, path: Path, doc: CommentedMap) -> str:
    lines = [f"Configuration error in {path}:"]
    for err in e.errors():
        loc = _loc_str(err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        line = _line_of(doc, err["loc"])
        where = f"{loc} (line {line})" if line else loc
        lines.append(f"  {where}: {msg}" if loc else f"  {msg}")
    return "\n".join(lines)


def _loc_str(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _line_of(doc: CommentedMap, loc: tuple[int | str, ...]) -> int | None:
    """Best effort: 1-based line of `loc` in the root file (not in `extends` bases)."""
    node: Any = doc
    line: int | None = None
    for part in loc:
        lc: Any = getattr(node, "lc", None)
        if isinstance(node, CommentedMap) and part in node:
            line = int(lc.key(part)[0]) + 1
            node = cast(Any, node)[part]
        elif isinstance(node, CommentedSeq) and isinstance(part, int) and part < len(node):
            line = int(lc.item(part)[0]) + 1
            node = cast(Any, node)[part]
        else:
            break
    return line

"""Expansion of the `matrix` section into concrete variants.

Semantics:
- every dotted key (`load.threads: [1, 2, 4]`) is an axis; axes are crossed (cartesian product);
- each `zip` group is a single axis whose lists advance together (paired values);
- `exclude` drops combinations matching all of its keys;
- `include` appends extra combinations verbatim (deduplicated).
Keys are dotted paths into `LlamaConfig` (the experiment's `defaults`).
"""

import itertools
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from llmbench.config.llama import LlamaConfig

type Variant = dict[str, Any]
"""Dotted path -> value. An empty variant means "the defaults as-is"."""

_FREE_FORM = {"env"}  # dict-typed fields whose keys may be created by the matrix


class Matrix(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    zip: list[dict[str, list[Any]]] = []
    include: list[dict[str, Any]] = []
    exclude: list[dict[str, Any]] = []

    @property
    def axes(self) -> dict[str, list[Any]]:
        return {k: cast(list[Any], v) for k, v in (self.model_extra or {}).items()}

    @model_validator(mode="after")
    def _shape(self) -> Self:
        seen: set[str] = set()
        for path, values in (self.model_extra or {}).items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"matrix.{path} must be a non-empty list")
            seen.add(path)
        for i, group in enumerate(self.zip):
            lengths = {len(v) for v in group.values()}
            if len(lengths) != 1 or 0 in lengths:
                raise ValueError(f"matrix.zip[{i}]: lists must be non-empty and equal length")
            if overlap := seen & group.keys():
                raise ValueError(f"matrix.zip[{i}]: keys already used elsewhere: {sorted(overlap)}")
            seen |= group.keys()
        return self

    def expand(self) -> list[Variant]:
        dims: list[list[Variant]] = [
            [{path: v} for v in values] for path, values in self.axes.items()
        ]
        for group in self.zip:
            n = len(next(iter(group.values())))
            dims.append([{p: vals[i] for p, vals in group.items()} for i in range(n)])

        variants: list[Variant] = []
        for combo in itertools.product(*dims):
            variant: Variant = {}
            for part in combo:
                variant |= part
            if not any(_matches(variant, ex) for ex in self.exclude):
                variants.append(variant)
        for inc in self.include:
            if inc not in variants:
                variants.append(dict(inc))
        return variants

    def validate_against(self, defaults: LlamaConfig) -> None:
        """Fail early with the exact matrix location of any invalid key or value."""
        for path, values in self.axes.items():
            for i, v in enumerate(values):
                _check(defaults, {path: v}, f"matrix.{path}[{i}]")
        for g, group in enumerate(self.zip):
            for path, values in group.items():
                for i, v in enumerate(values):
                    _check(defaults, {path: v}, f"matrix.zip[{g}].{path}[{i}]")
        for i, inc in enumerate(self.include):
            _check(defaults, inc, f"matrix.include[{i}]")
        for i, ex in enumerate(self.exclude):
            for path in ex:
                try:
                    _resolve_path(defaults.model_dump(), path, f"matrix.exclude[{i}]")
                except KeyError as e:
                    raise ValueError(e.args[0]) from None


def apply_variant(defaults: LlamaConfig, variant: Variant) -> LlamaConfig:
    data = defaults.model_dump()
    for path, value in variant.items():
        parent, key = _resolve_path(data, path, f"matrix key {path!r}")
        parent[key] = value
    return LlamaConfig.model_validate(data)


def variant_label(variant: Variant) -> str:
    return ", ".join(f"{k}={v}" for k, v in variant.items()) or "defaults"


def _check(defaults: LlamaConfig, variant: Variant, where: str) -> None:
    try:
        apply_variant(defaults, variant)
    except ValidationError as e:
        msg = "; ".join(err["msg"] for err in e.errors())
        raise ValueError(f"{where}: {msg}") from None
    except KeyError as e:
        raise ValueError(f"{where}: {e.args[0]}") from None


def _resolve_path(data: dict[str, Any], path: str, where: str) -> tuple[dict[str, Any], str]:
    parts = path.split(".")
    node: Any = data
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{where}: unknown config path {'.'.join(parts[: depth + 1])!r}")
        node = cast(dict[str, Any], node)[part]
    last = parts[-1]
    free = len(parts) == 2 and parts[0] in _FREE_FORM  # noqa: PLR2004
    if not isinstance(node, dict) or (last not in node and not free):
        raise KeyError(f"{where}: unknown config path {path!r}")
    return node, last  # pyright: ignore[reportUnknownVariableType]


def _matches(variant: Variant, pattern: dict[str, Any]) -> bool:
    return all(k in variant and variant[k] == v for k, v in pattern.items())

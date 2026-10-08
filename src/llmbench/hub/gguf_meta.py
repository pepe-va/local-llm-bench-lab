"""Minimal pure-Python GGUF reader: metadata and tensor shapes, never the weights.

Avoids depending on the `gguf` package (and numpy) just to read a header.
Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# gguf_type -> struct format (scalars)
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_STRING, _ARRAY = 8, 9

# llama_ftype (general.file_type)
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE",
}  # fmt: skip

# ggml_type (per tensor)
TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}  # fmt: skip

_MAX_KEPT_ARRAY = 16  # longer arrays (tokenizer vocab...) are summarized by length


class GGUFError(Exception):
    pass


@dataclass
class GGUFInfo:
    version: int
    metadata: dict[str, Any]
    tensor_count: int
    n_params: int
    tensor_types: dict[str, int] = field(default_factory=dict[str, int])
    """ggml type name -> number of parameters stored in that type."""

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    def arch(self, key: str) -> Any:
        return self.metadata.get(f"{self.architecture}.{key}")

    @property
    def file_type(self) -> str | None:
        ft = self.metadata.get("general.file_type")
        return FILE_TYPES.get(ft, f"type_{ft}") if isinstance(ft, int) else None

    @property
    def quant(self) -> str | None:
        """Declared file type, else the type holding most parameters."""
        if self.file_type:
            return self.file_type
        return (
            max(self.tensor_types, key=self.tensor_types.__getitem__) if self.tensor_types else None
        )

    def summary(self) -> dict[str, Any]:
        """Compact, stable facts worth storing with every run."""
        return {
            "architecture": self.architecture,
            "name": self.metadata.get("general.name"),
            "size_label": self.metadata.get("general.size_label"),
            "file_type": self.file_type,
            "quant": self.quant,
            "n_params": self.n_params,
            "context_length": self.arch("context_length"),
            "block_count": self.arch("block_count"),
            "embedding_length": self.arch("embedding_length"),
            "head_count": self.arch("attention.head_count"),
            "head_count_kv": self.arch("attention.head_count_kv"),
            "expert_count": self.arch("expert_count"),
            "expert_used_count": self.arch("expert_used_count"),
            "gguf_version": self.version,
        }


def read_gguf(paths: Path | list[Path] | tuple[Path, ...]) -> GGUFInfo:
    """Read one file, or every shard of a split model (metadata from the first)."""
    files = [paths] if isinstance(paths, Path) else list(paths)
    first = _read_one(files[0])
    for shard in files[1:]:
        other = _read_one(shard)
        first.tensor_count += other.tensor_count
        first.n_params += other.n_params
        for k, v in other.tensor_types.items():
            first.tensor_types[k] = first.tensor_types.get(k, 0) + v
    return first


def _read_one(path: Path) -> GGUFInfo:
    try:
        with path.open("rb") as f:
            return _parse(f)
    except (struct.error, UnicodeDecodeError) as e:
        raise GGUFError(f"{path}: truncated or corrupt GGUF header ({e})") from None


def _parse(f: BinaryIO) -> GGUFInfo:
    if f.read(4) != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    (version,) = _unpack(f, "I")
    if version < 2:  # noqa: PLR2004
        raise GGUFError(f"GGUF v{version} is not supported (need v2+)")
    tensor_count, kv_count = _unpack(f, "QQ")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _string(f)
        (vtype,) = _unpack(f, "I")
        metadata[key] = _value(f, vtype)

    n_params = 0
    by_type: Counter[str] = Counter()
    for _ in range(tensor_count):
        _string(f)  # name
        (n_dims,) = _unpack(f, "I")
        dims = _unpack(f, "Q" * n_dims)
        (ttype,) = _unpack(f, "I")
        _unpack(f, "Q")  # offset
        count = math.prod(dims)
        n_params += count
        by_type[TENSOR_TYPES.get(ttype, f"type_{ttype}")] += count
    return GGUFInfo(version, metadata, tensor_count, n_params, dict(by_type))


def _unpack(f: BinaryIO, fmt: str) -> tuple[Any, ...]:
    fmt = "<" + fmt
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _string(f: BinaryIO) -> str:
    (n,) = _unpack(f, "Q")
    return f.read(n).decode("utf-8")


def _value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _unpack(f, _SCALARS[vtype])[0]
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        (itype, n) = _unpack(f, "IQ")
        if itype in _SCALARS and n > _MAX_KEPT_ARRAY:
            f.seek(n * struct.calcsize("<" + _SCALARS[itype]), 1)  # skip without decoding
            return f"<array of {n}>"
        items = [_value(f, itype) for _ in range(n)]
        return items if n <= _MAX_KEPT_ARRAY else f"<array of {n}>"
    raise GGUFError(f"unknown GGUF value type {vtype}")

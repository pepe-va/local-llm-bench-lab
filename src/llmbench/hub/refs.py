"""Model references.

    hf://<org>/<repo>[@<revision>][:<QUANT>]       quant picked by filename, e.g. :Q4_K_M
    hf://<org>/<repo>[@<revision>]/<path/to/file.gguf>
    file://<path> | <path>                          local file (relative: ./ then models/)

`revision` is a branch, tag or commit sha. Without one, the repo's default branch is used
and the resolved commit sha is recorded with every run.
"""

import re
from dataclasses import dataclass

_HF = re.compile(
    r"^hf://(?P<repo>[\w.-]+/[\w.-]+)"
    r"(?:@(?P<rev>[\w.-]+))?"
    r"(?:(?::(?P<quant>[\w.-]+))|(?:/(?P<file>.+\.gguf)))?$"
)
_SHARD = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
DEFAULT_QUANT = "Q4_K_M"  # same default as llama.cpp's -hf


class RefError(ValueError):
    pass


@dataclass(frozen=True)
class HfRef:
    repo: str
    revision: str | None = None
    quant: str | None = None
    file: str | None = None

    @property
    def org(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]

    @property
    def folder(self) -> str:
        """Directory name under `models/` (mirrors the HF cache's `models--org--name`)."""
        return f"{self.org}--{self.name}"

    def __str__(self) -> str:
        s = f"hf://{self.repo}"
        if self.revision:
            s += f"@{self.revision}"
        if self.quant:
            s += f":{self.quant}"
        elif self.file:
            s += f"/{self.file}"
        return s


def parse_hf(source: str) -> HfRef:
    m = _HF.match(source)
    if m is None:
        raise RefError(
            f"invalid Hugging Face reference {source!r}; expected "
            "hf://org/repo[@revision][:QUANT] or hf://org/repo[@revision]/file.gguf"
        )
    return HfRef(m["repo"], m["rev"], m["quant"], m["file"])


def is_hf(source: str) -> bool:
    return source.startswith("hf://")


def shard_info(filename: str) -> tuple[str, int, int] | None:
    """`model-00002-of-00003.gguf` -> ("model", 2, 3)."""
    m = _SHARD.match(filename.rsplit("/", 1)[-1])
    if m is None:
        return None
    return m["prefix"], int(m["idx"]), int(m["total"])


def quant_matches(filename: str, quant: str) -> bool:
    """Case-insensitive quant token with name boundaries.

    `Q2_K` matches `Qwen3-1.7B-Q2_K.gguf` but not `Qwen3-1.7B-Q2_K_L.gguf`.
    """
    base = filename.rsplit("/", 1)[-1]
    pattern = rf"(?i)(?:^|[-_.]){re.escape(quant)}(?:[-.]|$)"
    return re.search(pattern, base) is not None

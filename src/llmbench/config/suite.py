"""Suites: the requests sent to llama-server (`suites/*.yaml`).

Sources: `synthetic` (exact token counts, for performance), inline `items`, a `file`, or a
Hugging Face dataset. Items may carry a reference and an evaluator (quality phase).
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import BeforeValidator, Field, model_validator

from llmbench.config.base import Strict
from llmbench.config.evaluators import EvaluatorSpec
from llmbench.config.experiment import Slug
from llmbench.config.llama.sampling import SamplingConfig


class Fixed(Strict):
    dist: Literal["fixed"] = "fixed"
    value: Annotated[int, Field(gt=0)]


class Uniform(Strict):
    dist: Literal["uniform"]
    min: Annotated[int, Field(gt=0)]
    max: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _order(self) -> Self:
        if self.min > self.max:
            raise ValueError("min must be <= max")
        return self


class Normal(Strict):
    dist: Literal["normal"]
    mean: Annotated[float, Field(gt=0)]
    std: Annotated[float, Field(ge=0)]
    min: Annotated[int, Field(gt=0)] = 1
    max: Annotated[int, Field(gt=0)] | None = None


def _shorthand(v: object) -> object:
    """`512` or `{value: 512}` mean a fixed size."""
    if isinstance(v, int):
        return {"dist": "fixed", "value": v}
    if isinstance(v, dict):
        d = cast(dict[str, Any], v)
        return d if "dist" in d else {"dist": "fixed", **d}
    return v


type Distribution = Annotated[
    Fixed | Uniform | Normal, Field(discriminator="dist"), BeforeValidator(_shorthand)
]


class Synthetic(Strict):
    """Random-but-reproducible prompts with an exact number of input tokens per model."""

    prompt_tokens: Distribution
    output_tokens: Distribution
    count: Annotated[int, Field(gt=0, le=100_000)] = 20
    seed: int = 42
    corpus: Literal["words"] | Path = "words"
    """`words`: built-in deterministic word stream; or a UTF-8 text file to sample from."""


class Item(Strict):
    """An inline request. `input` is optional text that templates and evaluators can use."""

    id: str | None = None
    category: str | None = None
    system: str | None = None
    prompt: str
    input: str | None = None
    reference: str | None = None
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    evaluator: EvaluatorSpec | None = None


class FileSource(Strict):
    path: Path
    """`.jsonl`, `.json` (list of objects), `.csv` or `.parquet`; relative to the suite file."""


class HfDataset(Strict):
    """A Hugging Face dataset, read from its parquet conversion (`refs/convert/parquet`)."""

    repo: str
    config: str = "default"
    split: str = "test"
    revision: str | None = None
    """Commit of the `refs/convert/parquet` branch; the resolved one is always recorded."""
    limit: Annotated[int, Field(gt=0)] | None = None
    shuffle_seed: int | None = None


class SuiteSource(Strict):
    """Exactly one of: synthetic, items, file, hf_dataset."""

    synthetic: Synthetic | None = None
    items: list[Item] | None = None
    file: FileSource | None = None
    hf_dataset: HfDataset | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        set_ = [k for k in ("synthetic", "items", "file", "hf_dataset") if getattr(self, k)]
        if len(set_) != 1:
            raise ValueError(f"set exactly one source, got {set_ or 'none'}")
        return self


class Columns(Strict):
    """Maps dataset/file columns to request fields (only for `file` / `hf_dataset`)."""

    prompt: str = "prompt"
    input: str | None = None
    reference: str | None = None
    id: str | None = None
    category: str | None = None
    system: str | None = None


class Suite(Strict):
    version: Literal[1] = 1
    name: Slug
    description: str = ""
    endpoint: Literal["completion", "chat"] = "completion"
    source: SuiteSource
    columns: Columns = Columns()
    system: str | None = None
    """Default system prompt (chat endpoint)."""
    template: str | None = None
    """Python format string over the row/item fields, e.g. "Solve:\n{prompt}"."""
    max_tokens: Annotated[int, Field(gt=0)] = 256
    evaluator: EvaluatorSpec | None = None
    """Default evaluator for every item without its own."""
    sampling: SamplingConfig = SamplingConfig()
    """Overrides on top of the experiment's `defaults.sampling` for this suite."""

    @property
    def is_synthetic(self) -> bool:
        return self.source.synthetic is not None

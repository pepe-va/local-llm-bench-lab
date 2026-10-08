"""Suites: the requests sent to llama-server (`suites/*.yaml`).

Phase 3 provides the `synthetic` source (exact token counts for performance work).
File and Hugging Face dataset sources, and evaluators, arrive with the quality phase.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import BeforeValidator, Field, model_validator

from llmbench.config.base import Strict
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


class SuiteSource(Strict):
    synthetic: Synthetic


class Suite(Strict):
    version: Literal[1] = 1
    name: Slug
    description: str = ""
    endpoint: Literal["completion"] = "completion"
    source: SuiteSource
    sampling: SamplingConfig = SamplingConfig()
    """Overrides on top of the experiment's `defaults.sampling` for this suite."""

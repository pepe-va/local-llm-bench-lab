from typing import Annotated, Any, Literal

from pydantic import Field

from llmbench.config.base import Strict

Sampler = Literal[
    "penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature",
    "infill", "adaptive_p",
]  # fmt: skip

Prob = Annotated[float, Field(ge=0, le=1)]
NonNeg = Annotated[float, Field(ge=0)]


class SamplingConfig(Strict):
    """Generation parameters, sent in the body of each request (not as server flags).

    Keeping them per request means a matrix over sampling reuses the same server process.
    `None` = do not send, the server default applies (note: llama.cpp defaults to temperature 0.8;
    use the `deterministic` profile for reproducible outputs).
    """

    temperature: NonNeg | None = None
    dynatemp_range: NonNeg | None = None
    dynatemp_exponent: NonNeg | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    top_p: Prob | None = None
    min_p: Prob | None = None
    typical_p: Prob | None = None
    top_n_sigma: float | None = None
    xtc_probability: Prob | None = None
    xtc_threshold: Prob | None = None
    repeat_penalty: NonNeg | None = None
    repeat_last_n: Annotated[int, Field(ge=-1)] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    dry_multiplier: NonNeg | None = None
    dry_base: NonNeg | None = None
    dry_allowed_length: Annotated[int, Field(ge=0)] | None = None
    dry_penalty_last_n: Annotated[int, Field(ge=-1)] | None = None
    dry_sequence_breakers: list[str] | None = None
    mirostat: Literal[0, 1, 2] | None = None
    mirostat_tau: NonNeg | None = None
    mirostat_eta: NonNeg | None = None
    samplers: list[Sampler] | None = None
    seed: Annotated[int, Field(ge=-1)] | None = None
    min_keep: Annotated[int, Field(ge=0)] | None = None
    n_probs: Annotated[int, Field(ge=0)] | None = None
    ignore_eos: bool | None = None
    stop: list[str] | None = None
    grammar: str | None = None
    json_schema: dict[str, Any] | None = None
    logit_bias: list[tuple[int, float]] | None = None

    def body(self) -> dict[str, Any]:
        """Fields to send in a /completion request (only the ones that are set)."""
        return self.model_dump(exclude_none=True)

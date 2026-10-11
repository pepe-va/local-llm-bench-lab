"""Evaluator configuration (`evaluator:` in suites and items). Logic lives in `llmbench.evaluation`.

Deterministic evaluators come first; an LLM judge exists but is never the default and its
scores are stored apart from objective ones (PRD §17-18).
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from llmbench.config.base import Strict

Normalization = Literal["strip", "lower", "collapse_ws", "strip_punct", "strip_accents"]


class Exact(Strict):
    """Pass when the whole answer equals the expected text (after normalization)."""

    type: Literal["exact"]
    value: str | None = None
    """Expected text; defaults to the item's `reference`."""
    normalize: list[Normalization] = ["strip", "collapse_ws"]


class Contains(Strict):
    """Pass when the answer contains the texts (score: the share found)."""

    type: Literal["contains"]
    values: Annotated[list[str], Field(min_length=1)]
    """Texts to look for, one per line."""
    mode: Literal["all", "any"] = "all"
    """`all`: every text must appear; `any`: one is enough."""
    case_sensitive: bool = False


class NotContains(Strict):
    """Pass when none of the texts appears in the answer."""

    type: Literal["not_contains"]
    values: Annotated[list[str], Field(min_length=1)]
    """Forbidden texts, one per line."""
    case_sensitive: bool = False


class Regex(Strict):
    """Pass when a regular expression (Python syntax) matches the answer."""

    type: Literal["regex"]
    pattern: str
    mode: Literal["search", "fullmatch"] = "search"
    ignore_case: bool = False


class RegexExtract(Strict):
    """Extract group 1 from the output and compare it with the reference (or `value`)."""

    type: Literal["regex_extract"]
    pattern: str
    reference_pattern: str | None = None
    """Optional pattern to extract the expected value from the reference too (e.g. GSM8K `####`)."""
    compare: Literal["exact", "numeric"] = "exact"
    value: str | None = None
    tolerance: Annotated[float, Field(ge=0)] = 1e-6


class Length(Strict):
    """Pass when the answer's length is within the limits."""

    type: Literal["length"]
    min_chars: Annotated[int, Field(ge=0)] | None = None
    max_chars: Annotated[int, Field(ge=0)] | None = None
    min_words: Annotated[int, Field(ge=0)] | None = None
    max_words: Annotated[int, Field(ge=0)] | None = None
    ratio_to: Literal["input", "reference"] | None = None
    """Compare the output length (words) with the item's `input` or `reference` text."""
    min_ratio: Annotated[float, Field(ge=0)] | None = None
    max_ratio: Annotated[float, Field(ge=0)] | None = None


class JsonSchema(Strict):
    """Pass when the answer (or its ```json block) is JSON valid against a schema."""

    type: Literal["json_schema"]
    schema_: Annotated[dict[str, Any], Field(alias="schema")]
    """A JSON Schema."""


class LlmJudge(Strict):
    """Subjective 1-5 rating per criterion by a judge model (see `phases.judge`)."""

    type: Literal["llm_judge"]
    criteria: Annotated[list[str], Field(min_length=1)] = [
        "grammar",
        "clarity",
        "instruction_following",
        "faithfulness",
    ]
    rubric: str | None = None


class All(Strict):
    """Pass when every part passes (score: the lowest)."""

    type: Literal["all"]
    of: Annotated[list["EvaluatorSpec"], Field(min_length=1)]


class AnyOf(Strict):
    """Pass when one part passes (score: the highest)."""

    type: Literal["any"]
    of: Annotated[list["EvaluatorSpec"], Field(min_length=1)]


class Weighted(Strict):
    """Weighted mean of the parts' scores; pass at `pass_threshold`."""

    type: Literal["weighted"]
    of: Annotated[list["WeightedPart"], Field(min_length=1)]
    pass_threshold: Annotated[float, Field(ge=0, le=1)] = 0.5


class WeightedPart(Strict):
    weight: Annotated[float, Field(gt=0)]
    evaluator: "EvaluatorSpec"


type EvaluatorSpec = Annotated[
    Exact
    | Contains
    | NotContains
    | Regex
    | RegexExtract
    | Length
    | JsonSchema
    | LlmJudge
    | All
    | AnyOf
    | Weighted,
    Field(discriminator="type"),
]

for _m in (All, AnyOf, Weighted, WeightedPart):
    _m.model_rebuild()

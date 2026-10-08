from typing import Annotated, Literal

from pydantic import Field

from llmbench.config.base import Strict
from llmbench.config.llama.kv import CacheType

SpecType = Literal[
    "none",
    "draft-simple",
    "draft-eagle3",
    "draft-mtp",
    "draft-dflash",
    "draft-dspark",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
]


class SpeculativeConfig(Strict):
    """Speculative decoding (`--spec-*`). llama-server only.

    `draft_model` is a model reference (`hf://...` or a path), resolved like any model.
    """

    types: list[SpecType] | None = None
    draft_model: str | None = None
    draft_n_max: Annotated[int, Field(ge=0)] | None = None
    draft_n_min: Annotated[int, Field(ge=0)] | None = None
    draft_p_min: Annotated[float, Field(ge=0, le=1)] | None = None
    draft_p_split: Annotated[float, Field(ge=0, le=1)] | None = None
    draft_sampling: Literal["greedy", "probabilistic"] | None = None
    draft_n_gpu_layers: Annotated[int, Field(ge=0)] | None = None
    draft_threads: Annotated[int, Field(gt=0)] | None = None
    draft_threads_batch: Annotated[int, Field(gt=0)] | None = None
    draft_cache_type_k: CacheType | None = None
    draft_cache_type_v: CacheType | None = None
    draft_devices: list[str] | None = None
    ngram_mod_n_min: Annotated[int, Field(ge=0)] | None = None
    ngram_mod_n_max: Annotated[int, Field(ge=0)] | None = None
    ngram_mod_n_match: Annotated[int, Field(gt=0)] | None = None
    ngram_simple_size_n: Annotated[int, Field(gt=0)] | None = None
    ngram_simple_size_m: Annotated[int, Field(gt=0)] | None = None
    ngram_simple_min_hits: Annotated[int, Field(gt=0)] | None = None

from typing import Annotated, Literal

from pydantic import Field

from llmbench.config.base import Strict


class RopeConfig(Strict):
    """RoPE / YaRN context extension (`--rope-*`, `--yarn-*`). llama-server only."""

    scaling: Literal["none", "linear", "yarn"] | None = None
    scale: Annotated[float, Field(gt=0)] | None = None
    freq_base: Annotated[float, Field(gt=0)] | None = None
    freq_scale: Annotated[float, Field(gt=0)] | None = None
    yarn_orig_ctx: Annotated[int, Field(ge=0)] | None = None
    yarn_ext_factor: float | None = None
    yarn_attn_factor: float | None = None
    yarn_beta_slow: float | None = None
    yarn_beta_fast: float | None = None

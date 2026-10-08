from typing import Annotated, Any, Literal

from pydantic import Field

from llmbench.config.base import Strict


class LoraAdapter(Strict):
    path: str
    scale: Annotated[float, Field(ge=0)] = 1.0


class ServerConfig(Strict):
    """llama-server process options. Host is always 127.0.0.1 and the port is chosen by llb."""

    parallel: Annotated[int, Field(gt=0)] | None = 1
    """Slots. Default 1 (not llama.cpp's auto=4) so single-stream numbers are not diluted."""
    cont_batching: bool | None = None
    cache_reuse: Annotated[int, Field(ge=0)] | None = None
    cache_ram_mib: Annotated[int, Field(ge=-1)] | None = None
    ctx_checkpoints: Annotated[int, Field(ge=0)] | None = None
    context_shift: bool | None = None
    keep: Annotated[int, Field(ge=-1)] | None = None
    warmup: bool | None = None
    special: bool | None = None

    jinja: bool | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning: Literal["on", "off", "auto"] | None = None
    reasoning_format: str | None = None
    reasoning_budget: Annotated[int, Field(ge=-1)] | None = None

    lora: list[LoraAdapter] = []
    threads_http: Annotated[int, Field(gt=0)] | None = None
    timeout_s: Annotated[int, Field(gt=0)] | None = None

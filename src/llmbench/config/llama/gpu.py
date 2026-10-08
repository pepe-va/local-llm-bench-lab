from typing import Annotated, Literal

from pydantic import Field

from llmbench.config.base import Strict


class GpuConfig(Strict):
    """Offloading and multi-device split (`-ngl`, `-sm`, `-ts`, `-ot`, `-ncmoe`, ...).

    Mostly irrelevant on a CPU-only build, but `override_tensor` and `n_cpu_moe`
    also matter for CPU repacking and MoE placement.
    """

    n_gpu_layers: Annotated[int, Field(ge=0)] | Literal["auto", "all"] | None = None
    split_mode: Literal["none", "layer", "row", "tensor"] | None = None
    main_gpu: Annotated[int, Field(ge=0)] | None = None
    tensor_split: list[Annotated[float, Field(ge=0)]] | None = None
    override_tensor: list[Annotated[str, Field(pattern=r"^.+=.+$")]] = []
    cpu_moe: bool | None = None
    n_cpu_moe: Annotated[int, Field(ge=0)] | None = None
    n_cpu_ffn: Annotated[int, Field(ge=0)] | None = None
    devices: list[str] | None = None
    op_offload: bool | None = None

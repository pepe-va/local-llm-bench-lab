from typing import Annotated, Literal

from pydantic import Field

from llmbench.config.base import Strict

PositiveInt = Annotated[int, Field(gt=0)]
HexMask = Annotated[str, Field(pattern=r"^(0x)?[0-9a-fA-F]+$")]
CpuRange = Annotated[str, Field(pattern=r"^\d+-\d+$")]


class LoadConfig(Strict):
    """Model loading, batching and CPU scheduling (`-t`, `-b`, `-C`, `--load-mode`, ...)."""

    threads: PositiveInt | None = None
    threads_batch: PositiveInt | None = None
    ctx_size: Annotated[int, Field(ge=0)] | None = None
    batch_size: PositiveInt | None = None
    ubatch_size: PositiveInt | None = None

    # CPU placement for generation...
    cpu_mask: HexMask | None = None
    cpu_range: CpuRange | None = None
    cpu_strict: bool | None = None
    poll: Annotated[int, Field(ge=0, le=100)] | None = None
    prio: Annotated[int, Field(ge=-1, le=3)] | None = None
    # ...and for batch / prompt processing (llama-server only)
    cpu_mask_batch: HexMask | None = None
    cpu_range_batch: CpuRange | None = None
    cpu_strict_batch: bool | None = None
    poll_batch: bool | None = None
    prio_batch: Annotated[int, Field(ge=0, le=3)] | None = None
    numa: Literal["distribute", "isolate", "numactl"] | None = None

    load_mode: Literal["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"] | None = None
    lazy_mode: Literal["on", "auto", "off"] | None = None
    repack: bool | None = None
    no_host: bool | None = None
    flash_attn: Literal["on", "off", "auto"] | None = None

    # llama-bench `--fit-target` / `--fit-ctx`
    fit_target_mib: Annotated[int, Field(ge=0)] | None = None
    fit_min_ctx: PositiveInt | None = None

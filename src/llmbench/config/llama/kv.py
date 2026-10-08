from typing import Literal

from llmbench.config.base import Strict

CacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]


class KvConfig(Strict):
    """KV cache layout (`-ctk`, `-ctv`, `-nkvo`, `--swa-full`, `--kv-unified`)."""

    cache_type_k: CacheType | None = None
    cache_type_v: CacheType | None = None
    offload: bool | None = None
    swa_full: bool | None = None
    unified: bool | None = None

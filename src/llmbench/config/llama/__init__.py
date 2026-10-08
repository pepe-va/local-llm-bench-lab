"""One pydantic model per family of llama.cpp flags.

Fields are `None` by default, meaning "do not pass the flag, use llama.cpp's own default".
Only `llamacpp/argv.py` knows how a field maps to a command-line flag.
"""

from llmbench.config.base import Strict
from llmbench.config.llama.common import LoadConfig
from llmbench.config.llama.gpu import GpuConfig
from llmbench.config.llama.kv import KvConfig


class ExtraArgs(Strict):
    """Raw flags appended per tool. Validated against the build's `--help`."""

    bench: list[str] = []
    server: list[str] = []


class LlamaConfig(Strict):
    """The full set of llama.cpp knobs for one run. `matrix` paths are relative to this."""

    load: LoadConfig = LoadConfig()
    kv: KvConfig = KvConfig()
    gpu: GpuConfig = GpuConfig()
    extra_args: ExtraArgs = ExtraArgs()
    env: dict[str, str] = {}


__all__ = ["ExtraArgs", "GpuConfig", "KvConfig", "LlamaConfig", "LoadConfig"]

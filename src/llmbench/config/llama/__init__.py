"""One pydantic model per family of llama.cpp flags.

Fields are `None` by default, meaning "do not pass the flag, use llama.cpp's own default".
Only `llamacpp/argv.py` knows how a field maps to a command-line flag.
"""

from llmbench.config.base import Strict
from llmbench.config.llama.common import LoadConfig
from llmbench.config.llama.gpu import GpuConfig
from llmbench.config.llama.kv import KvConfig
from llmbench.config.llama.rope import RopeConfig
from llmbench.config.llama.sampling import SamplingConfig
from llmbench.config.llama.server import LoraAdapter, ServerConfig
from llmbench.config.llama.speculative import SpeculativeConfig


class ExtraArgs(Strict):
    """Raw flags appended per tool. Validated against the build's `--help`."""

    bench: list[str] = []
    server: list[str] = []


class LlamaConfig(Strict):
    """The full set of llama.cpp knobs for one run. `matrix` paths are relative to this."""

    load: LoadConfig = LoadConfig()
    kv: KvConfig = KvConfig()
    gpu: GpuConfig = GpuConfig()
    rope: RopeConfig = RopeConfig()
    speculative: SpeculativeConfig = SpeculativeConfig()
    server: ServerConfig = ServerConfig()
    sampling: SamplingConfig = SamplingConfig()
    extra_args: ExtraArgs = ExtraArgs()
    env: dict[str, str] = {}

    def bench_identity(self) -> dict[str, object]:
        """The part of the config that can change a llama-bench measurement."""
        d = self.model_dump(mode="json", include={"load", "kv", "gpu", "env"})
        d["extra_args"] = self.extra_args.bench
        return d

    def server_identity(self) -> dict[str, object]:
        """The part that determines the llama-server process (everything except sampling)."""
        d = self.model_dump(mode="json", exclude={"sampling", "extra_args"})
        d["extra_args"] = self.extra_args.server
        return d


__all__ = [
    "ExtraArgs",
    "GpuConfig",
    "KvConfig",
    "LlamaConfig",
    "LoadConfig",
    "LoraAdapter",
    "RopeConfig",
    "SamplingConfig",
    "ServerConfig",
    "SpeculativeConfig",
]

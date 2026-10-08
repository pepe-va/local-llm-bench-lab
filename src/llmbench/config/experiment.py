"""Root schema of an experiment YAML."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from llmbench.config.base import Strict
from llmbench.config.llama import LlamaConfig
from llmbench.config.matrix import Matrix

Slug = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=80)]


class ExperimentMeta(Strict):
    name: Slug
    description: str = ""
    tags: list[str] = []


class BuildSpec(Strict):
    """A llama.cpp build: either managed (`ref` + `backend`) or pre-existing (`bin_dir`)."""

    id: Slug
    ref: str | None = None
    backend: Literal["cpu", "cuda", "vulkan", "metal", "blas"] = "cpu"
    cmake: dict[str, str | int | bool] = {}
    bin_dir: Path | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.ref is None) == (self.bin_dir is None):
            raise ValueError("set exactly one of `ref` (managed build) or `bin_dir` (existing)")
        if self.bin_dir is not None and self.cmake:
            raise ValueError("`cmake` only applies to managed builds (`ref`)")
        return self


class LlamaCppSection(Strict):
    builds: Annotated[list[BuildSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        _require_unique([b.id for b in self.builds], "llama_cpp.builds")
        return self


class ModelSpec(Strict):
    id: Slug
    source: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    enabled: bool = True
    retention: Literal["keep", "delete_after_run"] = "keep"

    @property
    def scheme(self) -> str:
        return self.source.split("://", 1)[0] if "://" in self.source else "file"


class BenchPhase(Strict):
    """`llama-bench` phase. List values are crossed natively by llama-bench in one call."""

    enabled: bool = True
    n_prompt: list[Annotated[int, Field(ge=0)]] = [512]
    n_gen: list[Annotated[int, Field(ge=0)]] = [128]
    n_depth: list[Annotated[int, Field(ge=0)]] = [0]
    pg: list[tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]] = []
    repetitions: Annotated[int, Field(ge=1)] = 5
    warmup: bool = True
    delay_s: Annotated[int, Field(ge=0)] | None = None
    timeout_s: Annotated[float, Field(gt=0)] = 3600
    unstable_cv: Annotated[float, Field(gt=0)] = 0.05


class LatencyPhase(Strict):
    """llama-server phase: send each suite's requests one at a time and time them.

    Measures what a user sees (TTFT, inter-token latency, end-to-end) plus the server's own
    prompt/generation throughput. Prompt caching is disabled per request.
    """

    enabled: bool = False
    suites: list[Path] = []
    """Suite files (resolved relative to the YAML that declares them)."""
    passes: Annotated[int, Field(ge=1)] = 1
    warmup_requests: Annotated[int, Field(ge=0)] = 1
    request_timeout_s: Annotated[float, Field(gt=0)] = 600
    startup_timeout_s: Annotated[float, Field(gt=0)] = 600

    @model_validator(mode="after")
    def _suites(self) -> Self:
        if self.enabled and not self.suites:
            raise ValueError("latency phase is enabled but lists no suites")
        return self


class Phases(Strict):
    bench: BenchPhase = BenchPhase()
    latency: LatencyPhase = LatencyPhase()


class Telemetry(Strict):
    hz: Annotated[float, Field(gt=0, le=50)] = 2
    sources: list[Literal["linux", "rpi"]] = ["linux", "rpi"]


class Cooldown(Strict):
    enabled: bool = True
    max_temp_c: Annotated[float, Field(gt=0)] = 60
    stable_for_s: Annotated[float, Field(ge=0)] = 10
    poll_s: Annotated[float, Field(gt=0)] = 5
    timeout_s: Annotated[float, Field(gt=0)] = 900


class Preflight(Strict):
    require_governor: str | None = None
    fail_on_throttled: bool = False


class Hardware(Strict):
    telemetry: Telemetry = Telemetry()
    cooldown: Cooldown = Cooldown()
    preflight: Preflight = Preflight()


class Output(Strict):
    """Results always go to SQLite (needed for --resume); `formats` adds exports."""

    directory: Path | None = None
    formats: list[Literal["jsonl", "csv"]] = ["jsonl", "csv"]


class Experiment(Strict):
    version: Literal[1] = 1
    experiment: ExperimentMeta
    llama_cpp: LlamaCppSection
    models: Annotated[list[ModelSpec], Field(min_length=1)]
    defaults: LlamaConfig = LlamaConfig()
    matrix: Matrix = Matrix()
    phases: Phases = Phases()
    hardware: Hardware = Hardware()
    output: Output = Output()

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_unique([m.id for m in self.models], "models")
        self.matrix.validate_against(self.defaults)
        return self

    @property
    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]


def _require_unique(ids: list[Any], where: str) -> None:
    seen: set[Any] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate id {i!r} in {where}")
        seen.add(i)

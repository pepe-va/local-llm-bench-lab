"""What every phase runner shares: progress events, the run context and small helpers."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llmbench.config.experiment import Experiment
from llmbench.engine.planner import RunSpec
from llmbench.hub.store import ResolvedModel
from llmbench.paths import runs_dir
from llmbench.results import export
from llmbench.results.models import RunRecord
from llmbench.results.store import Store
from llmbench.telemetry import sources, throttle
from llmbench.telemetry.cooldown import CooldownResult, wait_for_cooldown
from llmbench.telemetry.sampler import TelemetrySummary


class Events(Protocol):
    """Progress callbacks, implemented by the CLI (rich) or tests."""

    def log(self, message: str) -> None: ...
    def run_start(self, index: int, total: int, spec: RunSpec) -> None: ...
    def cooling(self, temp_c: float, elapsed_s: float) -> None: ...
    def run_end(self, index: int, total: int, record: RunRecord) -> None: ...
    def run_skipped(self, index: int, total: int, spec: RunSpec, reason: str) -> None: ...


@dataclass(frozen=True)
class ModelHandle:
    """A resolved model plus its identity and what gets stored about it."""

    path: Path
    sha256: str
    info: dict[str, Any]
    resolved: ResolvedModel


@dataclass(frozen=True)
class RunContext:
    exp: Experiment
    store: Store
    events: Events
    machine: dict[str, Any]
    out_dir: Path
    resume: bool

    @property
    def use_rpi(self) -> bool:
        return "rpi" in self.exp.hardware.telemetry.sources

    def cooldown(self) -> CooldownResult:
        return wait_for_cooldown(
            self.exp.hardware.cooldown,
            lambda: sources.temperature_c(use_rpi=self.use_rpi),
            on_wait=self.events.cooling,
        )

    def save(self, record: RunRecord, telemetry: TelemetrySummary | None) -> None:
        self.store.save_run(record, telemetry.samples if telemetry else None)
        if "jsonl" in self.exp.output.formats:
            export.append_jsonl(self.out_dir / "runs.jsonl", record)


def output_dir(exp: Experiment) -> Path:
    if exp.output.directory is not None:
        return exp.output.directory.expanduser().resolve()
    return runs_dir() / exp.experiment.name


def throttled_during(t: TelemetrySummary | None) -> bool:
    if t is None or t.throttled_after is None:
        return False
    after = throttle.decode(t.throttled_after)
    return after.is_throttled_now or t.throttled_after != t.throttled_before


def now() -> datetime:
    return datetime.now(UTC)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds")

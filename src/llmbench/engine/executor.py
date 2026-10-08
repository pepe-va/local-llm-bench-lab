"""The run loop: resolve -> (skip if done) -> cooldown -> llama-bench + telemetry -> persist.

A failure in one run is recorded as a result and never stops the suite.
Ctrl-C records the current run as `interrupted` and stops cleanly; `--resume` continues.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llmbench.config.experiment import Experiment, ModelSpec
from llmbench.engine import env as envmod
from llmbench.engine.planner import Plan, RunSpec, fingerprint
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.store import (
    ModelNotFoundError,
    ModelStore,
    ResolvedModel,
    combined_sha256,
    model_info,
)
from llmbench.llamacpp import builds
from llmbench.llamacpp.argv import bench_argv
from llmbench.llamacpp.bench import BenchError, run_bench
from llmbench.llamacpp.introspect import IntrospectionError, binary_flags, unknown_flags
from llmbench.paths import runs_dir
from llmbench.results import export
from llmbench.results.models import Measurement, RunRecord, RunStatus
from llmbench.results.stats import annotate
from llmbench.results.store import Store
from llmbench.telemetry import sources, throttle
from llmbench.telemetry.cooldown import CooldownResult, wait_for_cooldown
from llmbench.telemetry.sampler import Sampler, TelemetrySummary


class Events(Protocol):
    """Progress callbacks, implemented by the CLI (rich) or tests."""

    def log(self, message: str) -> None: ...
    def run_start(self, index: int, total: int, spec: RunSpec) -> None: ...
    def cooling(self, temp_c: float, elapsed_s: float) -> None: ...
    def run_end(self, index: int, total: int, record: RunRecord) -> None: ...
    def run_skipped(self, index: int, total: int, spec: RunSpec, reason: str) -> None: ...


@dataclass
class ExecutionSummary:
    output_dir: Path
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    interrupted: bool = False
    warnings: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class _Model:
    path: Path
    sha256: str
    info: dict[str, Any]
    resolved: ResolvedModel


class _Models:
    """Resolve each model right before its first run and release it after its last.

    On a Pi with little disk this keeps at most one downloaded model around when
    `retention: delete_after_run` is set.
    """

    def __init__(self, plan: Plan, store: Store, models: ModelStore, events: Events) -> None:
        self._store = store
        self._models = models
        self._events = events
        self._cache: dict[str, _Model | None] = {}
        self._last: dict[str, int] = {r.model.id: i for i, r in enumerate(plan.runs)}

    def get(self, spec: ModelSpec) -> _Model | None:
        if spec.id not in self._cache:
            self._cache[spec.id] = self._resolve(spec)
        return self._cache[spec.id]

    def done_with(self, index: int, spec: ModelSpec) -> None:
        model = self._cache.get(spec.id)
        if model is None or self._last[spec.id] != index or spec.retention != "delete_after_run":
            return
        freed = self._models.release(model.resolved)
        if freed:
            self._events.log(f"model {spec.id}: removed from the HF cache ({freed / 1e9:.2f} GB)")

    def _resolve(self, spec: ModelSpec) -> _Model | None:
        try:
            resolved = self._models.resolve(spec)
            meta = read_gguf(resolved.shards).summary()
        except (ModelNotFoundError, GGUFError) as e:
            self._events.log(f"[error] {e}")
            return None
        if resolved.sha256 is not None:
            digest = resolved.sha256
            if len(resolved.shards) == 1:
                self._store.remember_sha256(resolved.path, digest)
        else:
            self._events.log(f"model {spec.id}: hashing {resolved.path.name}...")
            digest = combined_sha256([self._store.model_sha256(p) for p in resolved.shards])
        if spec.sha256 is not None and spec.sha256 != digest:
            self._events.log(f"[error] model {spec.id!r}: sha256 mismatch (expected {spec.sha256})")
            return None
        where = (
            f"{resolved.repo}@{resolved.commit[:12]}" if resolved.repo and resolved.commit else ""
        )
        self._events.log(
            f"model {spec.id}: {resolved.path.name} {where} ({resolved.origin}, "
            f"{meta['quant']}, {meta['n_params'] / 1e9:.2f}B params)"
        )
        info = model_info(resolved, digest) | {"gguf": meta}
        return _Model(resolved.path, digest, info, resolved)


def output_dir(exp: Experiment) -> Path:
    if exp.output.directory is not None:
        return exp.output.directory.expanduser().resolve()
    return runs_dir() / exp.experiment.name


def execute(
    exp: Experiment,
    plan: Plan,
    events: Events,
    *,
    resume: bool = False,
    auto_build: bool = True,
    model_store: ModelStore | None = None,
) -> ExecutionSummary:
    out = output_dir(exp)
    summary = ExecutionSummary(out)
    use_rpi = "rpi" in exp.hardware.telemetry.sources
    summary.warnings += envmod.preflight(exp.hardware.preflight, use_rpi=use_rpi)
    for w in summary.warnings:
        events.log(f"[warning] {w}")

    store = Store(out / "benchmark.sqlite")
    try:
        machine = envmod.snapshot()
        resolved = _resolve_builds(exp, events, auto_build=auto_build)
        models = _Models(plan, store, model_store or ModelStore(log=events.log), events)
        total = len(plan.runs)
        for i, spec in enumerate(plan.runs, 1):
            rb = resolved.get(spec.build.id)
            model = models.get(spec.model) if rb is not None else None
            if rb is None or model is None:
                summary.failed += 1
                events.run_skipped(i, total, spec, "build or model could not be resolved")
                continue
            status = _run_one(
                exp=exp,
                spec=spec,
                rb=rb,
                model=model,
                machine=machine,
                store=store,
                events=events,
                index=i,
                total=total,
                resume=resume,
            )
            models.done_with(i - 1, spec.model)
            if status == "skipped":
                summary.skipped += 1
            elif status == "completed":
                summary.completed += 1
            elif status == "failed":
                summary.failed += 1
            else:
                summary.interrupted = True
                break
    finally:
        if "csv" in exp.output.formats:
            export.write_csv(out / "summary.csv", store, exp.experiment.name)
        store.close()
    return summary


def _resolve_builds(
    exp: Experiment, events: Events, *, auto_build: bool
) -> dict[str, builds.ResolvedBuild]:
    out: dict[str, builds.ResolvedBuild] = {}
    for spec in exp.llama_cpp.builds:
        try:
            try:
                rb = builds.resolve(spec)
            except builds.BuildError:
                if not (auto_build and spec.ref):
                    raise
                events.log(f"building llama.cpp {spec.ref} ({spec.backend}) for {spec.id!r}...")
                rb = builds.build(spec, log=events.log)
            binary_flags(rb.tool("llama-bench"))  # fail early if the binary is unusable
            out[spec.id] = rb
            events.log(f"build {spec.id}: {rb.version} @ {rb.commit[:12]} ({rb.backend})")
        except (builds.BuildError, IntrospectionError) as e:
            events.log(f"[error] {e}")
    return out


def _run_one(
    *,
    exp: Experiment,
    spec: RunSpec,
    rb: builds.ResolvedBuild,
    model: _Model,
    machine: dict[str, Any],
    store: Store,
    events: Events,
    index: int,
    total: int,
    resume: bool,
) -> RunStatus | str:
    bench = exp.phases.bench
    argv = bench_argv(rb.tool("llama-bench"), model.path, spec.config, bench)
    fp = fingerprint(
        build=rb.identity(), model_sha256=model.sha256, config=spec.config, bench=bench
    )
    if resume and store.is_completed(fp):
        events.run_skipped(index, total, spec, "already completed")
        return "skipped"

    events.run_start(index, total, spec)
    started = _now()
    status: RunStatus = "completed"
    error: str | None = None
    measurements: list[Measurement] = []
    usage: dict[str, float | None] = {}
    cooldown: CooldownResult | None = None
    telemetry: TelemetrySummary | None = None
    use_rpi = "rpi" in exp.hardware.telemetry.sources

    unknown = unknown_flags(argv.args, binary_flags(rb.tool("llama-bench")))
    if unknown:
        status = "failed"
        error = f"flags not supported by build {rb.id} ({rb.commit[:12]}): {', '.join(unknown)}"
    else:
        cooldown = wait_for_cooldown(
            exp.hardware.cooldown,
            lambda: sources.temperature_c(use_rpi=use_rpi),
            on_wait=events.cooling,
        )
        sampler = Sampler(exp.hardware.telemetry)
        sampler.start()
        try:
            result = run_bench(
                argv.args,
                env=spec.config.env,
                timeout_s=bench.timeout_s,
                on_start=sampler.attach,
            )
            measurements = result.measurements
            usage = {
                "max_rss_mb": result.max_rss_mb,
                "cpu_user_s": result.cpu_user_s,
                "cpu_sys_s": result.cpu_sys_s,
            }
        except BenchError as e:
            status, error = "failed", str(e)
        except KeyboardInterrupt:
            status, error = "interrupted", "interrupted by user"
        finally:
            telemetry = sampler.stop()

    throttled = _throttled_during(telemetry)
    measurements = [
        annotate(m, unstable_cv=bench.unstable_cv, throttled=throttled) for m in measurements
    ]
    finished = _now()
    record = RunRecord(
        fingerprint=fp,
        experiment=exp.experiment.name,
        phase="bench",
        status=status,
        build={"id": rb.id, "ref": rb.ref, "version": rb.version, **rb.identity()},
        model=model.info,
        variant=spec.variant,
        config=spec.config.model_dump(mode="json"),
        argv=argv.args,
        ignored=argv.ignored,
        env=machine,
        started_at=started.isoformat(timespec="milliseconds"),
        finished_at=finished.isoformat(timespec="milliseconds"),
        duration_s=(finished - started).total_seconds(),
        error=error,
        telemetry=(telemetry.to_dict() | usage) if telemetry else None,
        cooldown=asdict(cooldown) if cooldown else None,
        measurements=measurements,
    )
    store.save_run(record, telemetry.samples if telemetry else None)
    if "jsonl" in exp.output.formats:
        export.append_jsonl(output_dir(exp) / "runs.jsonl", record)
    events.run_end(index, total, record)
    return status


def _throttled_during(t: TelemetrySummary | None) -> bool:
    if t is None or t.throttled_after is None:
        return False
    after = throttle.decode(t.throttled_after)
    return after.is_throttled_now or t.throttled_after != t.throttled_before


def _now() -> datetime:
    return datetime.now(UTC)

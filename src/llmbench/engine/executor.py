"""The run loop: resolve -> (skip if done) -> cooldown -> run -> persist, for every phase.

A failure in one run is recorded as a result and never stops the suite.
Ctrl-C records the current run as `interrupted` and stops cleanly; `--resume` continues.
"""

from dataclasses import dataclass, field
from pathlib import Path

from llmbench.config.experiment import Experiment, ModelSpec
from llmbench.engine import env as envmod
from llmbench.engine.phases.bench import run_bench_spec
from llmbench.engine.phases.suites import ServerPool, run_suite_spec
from llmbench.engine.planner import Plan, RunSpec
from llmbench.engine.runtime import Events, ModelHandle, RunContext, output_dir
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.store import ModelNotFoundError, ModelStore, combined_sha256, model_info
from llmbench.llamacpp import builds
from llmbench.llamacpp.introspect import IntrospectionError, binary_flags
from llmbench.results import export
from llmbench.results.store import Store

__all__ = ["Events", "ExecutionSummary", "execute", "output_dir"]


@dataclass
class ExecutionSummary:
    output_dir: Path
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    interrupted: bool = False
    warnings: list[str] = field(default_factory=list[str])


class _Models:
    """Resolve each model right before its first run and release it after its last.

    On a Pi with little disk this keeps at most one downloaded model around when
    `retention: delete_after_run` is set.
    """

    def __init__(self, plan: Plan, store: Store, models: ModelStore, events: Events) -> None:
        self._store = store
        self._models = models
        self._events = events
        self._cache: dict[str, ModelHandle | None] = {}
        self._last: dict[str, int] = {r.model.id: i for i, r in enumerate(plan.runs)}

    def get(self, spec: ModelSpec) -> ModelHandle | None:
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

    def draft(self, source: str) -> ModelHandle | None:
        """Draft models for speculative decoding are resolved (and cached) like any model."""
        key = f"draft:{source}"
        if key not in self._cache:
            self._cache[key] = self._resolve(ModelSpec(id="draft", source=source))
        return self._cache[key]

    def _resolve(self, spec: ModelSpec) -> ModelHandle | None:
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
        return ModelHandle(resolved.path, digest, info, resolved)


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
    pool = ServerPool(events, out / "logs")
    try:
        ctx = RunContext(exp, store, events, envmod.snapshot(), out, resume)
        resolved = _resolve_builds(exp, plan, events, auto_build=auto_build)
        models = _Models(plan, store, model_store or ModelStore(log=events.log), events)
        total = len(plan.runs)
        for i, spec in enumerate(plan.runs, 1):
            status = _dispatch(
                ctx, spec, resolved=resolved, models=models, pool=pool, index=i, total=total
            )
            nxt = plan.runs[i] if i < total else None
            if nxt is None or nxt.phase == "bench" or nxt.server_key() != spec.server_key():
                pool.release()  # before retention may delete the model it serves
            models.done_with(i - 1, spec.model)
            if status is None:
                summary.skipped += 1
            elif status == "completed":
                summary.completed += 1
            elif status == "failed":
                summary.failed += 1
            else:
                summary.interrupted = True
                break
    finally:
        pool.release()
        if "csv" in exp.output.formats:
            export.write_csv(out / "summary.csv", store, exp.experiment.name)
            export.write_latency_csv(out / "latency.csv", store, exp.experiment.name)
            export.write_quality_csv(out / "quality.csv", store, exp.experiment.name)
        store.close()
    return summary


def _dispatch(
    ctx: RunContext,
    spec: RunSpec,
    *,
    resolved: dict[str, builds.ResolvedBuild],
    models: _Models,
    pool: ServerPool,
    index: int,
    total: int,
) -> str | None:
    rb = resolved.get(spec.build.id)
    model = models.get(spec.model) if rb is not None else None
    if rb is None or model is None:
        ctx.events.run_skipped(index, total, spec, "build or model could not be resolved")
        return "failed"
    if spec.phase == "bench":
        return run_bench_spec(ctx, spec, rb=rb, model=model, index=index, total=total)
    draft = None
    if spec.config.speculative.draft_model is not None:
        draft = models.draft(spec.config.speculative.draft_model)
        if draft is None:
            ctx.events.run_skipped(index, total, spec, "draft model could not be resolved")
            return "failed"
    return run_suite_spec(
        ctx, spec, rb=rb, model=model, draft=draft, pool=pool, index=index, total=total
    )


def _resolve_builds(
    exp: Experiment, plan: Plan, events: Events, *, auto_build: bool
) -> dict[str, builds.ResolvedBuild]:
    tools = {"llama-bench" if r.phase == "bench" else "llama-server" for r in plan.runs}
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
            for tool in sorted(tools):
                binary_flags(rb.tool(tool))  # fail early if a binary is missing or unusable
            out[spec.id] = rb
            events.log(f"build {spec.id}: {rb.version} @ {rb.commit[:12]} ({rb.backend})")
        except (builds.BuildError, IntrospectionError) as e:
            events.log(f"[error] {e}")
    return out

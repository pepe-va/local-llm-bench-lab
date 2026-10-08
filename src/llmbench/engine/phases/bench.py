"""Bench phase: one llama-bench call per run."""

from dataclasses import asdict

from llmbench.engine.planner import RunSpec, fingerprint
from llmbench.engine.runtime import ModelHandle, RunContext, iso, now, throttled_during
from llmbench.llamacpp.argv import bench_argv
from llmbench.llamacpp.bench import BenchError, run_bench
from llmbench.llamacpp.builds import ResolvedBuild
from llmbench.llamacpp.introspect import binary_flags, unknown_flags
from llmbench.results.models import Measurement, RunRecord, RunStatus
from llmbench.results.stats import annotate
from llmbench.telemetry.cooldown import CooldownResult
from llmbench.telemetry.sampler import Sampler, TelemetrySummary


def run_bench_spec(
    ctx: RunContext, spec: RunSpec, *, rb: ResolvedBuild, model: ModelHandle, index: int, total: int
) -> RunStatus | None:
    """Returns the run status, or None if skipped by --resume."""
    bench = ctx.exp.phases.bench
    argv = bench_argv(rb.tool("llama-bench"), model.path, spec.config, bench)
    fp = fingerprint(
        build=rb.identity(), model_sha256=model.sha256, config=spec.config, bench=bench
    )
    if ctx.resume and ctx.store.is_completed(fp):
        ctx.events.run_skipped(index, total, spec, "already completed")
        return None

    ctx.events.run_start(index, total, spec)
    started = now()
    status: RunStatus = "completed"
    error: str | None = None
    measurements: list[Measurement] = []
    usage: dict[str, float | None] = {}
    cooldown: CooldownResult | None = None
    telemetry: TelemetrySummary | None = None

    unknown = unknown_flags(argv.args, binary_flags(rb.tool("llama-bench")))
    if unknown:
        status = "failed"
        error = f"flags not supported by build {rb.id} ({rb.commit[:12]}): {', '.join(unknown)}"
    else:
        cooldown = ctx.cooldown()
        sampler = Sampler(ctx.exp.hardware.telemetry)
        sampler.start()
        try:
            result = run_bench(
                argv.args, env=spec.config.env, timeout_s=bench.timeout_s, on_start=sampler.attach
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

    throttled = throttled_during(telemetry)
    measurements = [
        annotate(m, unstable_cv=bench.unstable_cv, throttled=throttled) for m in measurements
    ]
    finished = now()
    record = RunRecord(
        fingerprint=fp,
        experiment=ctx.exp.experiment.name,
        phase="bench",
        status=status,
        build={"id": rb.id, "ref": rb.ref, "version": rb.version, **rb.identity()},
        model=model.info,
        variant=spec.variant,
        config=spec.config.model_dump(mode="json"),
        argv=argv.args,
        ignored=argv.ignored,
        env=ctx.machine,
        started_at=iso(started),
        finished_at=iso(finished),
        duration_s=(finished - started).total_seconds(),
        error=error,
        telemetry=(telemetry.to_dict() | usage) if telemetry else None,
        cooldown=asdict(cooldown) if cooldown else None,
        measurements=measurements,
    )
    ctx.save(record, telemetry)
    ctx.events.run_end(index, total, record)
    return status

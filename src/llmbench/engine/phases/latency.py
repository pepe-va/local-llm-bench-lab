"""Latency phase: suites of requests against a llama-server, one request at a time.

A `ServerPool` keeps at most one llama-server alive and reuses it while consecutive runs share
a server key (the planner orders runs that way).
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from llmbench.config.suite import Suite
from llmbench.data import synthetic
from llmbench.data.requests import Request
from llmbench.engine.planner import RunSpec, latency_fingerprint
from llmbench.engine.runtime import Events, ModelHandle, RunContext, iso, now
from llmbench.llamacpp.argv import Argv, server_argv
from llmbench.llamacpp.builds import ResolvedBuild
from llmbench.llamacpp.client import LlamaClient
from llmbench.llamacpp.introspect import binary_flags, unknown_flags
from llmbench.llamacpp.server import LlamaServer, ServerError
from llmbench.results.models import RequestResult, RunRecord, RunStatus
from llmbench.results.stats import latency_metrics
from llmbench.telemetry.cooldown import CooldownResult
from llmbench.telemetry.sampler import Sampler, TelemetrySummary


def effective_sampling(spec: RunSpec) -> dict[str, Any]:
    """Experiment sampling, overridden by the suite; synthetic suites ignore EOS by default."""
    assert spec.suite is not None
    body = spec.config.sampling.body() | spec.suite.sampling.body()
    body.setdefault("ignore_eos", True)  # fixed output lengths make runs comparable
    return body


@dataclass
class ServerPool:
    events: Events
    log_dir: Path
    key: str | None = None
    server: LlamaServer | None = None
    client: LlamaClient | None = None
    _count: int = 0
    _failed: dict[str, str] = field(default_factory=dict[str, str])
    _requests: dict[str, list[Request]] = field(default_factory=dict[str, list[Request]])

    def acquire(
        self, key: str, make_argv: Callable[[int], Argv], env: dict[str, str], timeout_s: float
    ) -> tuple[LlamaServer, LlamaClient, bool]:
        """Return the server for `key`, starting it if needed (third item: freshly started)."""
        if key in self._failed:
            raise ServerError(self._failed[key])
        if self.key == key and self.server is not None and self.client is not None:
            return self.server, self.client, False
        self.release()
        self._count += 1
        server = LlamaServer(
            make_argv, self.log_dir / f"server-{self._count:03d}.log", env, timeout_s
        )
        self.events.log(f"starting llama-server ({server.log_path.name})...")
        try:
            server.start()
        except ServerError as e:
            self._failed[key] = str(e)
            raise
        self.key, self.server, self.client = key, server, LlamaClient(server.url)
        self.events.log(
            f"llama-server ready in {server.load_s:.1f}s "
            f"({server.props.get('build_info')}, {server.props.get('total_slots')} slot(s))"
        )
        return server, self.client, True

    def requests(self, suite_path: Path, suite: Suite) -> list[Request]:
        """Materialize a suite with the current server's tokenizer (cached per server)."""
        assert self.client is not None
        assert self.key is not None
        cache_key = f"{self.key}|{suite_path}"
        if cache_key not in self._requests:
            client = self.client
            self._requests[cache_key] = synthetic.build(
                suite.source.synthetic, tokenize=client.tokenize, bos=client.bos(), suite=suite.name
            )
        return self._requests[cache_key]

    def release(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.server is not None:
            stats = self.server.stop()
            if stats.max_rss_mb is not None:
                self.events.log(f"llama-server stopped (peak RSS {stats.max_rss_mb:.0f} MB)")
        self.key = self.server = self.client = None
        self._requests.clear()


def run_latency_spec(  # noqa: PLR0915 (one linear story: resolve, run, record)
    ctx: RunContext,
    spec: RunSpec,
    *,
    rb: ResolvedBuild,
    model: ModelHandle,
    draft: ModelHandle | None,
    pool: ServerPool,
    index: int,
    total: int,
) -> RunStatus | None:
    assert spec.suite is not None
    assert spec.suite_path is not None
    phase = ctx.exp.phases.latency
    sampling = effective_sampling(spec)
    fp = latency_fingerprint(
        build=rb.identity(),
        model_sha256=model.sha256,
        draft_sha256=draft.sha256 if draft else None,
        config=spec.config,
        sampling=sampling,
        suite=spec.suite,
        latency=phase,
    )
    if ctx.resume and ctx.store.is_completed(fp):
        ctx.events.run_skipped(index, total, spec, "already completed")
        return None

    binary = rb.tool("llama-server")

    def make_argv(port: int) -> Argv:
        return server_argv(
            binary, model.path, spec.config, port=port, draft_model=draft.path if draft else None
        )

    argv = make_argv(0)
    ctx.events.run_start(index, total, spec)
    started = now()
    status: RunStatus = "completed"
    error: str | None = None
    results: list[RequestResult] = []
    metrics: dict[str, float | None] = {}
    cooldown: CooldownResult | None = None
    telemetry: TelemetrySummary | None = None

    unknown = unknown_flags(argv.args, binary_flags(binary))
    if unknown:
        status = "failed"
        error = f"flags not supported by build {rb.id} ({rb.commit[:12]}): {', '.join(unknown)}"
    else:
        cooldown = ctx.cooldown()
        sampler = Sampler(ctx.exp.hardware.telemetry)
        try:
            server, client, fresh = pool.acquire(
                spec.server_key(), make_argv, spec.config.env, phase.startup_timeout_s
            )
            argv = Argv(server.argv, argv.ignored)
            sampler.start()
            sampler.attach(server.pid)
            requests = pool.requests(spec.suite_path, spec.suite)
            for req in requests[: phase.warmup_requests]:
                client.complete(req, sampling)  # not recorded
            before = server.metrics()
            for p in range(phase.passes):
                for req in requests:
                    r = client.complete(req, sampling)
                    results.append(_tagged(r, p))
            after = server.metrics()
            metrics = latency_metrics(results) | _server_metrics(server, before, after, fresh)
            failed = [r for r in results if not r.ok]
            if failed or not results:
                status = "failed"
                error = f"{len(failed)}/{len(results)} request(s) failed: {failed[0].error}"
        except ServerError as e:
            status, error = "failed", str(e)
        except KeyboardInterrupt:
            status, error = "interrupted", "interrupted by user"
            pool.release()
        finally:
            telemetry = sampler.stop() if sampler.started else None

    finished = now()
    record = RunRecord(
        fingerprint=fp,
        experiment=ctx.exp.experiment.name,
        phase="latency",
        status=status,
        build={"id": rb.id, "ref": rb.ref, "version": rb.version, **rb.identity()},
        model=model.info | ({"draft": draft.info} if draft else {}),
        variant=spec.variant,
        config=spec.config.model_dump(mode="json") | {"effective_sampling": sampling},
        argv=argv.args,
        ignored=argv.ignored,
        env=ctx.machine,
        started_at=iso(started),
        finished_at=iso(finished),
        duration_s=(finished - started).total_seconds(),
        error=error,
        telemetry=telemetry.to_dict() if telemetry else None,
        cooldown=asdict(cooldown) if cooldown else None,
        suite=spec.suite.name,
        metrics=metrics,
        requests=results,
    )
    ctx.save(record, telemetry)
    ctx.events.run_end(index, total, record)
    return status


def _tagged(r: RequestResult, pass_index: int) -> RequestResult:
    return replace(r, id=f"{r.id}#{pass_index}")


def _server_metrics(
    server: LlamaServer, before: dict[str, float], after: dict[str, float], fresh: bool
) -> dict[str, float | None]:
    drafted = after.get("spec_decode_num_draft_tokens_total", 0) - before.get(
        "spec_decode_num_draft_tokens_total", 0
    )
    accepted = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get(
        "spec_decode_num_accepted_tokens_total", 0
    )
    return {
        "server_load_s": server.load_s if fresh else None,
        "server_peak_rss_mb": server.peak_rss_mb(),
        "spec_draft_tokens": drafted or None,
        "spec_acceptance": accepted / drafted if drafted else None,
    }

"""llama-server phases (latency, quality): suites of requests, one request at a time.

A `ServerPool` keeps at most one llama-server alive and reuses it while consecutive runs share
a server key (the planner orders runs that way, across both phases). Quality runs additionally
score every response with its evaluator; latency metrics are recorded for both.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from llmbench.config.suite import Suite
from llmbench.data import synthetic
from llmbench.data.hf_datasets import DatasetError
from llmbench.data.requests import Request, requests_digest
from llmbench.data.sources import Materialized, SuiteError, materialize
from llmbench.engine.planner import RunSpec, phase_config, suite_fingerprint
from llmbench.engine.runtime import Events, ModelHandle, RunContext, iso, now
from llmbench.evaluation.deterministic import evaluate
from llmbench.llamacpp.argv import Argv, server_argv
from llmbench.llamacpp.builds import ResolvedBuild
from llmbench.llamacpp.client import LlamaClient
from llmbench.llamacpp.introspect import binary_flags, unknown_flags
from llmbench.llamacpp.server import LlamaServer, ServerError
from llmbench.results.models import RequestResult, RunRecord, RunStatus
from llmbench.results.stats import latency_metrics, quality_metrics
from llmbench.telemetry.cooldown import CooldownResult
from llmbench.telemetry.sampler import Sampler, TelemetrySummary


def effective_sampling(spec: RunSpec) -> dict[str, Any]:
    """Experiment sampling, overridden by the suite; synthetic suites ignore EOS by default."""
    assert spec.suite is not None
    body = spec.config.sampling.body() | spec.suite.sampling.body()
    if spec.suite.is_synthetic:
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
    _synthetic: dict[str, list[Request]] = field(default_factory=dict[str, list[Request]])
    _materialized: dict[Path, Materialized] = field(default_factory=dict[Path, Materialized])

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

    def materialized(self, path: Path, suite: Suite) -> Materialized:
        """Items/files/datasets do not depend on the server: materialize once per suite."""
        if path not in self._materialized:
            if suite.source.hf_dataset is not None:
                self.events.log(f"loading dataset {suite.source.hf_dataset.repo}...")
            self._materialized[path] = materialize(suite)
        return self._materialized[path]

    def synthetic_requests(self, path: Path, suite: Suite) -> list[Request]:
        """Synthetic suites need the current server's tokenizer (cached per server)."""
        assert self.client is not None
        assert self.key is not None
        assert suite.source.synthetic is not None
        cache_key = f"{self.key}|{path}"
        if cache_key not in self._synthetic:
            client = self.client
            self._synthetic[cache_key] = synthetic.build(
                suite.source.synthetic, tokenize=client.tokenize, bos=client.bos(), suite=suite.name
            )
        return self._synthetic[cache_key]

    def release(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.server is not None:
            stats = self.server.stop()
            if stats.max_rss_mb is not None:
                self.events.log(f"llama-server stopped (peak RSS {stats.max_rss_mb:.0f} MB)")
        self.key = self.server = self.client = None
        self._synthetic.clear()


@dataclass
class _Outcome:
    status: RunStatus = "completed"
    error: str | None = None
    results: list[RequestResult] = field(default_factory=list[RequestResult])
    metrics: dict[str, float | None] = field(default_factory=dict[str, float | None])
    argv: Argv | None = None
    cooldown: CooldownResult | None = None
    telemetry: TelemetrySummary | None = None


def run_suite_spec(
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
    sampling = effective_sampling(spec)
    started = now()
    out = _Outcome()

    materialized: Materialized | None = None
    if not spec.suite.is_synthetic:
        try:
            materialized = pool.materialized(spec.suite_path, spec.suite)
        except (SuiteError, DatasetError) as e:
            out.status, out.error = "failed", str(e)
    fp = suite_fingerprint(
        phase=spec.phase,
        build=rb.identity(),
        model_sha256=model.sha256,
        draft_sha256=draft.sha256 if draft else None,
        config=spec.config,
        sampling=sampling,
        suite=spec.suite,
        phase_config=phase_config(ctx.exp, spec.phase),
        requests_digest=requests_digest(materialized.requests) if materialized else None,
    )
    if ctx.resume and ctx.store.is_completed(fp):
        ctx.events.run_skipped(index, total, spec, "already completed")
        return None

    binary = rb.tool("llama-server")

    def make_argv(port: int) -> Argv:
        return server_argv(
            binary, model.path, spec.config, port=port, draft_model=draft.path if draft else None
        )

    out.argv = make_argv(0)
    ctx.events.run_start(index, total, spec)
    if out.status == "completed":
        unknown = unknown_flags(out.argv.args, binary_flags(binary))
        if unknown:
            out.status = "failed"
            out.error = (
                f"flags not supported by build {rb.id} ({rb.commit[:12]}): {', '.join(unknown)}"
            )
        else:
            _measure(
                ctx,
                spec,
                pool=pool,
                make_argv=make_argv,
                sampling=sampling,
                materialized=materialized,
                out=out,
            )

    finished = now()
    record = RunRecord(
        fingerprint=fp,
        experiment=ctx.exp.experiment.name,
        phase=spec.phase,
        status=out.status,
        build={"id": rb.id, "ref": rb.ref, "version": rb.version, **rb.identity()},
        model=model.info | ({"draft": draft.info} if draft else {}),
        variant=spec.variant,
        config=spec.config.model_dump(mode="json") | {"effective_sampling": sampling},
        argv=out.argv.args,
        ignored=out.argv.ignored,
        env=ctx.machine,
        started_at=iso(started),
        finished_at=iso(finished),
        duration_s=(finished - started).total_seconds(),
        error=out.error,
        telemetry=out.telemetry.to_dict() if out.telemetry else None,
        cooldown=asdict(out.cooldown) if out.cooldown else None,
        suite=spec.suite.name,
        suite_provenance=materialized.provenance if materialized else {"source": "synthetic"},
        metrics=out.metrics,
        requests=out.results,
    )
    ctx.save(record, out.telemetry)
    ctx.events.run_end(index, total, record)
    return out.status


def _measure(
    ctx: RunContext,
    spec: RunSpec,
    *,
    pool: ServerPool,
    make_argv: Callable[[int], Argv],
    sampling: dict[str, Any],
    materialized: Materialized | None,
    out: _Outcome,
) -> None:
    assert spec.suite is not None
    assert spec.suite_path is not None
    phase = phase_config(ctx.exp, spec.phase)
    endpoint = spec.suite.endpoint
    out.cooldown = ctx.cooldown()
    sampler = Sampler(ctx.exp.hardware.telemetry)
    try:
        server, client, fresh = pool.acquire(
            spec.server_key(), make_argv, spec.config.env, phase.startup_timeout_s
        )
        out.argv = Argv(server.argv, out.argv.ignored if out.argv else [])
        sampler.start()
        sampler.attach(server.pid)
        requests = (
            materialized.requests
            if materialized is not None
            else pool.synthetic_requests(spec.suite_path, spec.suite)
        )
        for req in requests[: phase.warmup_requests]:
            client.complete(req, sampling, endpoint=endpoint)  # not recorded
        before = server.metrics()
        for p in range(phase.passes):
            for req in requests:
                r = client.complete(req, sampling, endpoint=endpoint)
                if spec.phase == "quality":
                    r = _scored(r, req)
                out.results.append(replace(r, id=f"{r.id}#{p}"))
        after = server.metrics()
        out.metrics = latency_metrics(out.results) | _server_metrics(server, before, after, fresh)
        if spec.phase == "quality":
            out.metrics |= quality_metrics(out.results)
        failed = [r for r in out.results if not r.ok]
        if failed or not out.results:
            out.status = "failed"
            first = failed[0].error if failed else "no requests"
            out.error = f"{len(failed)}/{len(out.results)} request(s) failed: {first}"
    except ServerError as e:
        out.status, out.error = "failed", str(e)
    except KeyboardInterrupt:
        out.status, out.error = "interrupted", "interrupted by user"
        pool.release()
    finally:
        out.telemetry = sampler.stop() if sampler.started else None


def _scored(r: RequestResult, req: Request) -> RequestResult:
    if not r.ok or req.evaluator is None:
        return r
    s = evaluate(req.evaluator, r.content, req)
    return replace(
        r,
        score=s.value if s.has_objective else None,
        passed=s.passed if s.has_objective else None,
        eval_detail=s.detail or None,
        judge_pending=s.pending_judge,
    )


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

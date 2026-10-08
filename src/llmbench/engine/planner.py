"""Experiment -> ordered list of runs, and run fingerprints.

Order: every bench run first, then the llama-server runs (latency and quality) grouped by
*server key* (build, model and the non-sampling config), so each server process is started once
and reused across phases, suites and sampling variants.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmbench.config.experiment import BenchPhase, BuildSpec, Experiment, ModelSpec, SuitePhase
from llmbench.config.llama import LlamaConfig
from llmbench.config.loader import load_suite
from llmbench.config.matrix import Variant, apply_variant
from llmbench.config.suite import Suite
from llmbench.results.models import Phase

# Fields that do not change what is measured, so they stay out of the fingerprint.
_NON_IDENTITY_BENCH_FIELDS = {"enabled", "timeout_s", "unstable_cv"}
_NON_IDENTITY_SUITE_FIELDS = {"enabled", "suites", "request_timeout_s", "startup_timeout_s"}
SERVER_PHASES: tuple[Phase, ...] = ("latency", "quality", "serving")
"""Phases that send suites to llama-server (`lm_eval` also uses the server, without suites)."""


@dataclass(frozen=True)
class RunSpec:
    phase: Phase
    build: BuildSpec
    model: ModelSpec
    variant: Variant
    config: LlamaConfig
    suite: Suite | None = None
    suite_path: Path | None = None
    concurrency: int | None = None
    """Serving phase: number of concurrent clients of this run."""

    def server_key(self) -> str:
        """Runs with the same key can share one llama-server process."""
        return _canonical(
            {
                "build": self.build.id,
                "model": self.model.id,
                "config": self.config.server_identity(),
                "mode": self.suite.server_mode if self.suite else "generate",
            }
        )


@dataclass(frozen=True)
class Plan:
    runs: list[RunSpec]
    builds: int
    models: int
    variants: int
    tests_per_run: int
    repetitions: int
    suites: dict[Phase, list[Suite]] = field(default_factory=dict[Phase, list[Suite]])
    passes: dict[Phase, int] = field(default_factory=dict[Phase, int])

    @property
    def bench_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase == "bench"]

    @property
    def latency_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase == "latency"]

    @property
    def quality_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase == "quality"]

    @property
    def serving_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase == "serving"]

    @property
    def lm_eval_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase == "lm_eval"]

    @property
    def server_runs(self) -> list[RunSpec]:
        return [r for r in self.runs if r.phase != "bench"]

    @property
    def total_tests(self) -> int:
        return len(self.bench_runs) * self.tests_per_run

    @property
    def total_executions(self) -> int:
        return self.total_tests * self.repetitions

    @property
    def servers(self) -> int:
        return len({r.server_key() for r in self.server_runs})

    serving_requests: int | None = None

    def requests(self, phase: Phase) -> int | None:
        """Timed requests of a phase; None if a dataset's size is unknown before download."""
        runs = [r for r in self.runs if r.phase == phase]
        if phase == "serving" and self.serving_requests is not None:
            return self.serving_requests * len(runs)
        sizes = [suite_size(r.suite) for r in runs]
        if any(s is None for s in sizes):
            return None
        return sum(s or 0 for s in sizes) * self.passes.get(phase, 1)


def make_plan(exp: Experiment, *, model_ids: list[str] | None = None) -> Plan:
    models = exp.enabled_models
    if model_ids:
        unknown = set(model_ids) - {m.id for m in exp.models}
        if unknown:
            raise ValueError(f"unknown model id(s): {', '.join(sorted(unknown))}")
        models = [m for m in exp.models if m.id in model_ids]

    variants = exp.matrix.expand()
    configs = [(v, apply_variant(exp.defaults, v)) for v in variants]
    bench = exp.phases.bench
    runs = _bench_runs(exp, models, configs) if bench.enabled else []

    suites: dict[Phase, list[Suite]] = {}
    passes: dict[Phase, int] = {}
    served: list[RunSpec] = []
    for phase in SERVER_PHASES:
        cfg_phase = phase_config(exp, phase)
        if not cfg_phase.enabled:
            continue
        loaded = [(p, load_suite(p)) for p in cfg_phase.suites]
        suites[phase] = [s for _, s in loaded]
        passes[phase] = cfg_phase.passes
        levels: list[int | None] = (
            list(exp.phases.serving.concurrency) if phase == "serving" else [None]
        )
        served += [
            RunSpec(phase, build, model, variant, _serving_config(cfg, levels), suite, path, c)
            for build in exp.llama_cpp.builds
            for model in models
            for variant, cfg in configs
            for path, suite in loaded
            if model.suites is None or suite.name in model.suites
            for c in levels
        ]
    if exp.phases.lm_eval.enabled:
        served += [
            RunSpec("lm_eval", build, model, variant, cfg)
            for build in exp.llama_cpp.builds
            for model in models
            for variant, cfg in configs
            if model.suites is None  # models limited to some suites are not general LMs
        ]
    # stable sort: keeps phase and matrix order inside each server group
    order = {k: i for i, k in enumerate(dict.fromkeys(r.server_key() for r in served))}
    runs += sorted(served, key=lambda r: order[r.server_key()])

    return Plan(
        runs=runs,
        builds=len(exp.llama_cpp.builds),
        models=len(models),
        variants=len(variants),
        tests_per_run=tests_per_run(bench),
        repetitions=bench.repetitions,
        suites=suites,
        passes=passes,
        serving_requests=exp.phases.serving.requests,
    )


def _serving_config(cfg: LlamaConfig, levels: list[int | None]) -> LlamaConfig:
    """Serving needs a slot per concurrent client: raise `server.parallel` if it is lower."""
    top = max((c for c in levels if c is not None), default=None)
    if top is None or (cfg.server.parallel is not None and cfg.server.parallel >= top):
        return cfg
    return cfg.model_copy(update={"server": cfg.server.model_copy(update={"parallel": top})})


def _bench_runs(
    exp: Experiment, models: list[ModelSpec], configs: list[tuple[Variant, LlamaConfig]]
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for build in exp.llama_cpp.builds:
        for model in models:
            seen: set[str] = set()
            for variant, cfg in configs:
                key = _canonical(cfg.bench_identity())
                if key in seen:  # e.g. variants that only differ in sampling
                    continue
                seen.add(key)
                runs.append(RunSpec("bench", build, model, variant, cfg))
    return runs


def phase_config(exp: Experiment, phase: Phase) -> SuitePhase:
    if phase == "serving":
        return exp.phases.serving
    return exp.phases.latency if phase == "latency" else exp.phases.quality


def tests_per_run(bench: BenchPhase) -> int:
    """llama-bench crosses its own lists: each pp/tg/pg shape runs at every depth."""
    shapes = sum(1 for p in bench.n_prompt if p) + sum(1 for n in bench.n_gen if n) + len(bench.pg)
    return shapes * len(bench.n_depth)


def fingerprint(
    *,
    build: dict[str, Any],
    model_sha256: str,
    config: LlamaConfig,
    bench: BenchPhase,
) -> str:
    """Stable id of a bench measurement: same inputs on the same build -> same fingerprint."""
    return _sha(
        {
            "phase": "bench",
            "build": build,
            "model_sha256": model_sha256,
            "config": config.bench_identity(),
            "bench": bench.model_dump(mode="json", exclude=_NON_IDENTITY_BENCH_FIELDS),
        }
    )


def suite_fingerprint(
    *,
    phase: Phase,
    build: dict[str, Any],
    model_sha256: str,
    draft_sha256: str | None,
    config: LlamaConfig,
    mmproj_sha256: str | None = None,
    sampling: dict[str, Any],
    suite: Suite,
    phase_config: SuitePhase,
    requests_digest: str | None = None,
    concurrency: int | None = None,
) -> str:
    """Identity of a llama-server run.

    Synthetic requests derive deterministically from the suite and the model's tokenizer; for
    items/files/datasets the digest of the materialized requests is included.
    """
    return _sha(
        {
            "phase": phase,
            "build": build,
            "model_sha256": model_sha256,
            "draft_sha256": draft_sha256,
            "mmproj_sha256": mmproj_sha256,
            "config": config.server_identity(),
            "sampling": sampling,
            "suite": suite.model_dump(mode="json", by_alias=True),
            "requests": requests_digest,
            "concurrency": concurrency,
            "phase_config": phase_config.model_dump(
                mode="json", exclude=_NON_IDENTITY_SUITE_FIELDS
            ),
        }
    )


def suite_size(suite: Suite | None) -> int | None:
    if suite is None:
        return 0
    src = suite.source
    if src.synthetic is not None:
        return src.synthetic.count
    if src.items is not None:
        return len(src.items)
    if src.hf_dataset is not None:
        return src.hf_dataset.limit
    return None


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()

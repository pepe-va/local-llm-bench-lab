"""Experiment -> ordered list of runs, and run fingerprints."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from llmbench.config.experiment import BenchPhase, BuildSpec, Experiment, ModelSpec
from llmbench.config.llama import LlamaConfig
from llmbench.config.matrix import Variant, apply_variant

# Fields that do not change what is measured, so they stay out of the fingerprint.
_NON_IDENTITY_BENCH_FIELDS = {"enabled", "timeout_s", "unstable_cv"}


@dataclass(frozen=True)
class RunSpec:
    build: BuildSpec
    model: ModelSpec
    variant: Variant
    config: LlamaConfig


@dataclass(frozen=True)
class Plan:
    runs: list[RunSpec]
    builds: int
    models: int
    variants: int
    tests_per_run: int
    repetitions: int

    @property
    def total_tests(self) -> int:
        return len(self.runs) * self.tests_per_run

    @property
    def total_executions(self) -> int:
        return self.total_tests * self.repetitions


def make_plan(exp: Experiment, *, model_ids: list[str] | None = None) -> Plan:
    models = exp.enabled_models
    if model_ids:
        unknown = set(model_ids) - {m.id for m in exp.models}
        if unknown:
            raise ValueError(f"unknown model id(s): {', '.join(sorted(unknown))}")
        models = [m for m in exp.models if m.id in model_ids]

    variants = exp.matrix.expand()
    configs = [(v, apply_variant(exp.defaults, v)) for v in variants]
    runs = [
        RunSpec(build, model, variant, cfg)
        for build in exp.llama_cpp.builds
        for model in models
        for variant, cfg in configs
    ]
    bench = exp.phases.bench
    return Plan(
        runs=runs if bench.enabled else [],
        builds=len(exp.llama_cpp.builds),
        models=len(models),
        variants=len(variants),
        tests_per_run=tests_per_run(bench),
        repetitions=bench.repetitions,
    )


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
    """Stable id of *what is measured*: same inputs on the same build -> same fingerprint."""
    payload = {
        "phase": "bench",
        "build": build,
        "model_sha256": model_sha256,
        "config": config.model_dump(mode="json"),
        "bench": bench.model_dump(mode="json", exclude=_NON_IDENTITY_BENCH_FIELDS),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

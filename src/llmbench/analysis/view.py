"""One row per configuration (build x model x variant) with every metric it produced.

This is the shared input of gates, `llb guide`, `llb report` and the dashboard. It reads only
the SQLite store, so it works on any results directory, also on another machine.
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from llmbench.config.llama import LlamaConfig
from llmbench.results.store import Store

type Agg = Literal["mean", "median", "min", "max"]


@dataclass
class ConfigRow:
    build_id: str
    build_commit: str
    model_id: str
    variant: dict[str, Any]
    model: dict[str, Any]
    config: dict[str, Any]
    values: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    runs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    @property
    def key(self) -> str:
        return json.dumps([self.build_id, self.model_id, self.variant], sort_keys=True)

    @property
    def label(self) -> str:
        variant = ", ".join(f"{k}={v}" for k, v in self.variant.items()) or "defaults"
        return f"{self.model_id} · {self.build_id} · {variant}"

    @property
    def quant(self) -> str | None:
        gguf = cast(dict[str, Any], self.model.get("gguf") or {})
        return cast(str | None, gguf.get("quant"))

    @property
    def failed_runs(self) -> int:
        return sum(1 for r in self.runs if r["status"] != "completed")

    def add(self, name: str, value: float | None) -> None:
        if value is not None:
            self.values.setdefault(name, []).append(float(value))

    def metric(self, name: str, agg: Agg = "mean") -> float | None:
        xs = self.values.get(name)
        if not xs:
            return None
        if agg == "mean":
            return statistics.fmean(xs)
        if agg == "median":
            return statistics.median(xs)
        return min(xs) if agg == "min" else max(xs)


def build_view(store: Store, experiment: str) -> list[ConfigRow]:
    runs = store.latest_runs(experiment)
    rows: dict[str, ConfigRow] = {}
    for run in runs:
        variant = cast(dict[str, Any], run["variant"] or {})
        key = json.dumps([run["build_id"], run["model_id"], variant], sort_keys=True)
        if key not in rows:
            rows[key] = ConfigRow(
                build_id=run["build_id"],
                build_commit=run["build_commit"],
                model_id=run["model_id"],
                variant=variant,
                model=run["model"] or {},
                config=run["config"] or {},
            )
        rows[key].runs.append(run)

    for run in runs:
        if run["status"] != "completed":
            continue
        if run["phase"] == "bench":
            # llama-bench is deduplicated across variants that only differ in what it cannot
            # see (e.g. sampling): its numbers belong to every such configuration.
            ident = _bench_identity(run["config"])
            targets = [
                r
                for r in rows.values()
                if (r.build_id, r.model_id) == (run["build_id"], run["model_id"])
                and _bench_identity(r.config) == ident
            ]
        else:
            targets = [rows[_key(run)]]
        for row in targets:
            _add_run(row, run)
    return list(rows.values())


def _key(run: dict[str, Any]) -> str:
    return json.dumps([run["build_id"], run["model_id"], run["variant"] or {}], sort_keys=True)


def _bench_identity(config: dict[str, Any] | None) -> str:
    data = {k: v for k, v in (config or {}).items() if k != "effective_sampling"}
    return json.dumps(LlamaConfig.model_validate(data).bench_identity(), sort_keys=True)


def _add_run(row: ConfigRow, run: dict[str, Any]) -> None:
    tel: dict[str, Any] = run["telemetry"] or {}
    row.add("temp_peak_c", tel.get("temp_peak_c"))
    row.add("peak_rss_mb", tel.get("max_rss_mb"))
    row.add("throttled", 1.0 if _throttled(tel) else 0.0)
    if run["phase"] == "bench":
        for m in run["measurements"]:
            row.add(f"tps:{m['label']}", m["avg_ts"])
        return
    for name, value in run["metrics"].items():
        row.add(name, value)
        row.add(f"{name}@{run['suite']}", value)
    row.add("peak_rss_mb", run["metrics"].get("server_peak_rss_mb"))


def _throttled(tel: dict[str, Any]) -> bool:
    before, after = tel.get("throttled_before"), tel.get("throttled_after")
    return after is not None and (after & 0xF != 0 or after != before)

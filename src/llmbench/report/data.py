"""Everything a report shows, computed once from the store (no rendering dependencies)."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from llmbench import __version__
from llmbench.analysis.view import ConfigRow, build_view
from llmbench.config.experiment import Gate, Select
from llmbench.gates.deploy import Deployment, deployment
from llmbench.gates.rules import Verdict, check, select
from llmbench.results.store import Store


@dataclass
class ReportData:
    experiment: str
    generated_at: str
    llb_version: str
    machine: dict[str, Any]
    rows: list[ConfigRow]
    bench: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    latency: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    quality: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    failures: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    gates: list[Gate] = field(default_factory=list[Gate])
    verdicts: list[Verdict] = field(default_factory=list[Verdict])
    winner: Verdict | None = None
    select: Select | None = None
    deploy: Deployment | None = None

    @property
    def runs(self) -> int:
        return sum(len(r.runs) for r in self.rows)


def collect(
    store: Store, experiment: str, gates: list[Gate] | None = None, sel: Select | None = None
) -> ReportData:
    rows = build_view(store, experiment)
    data = ReportData(
        experiment=experiment,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        llb_version=__version__,
        machine=_machine(store, experiment),
        rows=rows,
        gates=gates or [],
        select=sel,
    )
    for row in rows:
        for run in row.runs:
            _add_run(data, row, run)
    if data.gates or sel is not None:
        data.verdicts = check(data.gates, rows)
        data.winner = select(data.verdicts, sel)
        if data.winner is not None:
            data.deploy = deployment(data.winner.row)
    return data


def _machine(store: Store, experiment: str) -> dict[str, Any]:
    row = store.db.execute(
        "SELECT env FROM runs WHERE experiment = ? ORDER BY id LIMIT 1", (experiment,)
    ).fetchone()
    return cast(dict[str, Any], json.loads(row[0])) if row else {}


def _variant(row: ConfigRow) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.variant.items()) or "defaults"


def _add_run(data: ReportData, row: ConfigRow, run: dict[str, Any]) -> None:
    base = {
        "model": row.model_id,
        "quant": row.quant,
        "build": f"{row.build_id} ({row.build_commit[:7]})",
        "variant": _variant(row),
        "config": row.label,
    }
    tel = cast(dict[str, Any], run.get("telemetry") or {})
    metrics = cast(dict[str, Any], run.get("metrics") or {})
    if run["status"] != "completed":
        error = str(run.get("error") or "").strip().splitlines()
        data.failures.append(
            base | {"phase": run["phase"], "suite": run.get("suite"), "status": run["status"],
                    "error": error[0] if error else ""}
        )  # fmt: skip
        return
    if run["phase"] == "bench":
        for m in cast(list[dict[str, Any]], run["measurements"]):
            data.bench.append(
                base
                | {
                    "test": m["label"],
                    "tps": m["avg_ts"],
                    "stddev": m["stddev_ts"],
                    "cv": m["cv"],
                    "unstable": bool(m["unstable"]),
                    "rss_mb": tel.get("max_rss_mb"),
                    "temp_peak": tel.get("temp_peak_c"),
                }
            )
    elif run["phase"] == "latency":
        data.latency.append(base | {"suite": run["suite"]} | metrics)
    elif run["phase"] == "quality":
        data.quality.append(base | {"suite": run["suite"]} | metrics)

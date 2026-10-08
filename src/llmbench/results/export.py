"""Flat exports derived from the SQLite store (it stays the source of truth)."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llmbench.results.models import RunRecord
from llmbench.results.store import Store

CSV_COLUMNS = [
    "experiment",
    "status",
    "model_id",
    "build_id",
    "build_commit",
    "variant",
    "threads",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "flash_attn",
    "test",
    "n_prompt",
    "n_gen",
    "n_depth",
    "avg_ts",
    "stddev_ts",
    "median_ts",
    "cv",
    "unstable",
    "temp_before_c",
    "temp_peak_c",
    "rss_peak_mb",
    "throttled_after",
    "started_at",
    "fingerprint",
    "error",
]


def append_jsonl(path: Path, run: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(run), default=str) + "\n")


def write_csv(path: Path, store: Store, experiment: str) -> int:
    rows = [_csv_row(r) for r in store.summary_rows(experiment)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _csv_row(r: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(r["config"])
    tel: dict[str, Any] = json.loads(r["telemetry"]) if r["telemetry"] else {}
    variant = json.loads(r["variant"])
    return {
        "experiment": r["experiment"],
        "status": r["status"],
        "model_id": r["model_id"],
        "build_id": r["build_id"],
        "build_commit": r["build_commit"],
        "variant": ";".join(f"{k}={v}" for k, v in variant.items()),
        "threads": cfg["load"]["threads"],
        "batch_size": cfg["load"]["batch_size"],
        "ubatch_size": cfg["load"]["ubatch_size"],
        "cache_type_k": cfg["kv"]["cache_type_k"],
        "cache_type_v": cfg["kv"]["cache_type_v"],
        "flash_attn": cfg["load"]["flash_attn"],
        "test": r["label"],
        "n_prompt": r["n_prompt"],
        "n_gen": r["n_gen"],
        "n_depth": r["n_depth"],
        "avg_ts": _round(r["avg_ts"]),
        "stddev_ts": _round(r["stddev_ts"]),
        "median_ts": _round(r["median_ts"]),
        "cv": _round(r["cv"], 4),
        "unstable": r["unstable"],
        "temp_before_c": tel.get("temp_before_c"),
        "temp_peak_c": tel.get("temp_peak_c"),
        "rss_peak_mb": _round(tel.get("max_rss_mb") or tel.get("rss_peak_mb")),
        "throttled_after": tel.get("throttled_after"),
        "started_at": r["started_at"],
        "fingerprint": r["fingerprint"][:12],
        "error": (r["error"] or "").splitlines()[0] if r["error"] else "",
    }


def _round(v: float | None, nd: int = 2) -> float | None:
    return round(v, nd) if v is not None else None

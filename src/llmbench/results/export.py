"""Flat exports derived from the SQLite store (it stays the source of truth)."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from llmbench.results.models import RunRecord
from llmbench.results.store import Store

CSV_COLUMNS = [
    "experiment",
    "status",
    "model_id",
    "model_quant",
    "model_params",
    "model_commit",
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
    "ci_low_ts",
    "ci_high_ts",
    "median_ts",
    "iqr_ts",
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


LATENCY_METRICS = [
    "requests",
    "requests_failed",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "e2e_ms_p50",
    "e2e_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "prompt_tps_mean",
    "gen_tps_mean",
    "gen_tps_p50",
    "prompt_tokens_total",
    "gen_tokens_total",
    "server_load_s",
    "server_peak_rss_mb",
    "spec_acceptance",
]
LATENCY_COLUMNS = [
    "experiment",
    "status",
    "model_id",
    "model_quant",
    "build_id",
    "build_commit",
    "suite",
    "variant",
    *LATENCY_METRICS,
    "temp_peak_c",
    "started_at",
    "fingerprint",
    "error",
]


QUALITY_METRICS = [
    "requests",
    "requests_failed",
    "scored",
    "score_mean",
    "pass_rate",
    "judge_pending",
    "ttft_ms_p50",
    "e2e_ms_p50",
    "gen_tps_mean",
    "words_per_s_mean",
]


SERVING_METRICS = [
    "concurrency",
    "requests",
    "requests_failed",
    "requests_per_s",
    "gen_tps_aggregate",
    "prompt_tps_aggregate",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "e2e_ms_p50",
    "e2e_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "error_rate",
    "server_peak_rss_mb",
]


def write_serving_csv(path: Path, store: Store, experiment: str) -> int:
    return _write_server_csv(path, store, experiment, "serving", SERVING_METRICS)


def write_latency_csv(path: Path, store: Store, experiment: str) -> int:
    return _write_server_csv(path, store, experiment, "latency", LATENCY_METRICS)


def write_quality_csv(path: Path, store: Store, experiment: str) -> int:
    return _write_server_csv(path, store, experiment, "quality", QUALITY_METRICS, extra=True)


def _write_server_csv(
    path: Path,
    store: Store,
    experiment: str,
    phase: str,
    metrics: list[str],
    *,
    extra: bool = False,
) -> int:
    rows = store.latency_rows(experiment, phase)
    if not rows:
        return 0
    # quality runs also get per-category and judge columns (whatever the runs recorded)
    names = list(metrics)
    if extra:
        for r in rows:
            names += [
                k[2:]
                for k in r
                if k.startswith("m.")
                and k[2:] not in names
                and (k.startswith("m.score_mean:") or k.startswith("m.judge."))
            ]
    columns = [c for c in LATENCY_COLUMNS if c not in LATENCY_METRICS]
    columns = (
        columns[: columns.index("temp_peak_c")] + names + columns[columns.index("temp_peak_c") :]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            model: dict[str, Any] = json.loads(r["model"])
            tel: dict[str, Any] = json.loads(r["telemetry"]) if r["telemetry"] else {}
            variant = json.loads(r["variant"])
            w.writerow(
                {
                    "experiment": r["experiment"],
                    "status": r["status"],
                    "model_id": r["model_id"],
                    "model_quant": cast(dict[str, Any], model.get("gguf") or {}).get("quant"),
                    "build_id": r["build_id"],
                    "build_commit": r["build_commit"],
                    "suite": r["suite"],
                    "variant": ";".join(f"{k}={v}" for k, v in variant.items()),
                    **{m: _round(r.get(f"m.{m}"), 3) for m in names},
                    "temp_peak_c": tel.get("temp_peak_c"),
                    "started_at": r["started_at"],
                    "fingerprint": r["fingerprint"][:12],
                    "error": (r["error"] or "").splitlines()[0] if r["error"] else "",
                }
            )
    return len(rows)


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
    model: dict[str, Any] = json.loads(r["model"])
    gguf: dict[str, Any] = model.get("gguf") or {}
    return {
        "experiment": r["experiment"],
        "status": r["status"],
        "model_id": r["model_id"],
        "model_quant": gguf.get("quant"),
        "model_params": gguf.get("n_params"),
        "model_commit": (model.get("commit") or "")[:12],
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
        "ci_low_ts": _round(r["ci_low_ts"]),
        "ci_high_ts": _round(r["ci_high_ts"]),
        "median_ts": _round(r["median_ts"]),
        "iqr_ts": _round(r["iqr_ts"]),
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

"""What the dashboard shows beyond `llb report`: the plan, the hardware and cross-experiment
comparisons, all read back from the results store (no experiment YAML needed)."""

import json
from typing import Any, cast

from llmbench.analysis.view import ConfigRow
from llmbench.results.store import Store

ENV_FIELDS = [
    ("hostname", "host"),
    ("cpu_model", "CPU"),
    ("cpu_count", "logical CPUs"),
    ("ram_total_mb", "RAM (MB)"),
    ("machine", "arch"),
    ("os", "OS"),
    ("kernel", "kernel"),
    ("governor", "CPU governor"),
    ("python", "python"),
    ("llb_version", "llb"),
]

SETTING_HELP = {
    "load.threads": "CPU threads used to generate tokens",
    "load.threads_batch": "CPU threads used to process the prompt (default: same as threads)",
    "load.ctx_size": "context window in tokens: prompt + answer must fit",
    "load.batch_size": "logical batch: max prompt tokens submitted per step",
    "load.ubatch_size": "physical micro-batch actually computed at once (<= batch_size)",
    "load.flash_attn": "flash attention kernel (on/off/auto): less memory, often faster",
    "load.repack": "repack weights at load time for faster CPU kernels",
    "kv.cache_type_k": "precision of the attention K cache (f16, q8_0, q4_0…): lower = less RAM",
    "kv.cache_type_v": "precision of the attention V cache",
    "gpu.n_gpu_layers": "layers offloaded to the GPU (0 = CPU only)",
    "server.parallel": "request slots in llama-server (concurrent requests)",
    "server.cont_batching": "continuous batching: mix new requests into running ones",
    "server.reasoning": "whether thinking models may emit a reasoning block",
    "sampling.temperature": "randomness of sampling (0 = always the most likely token)",
    "sampling.seed": "random seed, fixed for repeatable outputs",
    "sampling.top_k": "sample only among the k most likely tokens",
    "sampling.top_p": "sample among the most likely tokens up to this cumulative probability",
    "sampling.max_tokens": "maximum tokens generated per answer",
}

COMPARE_METRICS = [
    ("peak_rss_mb", "peak RSS MB", "lower"),
    ("ttft_ms_p50", "TTFT p50 ms", "lower"),
    ("ttft_ms_p95", "TTFT p95 ms", "lower"),
    ("itl_ms_p50", "ITL p50 ms", "lower"),
    ("gen_tps_mean", "gen t/s", "higher"),
    ("score_mean", "score", "higher"),
    ("requests_per_s", "req/s", "higher"),
]


def _decoded(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) and raw else raw


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"load": {"threads": 8}}` -> `{"load.threads": 8}`, without unset (None/empty) values."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(cast(dict[str, Any], value), f"{name}.")
        elif value is not None and value not in ([], ""):
            out[name] = value
    return out


def plan(store: Store, experiment: str, rows: list[ConfigRow]) -> dict[str, Any]:
    runs = store.latest_runs(experiment)
    models: dict[str, dict[str, Any]] = {}
    builds: dict[str, dict[str, Any]] = {}
    for r in runs:
        models.setdefault(r["model_id"], _model_summary(r["model"] or {}, r["model_id"]))
        builds.setdefault(r["build_id"], r["build"] or {"id": r["build_id"]})
    matrix: dict[str, list[Any]] = {}
    for row in rows:
        for k, v in row.variant.items():
            if v not in matrix.setdefault(k, []):
                matrix[k].append(v)
    settings: dict[str, list[Any]] = {}
    for r in runs:
        for k, v in flatten(r["config"] or {}).items():
            shown = json.dumps(v) if isinstance(v, list | dict) else v
            if shown not in settings.setdefault(k, []):
                settings[k].append(shown)
    common = {k: v[0] for k, v in sorted(settings.items()) if len(v) == 1}
    varying = {k: v for k, v in sorted(settings.items()) if len(v) > 1}
    return {
        "models": list(models.values()),
        "builds": list(builds.values()),
        "matrix": matrix,
        "phases": _phases(store, experiment, runs),
        "common": common,
        "varying": varying,
        "combinations": len(rows),
    }


def _model_summary(model: dict[str, Any], model_id: str) -> dict[str, Any]:
    gguf = cast(dict[str, Any], model.get("gguf") or {})
    n_params = gguf.get("n_params")
    size = model.get("size")
    return {
        "id": model_id,
        "source": model.get("source"),
        "arch": gguf.get("architecture"),
        "quant": gguf.get("quant"),
        "params": f"{n_params / 1e9:.2f} B" if n_params else None,
        "size_gb": round(size / 1e9, 2) if size else None,
        "context": gguf.get("context_length"),
        "layers": gguf.get("block_count"),
        "commit": (model.get("commit") or "")[:10] or None,
        "sha256": (model.get("sha256") or "")[:12] or None,
    }


def _phases(store: Store, experiment: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in runs:
        p = out.setdefault(
            r["phase"],
            {"phase": r["phase"], "runs": 0, "failed": 0, "seconds": 0.0, "suites": [],
             "concurrency": [], "tests": [], "repetitions": []},
        )  # fmt: skip
        p["runs"] += 1
        p["failed"] += r["status"] != "completed"
        p["seconds"] += r["duration_s"] or 0.0
        for key, value in (("suites", r.get("suite")), ("concurrency", r.get("concurrency"))):
            if value is not None and value not in p[key]:
                p[key].append(value)
    for label, n_prompt, n_gen, n_depth, samples in store.db.execute(
        """SELECT m.label, m.n_prompt, m.n_gen, m.n_depth, m.samples_ts
           FROM measurements m JOIN runs r ON r.id = m.run_id WHERE r.experiment = ?""",
        (experiment,),
    ):
        bench = out.get("bench")
        if bench is None:
            break
        test = f"{label} (prompt {n_prompt}, gen {n_gen}, depth {n_depth})"
        if test not in bench["tests"]:
            bench["tests"].append(test)
        reps = len(json.loads(samples)) if samples else 0
        if reps not in bench["repetitions"]:
            bench["repetitions"].append(reps)
    return list(out.values())


def hardware(store: Store, experiment: str) -> dict[str, Any]:
    rows = [
        dict(r)
        for r in store.db.execute(
            """SELECT id, phase, status, model_id, variant, suite, concurrency, env, telemetry,
                      cooldown, duration_s
               FROM runs WHERE experiment = ? ORDER BY id""",
            (experiment,),
        )
    ]
    envs: list[dict[str, Any]] = [_decoded(r["env"]) or {} for r in rows]
    fields: list[dict[str, Any]] = []
    for key, label in ENV_FIELDS:
        values = list(dict.fromkeys(str(e.get(key)) for e in envs if e.get(key) is not None))
        if values:
            fields.append({"key": label, "value": values[0], "others": values[1:]})
    usage: list[dict[str, Any]] = []
    for r in rows:
        tel = cast(dict[str, Any], _decoded(r["telemetry"]) or {})
        cool = cast(dict[str, Any], _decoded(r["cooldown"]) or {})
        variant = cast(dict[str, Any], _decoded(r["variant"]) or {})
        usage.append(
            {
                "id": r["id"],
                "phase": r["phase"],
                "status": r["status"],
                "model": r["model_id"],
                "variant": ", ".join(f"{k}={v}" for k, v in variant.items()) or "defaults",
                "duration_s": r["duration_s"],
                "rss_peak_mb": tel.get("max_rss_mb") or tel.get("rss_peak_mb"),
                "ram_before_mb": tel.get("ram_before_mb"),
                "ram_peak_mb": tel.get("ram_peak_mb"),
                "cpu_freq_avg_mhz": tel.get("cpu_freq_avg_mhz"),
                "cpu_freq_min_mhz": tel.get("cpu_freq_min_mhz"),
                "cpu_s": (tel.get("cpu_user_s") or 0) + (tel.get("cpu_sys_s") or 0) or None,
                "temp_peak_c": tel.get("temp_peak_c"),
                "cooldown_s": cool.get("waited_s"),
                "cooldown_note": cool.get("skipped"),
            }
        )
    return {"env": fields, "usage": usage, "env_changed": any(f["others"] for f in fields)}


def compare_rows(experiment: str, sid: str, rows: list[ConfigRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics = {
            name.removeprefix("tps:"): row.metric(name)
            for name in sorted(row.values)
            if name.startswith("tps:")
        }
        metrics |= {name: row.metric(name) for name, _, _ in COMPARE_METRICS}
        out.append(
            {
                "experiment": experiment,
                "sid": sid,
                "model": row.model_id,
                "quant": row.quant,
                "build": f"{row.build_id} ({row.build_commit[:7]})",
                "variant": ", ".join(f"{k}={v}" for k, v in row.variant.items()) or "defaults",
                "metrics": metrics,
                "failed": row.failed_runs,
            }
        )
    return out


def compare_columns(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(metric key, header, better) for every metric at least one compared row has."""
    tests = sorted(
        {k for r in rows for k, v in r["metrics"].items() if v is not None}
        - {name for name, _, _ in COMPARE_METRICS},
        key=lambda t: (not t.startswith("pp"), t),
    )
    cols = [(t, f"{t} t/s", "higher") for t in tests]
    cols += [
        (name, header, better)
        for name, header, better in COMPARE_METRICS
        if any(r["metrics"].get(name) is not None for r in rows)
    ]
    return cols

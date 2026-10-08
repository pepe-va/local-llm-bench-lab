"""Descriptive statistics over per-repetition samples."""

import json
import statistics
from dataclasses import replace

from llmbench.evaluation.retrieval import pearson, spearman
from llmbench.results.models import Measurement, RequestResult


def cv(samples: list[float]) -> float:
    """Coefficient of variation (sample stddev / mean); 0 for fewer than 2 samples."""
    if len(samples) < 2:  # noqa: PLR2004
        return 0.0
    mean = statistics.fmean(samples)
    return statistics.stdev(samples) / mean if mean else 0.0


def annotate(m: Measurement, *, unstable_cv: float, throttled: bool) -> Measurement:
    """Add median, CV and the `unstable` flag. Unstable runs are kept, never hidden."""
    c = cv(m.samples_ts)
    median = statistics.median(m.samples_ts) if m.samples_ts else m.avg_ts
    return replace(m, median_ts=median, cv=c, unstable=throttled or c > unstable_cv)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100); None for no data."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_metrics(results: list[RequestResult]) -> dict[str, float | None]:
    """Aggregates over the successful requests of a latency run."""
    ok = [r for r in results if r.ok]
    out: dict[str, float | None] = {
        "requests": float(len(results)),
        "requests_failed": float(len(results) - len(ok)),
    }

    def series(name: str, values: list[float | None]) -> None:
        xs = [v for v in values if v is not None]
        out[f"{name}_mean"] = statistics.fmean(xs) if xs else None
        out[f"{name}_p50"] = percentile(xs, 50)
        out[f"{name}_p95"] = percentile(xs, 95)
        out[f"{name}_max"] = max(xs) if xs else None

    series("ttft_ms", [r.ttft_ms for r in ok])
    series("e2e_ms", [r.e2e_ms for r in ok])
    series("itl_ms", [x for r in ok for x in r.itl_ms])
    series("prompt_tps", [r.prompt_tps for r in ok])
    series("gen_tps", [r.gen_tps for r in ok])
    out["prompt_tokens_total"] = float(sum(r.prompt_n or 0 for r in ok))
    out["gen_tokens_total"] = float(sum(r.predicted_n or 0 for r in ok))
    return out


def quality_metrics(results: list[RequestResult]) -> dict[str, float | None]:
    """Scores of a quality run (objective evaluators only; judge scores are added later)."""
    scored = [r for r in results if r.ok and r.score is not None]
    out: dict[str, float | None] = {
        "scored": float(len(scored)),
        "score_mean": statistics.fmean(r.score for r in scored if r.score is not None)
        if scored
        else None,
        "pass_rate": sum(1 for r in scored if r.passed) / len(scored) if scored else None,
        "judge_pending": float(sum(1 for r in results if r.judge_pending)),
    }
    categories = sorted({r.category for r in scored if r.category})
    for c in categories:
        xs = [r.score for r in scored if r.category == c and r.score is not None]
        out[f"score_mean:{c}"] = statistics.fmean(xs) if xs else None
    chars = [len(r.content) / (r.e2e_ms / 1000) for r in results if r.ok and r.e2e_ms]
    words = [len(r.content.split()) / (r.e2e_ms / 1000) for r in results if r.ok and r.e2e_ms]
    out["chars_per_s_mean"] = statistics.fmean(chars) if chars else None
    out["words_per_s_mean"] = statistics.fmean(words) if words else None
    return out


def retrieval_metrics(
    results: list[RequestResult], endpoint: str, k: int
) -> dict[str, float | None]:
    """Embeddings: correlation of cosine similarity with labels. Rerank: nDCG@k, MRR, recall@k."""
    ok = [r for r in results if r.ok]
    if endpoint == "embeddings":
        pairs = [
            (r.similarity, r.label) for r in ok if r.similarity is not None and r.label is not None
        ]
        sims = [s for s, _ in pairs]
        labels = [lab for _, lab in pairs]
        return {
            "spearman": spearman(sims, labels),
            "pearson": pearson(sims, labels),
            "cosine_mean": statistics.fmean(sims) if sims else None,
            "embed_tps_mean": _mean([r.prompt_tps for r in ok]),
        }
    details = [json.loads(r.eval_detail) for r in ok if r.eval_detail]
    return {
        f"ndcg@{k}": _mean([r.score for r in ok]),
        "mrr": _mean([d["mrr"] for d in details]),
        f"recall@{k}": _mean([d[f"recall@{k}"] for d in details]),
        "top1_accuracy": _mean([1.0 if r.passed else 0.0 for r in ok]),
        "rerank_tps_mean": _mean([r.prompt_tps for r in ok]),
    }


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None

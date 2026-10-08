"""Descriptive statistics over per-repetition samples."""

import statistics
from dataclasses import replace

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

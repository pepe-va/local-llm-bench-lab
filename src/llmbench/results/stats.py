"""Descriptive statistics over per-repetition samples."""

import statistics
from dataclasses import replace

from llmbench.results.models import Measurement


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

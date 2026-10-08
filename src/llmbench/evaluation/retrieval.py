"""Embedding similarity and ranking metrics (no numpy: lists are short)."""

import math
import statistics


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:  # noqa: PLR2004
        return None
    return statistics.correlation(xs, ys)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation between predicted similarities and human labels (STS-style)."""
    if len(xs) < 2:  # noqa: PLR2004
        return None
    return pearson(_ranks(xs), _ranks(ys))


def ndcg_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance nDCG@k of a ranking (document indices, best first)."""
    dcg = sum(1 / math.log2(pos + 2) for pos, doc in enumerate(ranking[:k]) if doc in relevant)
    ideal = sum(1 / math.log2(pos + 2) for pos in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def mrr(ranking: list[int], relevant: set[int]) -> float:
    for pos, doc in enumerate(ranking):
        if doc in relevant:
            return 1 / (pos + 1)
    return 0.0


def recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0

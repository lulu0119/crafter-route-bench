"""Intervals for paired comparisons. A difference is reported only when the interval excludes zero."""

from __future__ import annotations

import math
import random
import statistics


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_delta(left: list[float], right: list[float], rng: random.Random, replicates: int = 4000) -> dict:
    deltas = [a - b for a, b in zip(left, right)]
    count = len(deltas)
    if count == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0, "p": 1.0, "effect": None}
    mean = statistics.fmean(deltas)
    if count == 1:
        return {"mean": mean, "low": mean, "high": mean, "n": 1, "p": 1.0, "effect": None}
    samples = []
    for _ in range(replicates):
        picks = [deltas[rng.randrange(count)] for _ in range(count)]
        samples.append(statistics.fmean(picks))
    samples.sort()
    low = samples[int(0.025 * (replicates - 1))]
    high = samples[int(0.975 * (replicates - 1))]
    return {
        "mean": mean,
        "low": low,
        "high": high,
        "n": count,
        "p": _permutation_p(deltas, rng, replicates),
        "effect": _paired_effect(deltas),
    }


def decision_scores(should: list, handed: list) -> dict:
    pairs = [(flag, bool(done)) for flag, done in zip(should, handed) if flag is not None]
    positive = [done for flag, done in pairs if flag]
    negative = [done for flag, done in pairs if not flag]
    true_positive = sum(positive)
    false_positive = sum(negative)
    needed = len(positive)
    negative_count = len(negative)
    decided = true_positive + false_positive
    recall = true_positive / needed if needed else None
    specificity = (negative_count - false_positive) / negative_count if negative_count else None
    balanced = None
    if recall is not None and specificity is not None:
        balanced = (recall + specificity) / 2
    return {
        "miss": (needed - true_positive, needed),
        "over": (false_positive, negative_count),
        "recall": (true_positive, needed),
        "precision": (true_positive, decided),
        "balanced": balanced,
    }


def interval_verdict(low: float, high: float) -> str:
    if low > 0:
        return "显著更高"
    if high < 0:
        return "显著更低"
    return "证据不足"


def cohen_kappa(left: list, right: list) -> float | None:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    expected = 0.0
    for label in labels:
        expected += (sum(a == label for a, _ in pairs) / total) * (sum(b == label for _, b in pairs) / total)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _paired_effect(deltas: list[float]) -> float | None:
    if len(deltas) < 2:
        return None
    spread = statistics.stdev(deltas)
    if spread == 0:
        return None
    return statistics.fmean(deltas) / spread


def _permutation_p(deltas: list[float], rng: random.Random, replicates: int) -> float:
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    extreme = 0
    for _ in range(replicates):
        signed = sum(delta if rng.randrange(2) else -delta for delta in deltas)
        if abs(signed) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (replicates + 1)

"""Small-sample statistics used by the v15 evaluation plan.

Only standard-library implementations are used so the analysis is reproducible
on the CPU instance without scipy/statsmodels.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    """A two-sided confidence interval."""

    low: float
    high: float


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Exact paired binary test result."""

    base_only: int
    treatment_only: int
    discordant: int
    p_value: float


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return Interval(max(0.0, center - half), min(1.0, center + half))


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def mcnemar_exact(base_only: int, treatment_only: int) -> McNemarResult:
    """Exact two-sided McNemar test using the binomial likelihood ordering."""

    if base_only < 0 or treatment_only < 0:
        raise ValueError("discordant counts must be non-negative")
    n = base_only + treatment_only
    if n == 0:
        return McNemarResult(0, 0, 0, 1.0)
    observed = _binom_pmf(base_only, n, 0.5)
    tolerance = observed * 1e-12 + 1e-15
    p_value = sum(
        probability
        for k in range(n + 1)
        if (probability := _binom_pmf(k, n, 0.5)) <= observed + tolerance
    )
    return McNemarResult(base_only, treatment_only, n, min(1.0, p_value))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-FDR adjusted q-values in the original order."""

    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _rejection_region(n: int, alpha: float = 0.05) -> set[int]:
    return {k for k in range(n + 1) if mcnemar_exact(k, n - k).p_value < alpha}


def binomial_power(n: int, true_share: float, alpha: float = 0.05) -> float:
    """Power of the exact two-sided McNemar test under a discordant share."""

    if not 0.0 <= true_share <= 1.0:
        raise ValueError("true_share must be in [0, 1]")
    return sum(_binom_pmf(k, n, true_share) for k in _rejection_region(n, alpha))


def minimum_detectable_share(
    n_discordant: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float | None:
    """Smallest true discordant share reaching target power."""

    if n_discordant <= 0:
        return None
    for step in range(5001):
        share = 0.5 + 0.5 * step / 5000
        if binomial_power(n_discordant, share, alpha) >= power:
            return share
    return 1.0


def minimum_detectable_risk_difference(
    n_total: int,
    n_discordant: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float | None:
    """Translate the discordant-share MDR to an aggregate risk difference."""

    share = minimum_detectable_share(n_discordant, alpha=alpha, power=power)
    if share is None or n_total <= 0:
        return None
    return (2.0 * share - 1.0) * n_discordant / n_total


def cluster_bootstrap_rate_diff(
    clusters: Sequence[Sequence[tuple[bool, bool]]],
    *,
    iterations: int = 20000,
    seed: int = 20261102,
) -> Interval:
    """Task-cluster bootstrap CI for treatment minus base violation rate.

    Each cluster is one task; each tuple is ``(base_violation, treatment_violation)``.
    """

    if not clusters:
        raise ValueError("clusters must be non-empty")
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        pairs = [pair for cluster in sampled for pair in cluster]
        if not pairs:
            continue
        base_rate = sum(base for base, _ in pairs) / len(pairs)
        treatment_rate = sum(treatment for _, treatment in pairs) / len(pairs)
        differences.append(treatment_rate - base_rate)
    differences.sort()
    low = differences[int(0.025 * (len(differences) - 1))]
    high = differences[int(0.975 * (len(differences) - 1))]
    return Interval(low, high)

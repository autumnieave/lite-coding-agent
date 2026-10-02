"""Pure helpers for per-sample severity weighting (no torch dependency)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def weight_for_prompt(prompt: str, weights: Mapping[str, float]) -> float:
    """Return the severity weight of the first constraint id found in ``prompt``.

    Prompts carry the constraint id as ``[V14P-C3-01]`` etc.  A prompt without a
    known id falls back to 1.0 so unrelated pairs keep their original scale.
    """

    for constraint_id, weight in weights.items():
        if constraint_id and constraint_id in prompt:
            return float(weight)
    return 1.0


def normalised_weights(values: Sequence[float]) -> list[float]:
    """Scale weights so their mean is 1.0 (keeps the mean loss comparable)."""

    if not values:
        return []
    mean = sum(values) / len(values)
    if mean <= 0:
        return [1.0 for _ in values]
    return [value / mean for value in values]

"""Tests for the v15 standard-library statistics helpers."""

from __future__ import annotations

import pytest

from training.v15_stats import (
    benjamini_hochberg,
    cluster_bootstrap_rate_diff,
    mcnemar_exact,
    minimum_detectable_risk_difference,
    wilson_ci,
)


def test_wilson_ci_known_bounds() -> None:
    interval = wilson_ci(0, 10)
    assert interval.low == 0.0
    assert 0.27 < interval.high < 0.29


def test_mcnemar_exact_symmetric_and_asymmetric() -> None:
    symmetric = mcnemar_exact(5, 5)
    assert symmetric.p_value == pytest.approx(1.0)
    asymmetric = mcnemar_exact(10, 1)
    assert asymmetric.p_value < 0.02


def test_benjamini_hochberg_monotone() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.04 * 4 / 3)
    assert adjusted[3] == pytest.approx(0.2)


def test_mdr_and_cluster_bootstrap() -> None:
    mdr = minimum_detectable_risk_difference(100, 20)
    assert mdr is not None and 0.0 < mdr < 0.5
    ci = cluster_bootstrap_rate_diff(
        [
            [(False, False), (True, False)],
            [(False, False), (False, False)],
        ],
        iterations=500,
    )
    assert ci.low <= ci.high

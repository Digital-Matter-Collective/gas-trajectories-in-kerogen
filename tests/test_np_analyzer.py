import warnings

import numpy as np
import pytest

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from processes.distribution_fitter import GammaFitter, WeibullFitter
from processes.trajectory_analyzer.np import NeymanPearsonAnalyzer


class _ZeroFitter:
    """Fake fitter whose density is exactly zero everywhere, to exercise the
    L_T/L_C == 0 division guard directly."""

    def pdf(self, x) -> np.ndarray:
        return np.zeros(np.asarray(x).shape, dtype=np.float64)


class _ConstantFitter:
    def __init__(self, value: float) -> None:
        self.value = value

    def pdf(self, x) -> np.ndarray:
        return np.full(np.asarray(x).shape, self.value, dtype=np.float64)


def _make_trajectory(n: int = 12) -> Trajectory:
    points = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.1, axis=0).astype(
        np.float32
    )
    times = np.arange(n, dtype=np.float32)
    box = BoundingBox(Range(0, 100), Range(0, 100), Range(0, 100))
    return Trajectory(points=points, times=times, box=box)


# --- analyze(): zero-density division guard ----------------------------------


def test_analyze_does_not_warn_or_produce_nan_when_both_densities_are_zero() -> (
    None
):
    trj = _make_trajectory()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        likelihood = NeymanPearsonAnalyzer.analyze(
            trj, _ZeroFitter(), _ZeroFitter()
        )

    assert np.all(np.isfinite(likelihood))


def test_analyze_produces_a_large_finite_ratio_when_trapped_density_is_zero() -> (
    None
):
    trj = _make_trajectory()

    likelihood = NeymanPearsonAnalyzer.analyze(
        trj, _ConstantFitter(0.3), _ZeroFitter()
    )

    assert np.all(np.isfinite(likelihood))
    assert np.all(likelihood > 0)


# --- calculate_threshold(): support beyond 1 nm (P1-12) -----------------------


def test_calculate_threshold_within_default_range_still_works() -> None:
    rng = np.random.default_rng(0)
    trap_fitter = GammaFitter()
    trap_fitter.fit(rng.gamma(shape=3.0, scale=0.1, size=2000))
    transition_fitter = WeibullFitter()
    transition_fitter.fit(rng.gamma(shape=2.0, scale=0.08, size=2000))

    threshold = NeymanPearsonAnalyzer.calculate_threshold(
        trap_fitter, transition_fitter, epsilon=0.01
    )
    assert threshold > 0


def test_calculate_threshold_default_x_max_fails_for_support_beyond_1nm() -> (
    None
):
    # Regression check for the bug itself: with support entirely beyond
    # 1 nm, the default [0, 1] search window finds essentially no
    # probability mass and cannot locate a threshold.
    rng = np.random.default_rng(0)
    trap_fitter = GammaFitter()
    trap_fitter.fit(rng.gamma(shape=5.0, scale=0.6, size=5000) + 2.0)
    transition_fitter = WeibullFitter()
    transition_fitter.fit(rng.gamma(shape=4.0, scale=0.5, size=5000) + 2.0)

    with pytest.raises(IndexError):
        NeymanPearsonAnalyzer.calculate_threshold(
            trap_fitter, transition_fitter, epsilon=0.01
        )


def test_calculate_threshold_x_max_covers_support_beyond_1nm() -> None:
    rng = np.random.default_rng(0)
    trap_fitter = GammaFitter()
    trap_fitter.fit(rng.gamma(shape=5.0, scale=0.6, size=5000) + 2.0)
    transition_fitter = WeibullFitter()
    transition_fitter.fit(rng.gamma(shape=4.0, scale=0.5, size=5000) + 2.0)

    threshold = NeymanPearsonAnalyzer.calculate_threshold(
        trap_fitter, transition_fitter, epsilon=0.01, x_max=10.0
    )
    assert threshold > 0
    assert np.isfinite(threshold)

import numpy as np
import pytest

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from processes.trajectory_analyzer.bayes import BayesAnalyzer


class _ConstantFitter:
    """Fake fitter returning the same density everywhere, for deterministic
    control over the EM update dynamics."""

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


def test_em_loop_converges_within_default_max_iter() -> None:
    trj = _make_trajectory()
    trapped = _ConstantFitter(2.0)
    transition = _ConstantFitter(1.0)

    p_trap, gamma = BayesAnalyzer.analyze(
        trj, transition, trapped, critical_probability=1e-3
    )

    assert 0.0 <= p_trap <= 1.0
    assert gamma.shape == (trj.count_points - 1,)


def test_em_loop_raises_on_non_convergence_instead_of_looping_forever() -> None:
    trj = _make_trajectory()
    trapped = _ConstantFitter(2.0)
    transition = _ConstantFitter(1.0)

    # This pair converges toward p_trap=1 very slowly (see docstring math in
    # bayes.py); a tiny critical_probability combined with a small max_iter
    # deterministically exceeds the budget instead of ever finishing.
    with pytest.raises(RuntimeError, match="did not converge"):
        BayesAnalyzer.analyze(
            trj,
            transition,
            trapped,
            critical_probability=1e-9,
            max_iter=5,
        )


def test_max_iter_is_configurable() -> None:
    trj = _make_trajectory()
    trapped = _ConstantFitter(2.0)
    transition = _ConstantFitter(1.0)

    with pytest.raises(RuntimeError, match=r"5 iterations"):
        BayesAnalyzer.analyze(
            trj,
            transition,
            trapped,
            critical_probability=1e-9,
            max_iter=5,
        )
    with pytest.raises(RuntimeError, match=r"3 iterations"):
        BayesAnalyzer.analyze(
            trj,
            transition,
            trapped,
            critical_probability=1e-9,
            max_iter=3,
        )

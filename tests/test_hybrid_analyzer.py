from typing import Any, Tuple, cast

import numpy as np
import pytest

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from processes.distribution_fitter import GammaFitter, WeibullFitter
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from processes.trajectory_analyzer.hybrid import HybridAnalyzer, HybridParams
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)


def _trajectory(point_count: int) -> Trajectory:
    points = np.zeros((point_count, 3), dtype=np.float32)
    times = np.arange(point_count, dtype=np.float32)
    box = BoundingBox(Range(0, 1), Range(0, 1), Range(0, 1))
    return Trajectory(points, times, box)


def test_hybrid_recomputes_dm_mask_for_each_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dm_results = [
        np.zeros(9, dtype=np.bool_),
        np.ones(10, dtype=np.bool_),
    ]

    def fake_dm_run(self: Any, trj: Any) -> np.ndarray:
        return dm_results.pop(0)

    def fake_sib_analyze(self: Any, trj: Any) -> Tuple[float, np.ndarray]:
        probabilities = np.full(trj.count_points - 1, 0.5, dtype=np.float32)
        return 0.5, probabilities

    monkeypatch.setattr(DistanceMatrixAnalyzer, "run", fake_dm_run)
    monkeypatch.setattr(
        StructureInformedBayesAnalyzer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        StructureInformedBayesAnalyzer,
        "analyze",
        fake_sib_analyze,
    )

    analyzer = HybridAnalyzer(
        HybridParams(
            StructureInformedBayesParams(),
            DistanceMatrixParams(),
        ),
        pi_l_gf=cast(GammaFitter, None),
        throat_lengths_wf=cast(WeibullFitter, None),
    )

    first = analyzer.run(_trajectory(10))
    second = analyzer.run(_trajectory(11))

    np.testing.assert_array_equal(first, np.zeros(9, dtype=np.bool_))
    np.testing.assert_array_equal(second, np.ones(10, dtype=np.bool_))
    assert not dm_results


def test_hybrid_dm_override_is_used_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computed = np.zeros(9, dtype=np.bool_)

    def fake_dm_run(self: Any, trj: Any) -> np.ndarray:
        return computed

    def fake_sib_analyze(self: Any, trj: Any) -> Tuple[float, np.ndarray]:
        probabilities = np.full(trj.count_points - 1, 0.5, dtype=np.float32)
        return 0.5, probabilities

    monkeypatch.setattr(DistanceMatrixAnalyzer, "run", fake_dm_run)
    monkeypatch.setattr(
        StructureInformedBayesAnalyzer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        StructureInformedBayesAnalyzer,
        "analyze",
        fake_sib_analyze,
    )

    analyzer = HybridAnalyzer(
        HybridParams(
            StructureInformedBayesParams(),
            DistanceMatrixParams(),
        ),
        pi_l_gf=cast(GammaFitter, None),
        throat_lengths_wf=cast(WeibullFitter, None),
    )
    analyzer.set_trap_approx(np.ones(9, dtype=np.bool_))

    overridden = analyzer.run(_trajectory(10))
    recomputed = analyzer.run(_trajectory(10))

    np.testing.assert_array_equal(overridden, np.ones(9, dtype=np.bool_))
    np.testing.assert_array_equal(recomputed, np.zeros(9, dtype=np.bool_))

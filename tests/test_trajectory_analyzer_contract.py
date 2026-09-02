import numpy as np
import pytest

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from processes.trajectory_analyzer.dm import DistanceMatrixAnalyzer
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer


def _trajectory(point_count: int) -> Trajectory:
    points = np.zeros((point_count, 3), dtype=np.float32)
    times = np.arange(point_count, dtype=np.float32)
    box = BoundingBox(Range(0, 1), Range(0, 1), Range(0, 1))
    return Trajectory(points, times, box)


def test_trajectory_analyzers_reject_fewer_than_ten_points() -> None:
    with pytest.raises(ValueError, match="requires at least 10.*got 9"):
        TrajectoryAnalyzer.validate_trajectory(_trajectory(9))

    TrajectoryAnalyzer.validate_trajectory(_trajectory(10))


def test_dm_edge_label_uses_the_destination_point() -> None:
    point_labels = np.array([False, False, True, True, False])

    edge_labels = DistanceMatrixAnalyzer.point_labels_to_edge_labels(
        point_labels
    )

    # The edge entering the first trapped point is therefore labeled trapped.
    np.testing.assert_array_equal(edge_labels, [False, True, True, False])


def test_dm_scale_shorter_than_its_threshold_returns_a_complete_result() -> (
    None
):
    analyzer = object.__new__(DistanceMatrixAnalyzer)
    point_count = 10
    threshold = np.full((12, 100), 20, dtype=np.int32)

    flag, invariant, point_labels = analyzer.analyse_by_mu(
        np.zeros((point_count, point_count), dtype=np.float32),
        p_value=0.01,
        nu=0.5,
        mu=0.5,
        list_threshold=threshold,
    )

    assert flag is False
    assert invariant.shape == (point_count,)
    assert point_labels.shape == (point_count,)
    assert point_labels.dtype == np.bool_


def test_dm_keeps_a_valid_scale_without_short_trapped_runs(monkeypatch) -> None:
    analyzer = object.__new__(DistanceMatrixAnalyzer)
    analyzer.diag_fill_list = [0] * 12
    point_count = 10
    threshold = np.full((12, 100), 2, dtype=np.int32)

    def all_points_form_one_valid_trapped_run(*args, **kwargs):
        del args, kwargs
        return (
            np.full(point_count, 2, dtype=np.int32),
            np.ones(point_count, dtype=np.int32),
            np.ones(point_count, dtype=np.int32),
        )

    monkeypatch.setattr(
        analyzer,
        "RQA_block_measures",
        all_points_form_one_valid_trapped_run,
    )

    flag, invariant, point_labels = analyzer.analyse_by_mu(
        np.zeros((point_count, point_count), dtype=np.float32),
        p_value=0.01,
        nu=0.5,
        mu=0.5,
        list_threshold=threshold,
    )

    assert flag is True
    np.testing.assert_array_equal(invariant, np.full(point_count, 2.0))
    np.testing.assert_array_equal(
        point_labels, np.ones(point_count, dtype=np.bool_)
    )

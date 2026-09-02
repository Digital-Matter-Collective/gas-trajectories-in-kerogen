import io

import numpy as np
import pytest

from base.boundingbox import BoundingBox, Range
from base.reader import Reader, StepsInfo
from base.trajectory import Trajectory
from utils.timer import Timer, TimerError


def _make_trajectory(n: int = 5) -> Trajectory:
    points = np.cumsum(
        np.ones((n, 3), dtype=np.float32) * 0.1, axis=0
    ).astype(np.float32)
    times = np.arange(n, dtype=np.float32)
    box = BoundingBox(Range(0, 100), Range(0, 100), Range(0, 100))
    return Trajectory(points=points, times=times, box=box)


# --- Trajectory.msd() ------------------------------------------------------


def test_msd_returns_an_array_not_none() -> None:
    trj = _make_trajectory()
    result = trj.msd()
    assert result is not None
    assert len(result) == trj.count_points
    assert result[0] == pytest.approx(0.0)


# --- Trajectory.is_intersect_borders() / Writer -----------------------------


def test_is_intersect_borders_does_not_raise_and_returns_bool() -> None:
    trj = _make_trajectory()
    result = trj.is_intersect_borders()
    assert result in (True, False, np.True_, np.False_)


def test_writer_reads_points_without_periodic_as_a_property(tmp_path) -> None:
    from base.writer import Writer

    trj = _make_trajectory()
    cwd = tmp_path
    import os

    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        Writer.trajectory_to_mat([trj])
        assert (cwd / "trajectories.mat").exists()
    finally:
        os.chdir(old_cwd)


# --- Range.__str__() ---------------------------------------------------------


def test_range_str_does_not_raise_invalid_format_specifier() -> None:
    r = Range(1.23456, 7.891011)
    assert str(r) == "Range(min=1.235_max=7.891)"


# --- Timer -------------------------------------------------------------------


def test_timer_double_start_raises_timer_error_not_typeerror() -> None:
    timer = Timer(logger=None)
    timer.start()
    with pytest.raises(TimerError, match="already running"):
        timer.start()


def test_timer_stop_without_start_raises_timer_error_not_typeerror() -> None:
    timer = Timer(logger=None)
    with pytest.raises(TimerError, match="not running"):
        timer.stop()


# --- Trajectory.cut() cache invalidation -------------------------------------


def test_cut_invalidates_cached_count_points() -> None:
    trj = _make_trajectory(n=10)
    assert trj.count_points == 10  # populate the cache

    trj.cut(start=0, stop=4)

    assert trj.count_points == 4
    assert len(trj.points) == 4


def test_cut_invalidates_cached_points_without_periodic() -> None:
    trj = _make_trajectory(n=10)
    _ = trj.points_without_periodic  # populate the cache

    trj.cut(start=2, stop=6)

    assert trj.points_without_periodic.shape[0] == 4
    np.testing.assert_array_equal(trj.points_without_periodic[0], trj.points[0])


def test_cut_invalidates_cached_delta_time() -> None:
    trj = _make_trajectory(n=10)
    trj.times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 20], dtype=np.float32)
    assert trj.delta_time == pytest.approx(1.0)  # populate the cache

    trj.cut(start=8, stop=10)  # keeps the [8, 20] pair -> delta 12

    assert trj.delta_time == pytest.approx(12.0)


# --- Reader.read_head_struct mutable default --------------------------------


def _make_frame(step: int, t: float, count: int = 1) -> bytes:
    header = f"Kerogen t= {t:.5f} step= {step}\n".encode()
    return header + f"{count}\n".encode()


def test_read_head_struct_default_info_does_not_leak_across_calls() -> None:
    frame_a = io.StringIO(f"Kerogen t= 10.00000 step= 100\n1\n")
    frame_b = io.StringIO(f"Kerogen t= 20.00000 step= 200\n1\n")

    num_a, time_a = Reader.read_head_struct(frame_a)
    num_b, time_b = Reader.read_head_struct(frame_b)

    assert (num_a, time_a) == (100, 10.0)
    assert (num_b, time_b) == (200, 20.0)


def test_read_head_struct_preserves_fractional_time() -> None:
    frame = io.StringIO("Kerogen t= 12.34500 step= 5\n1\n")
    _, t = Reader.read_head_struct(frame)
    assert t == pytest.approx(12.345)


def test_read_head_struct_explicit_info_still_accumulates() -> None:
    info = StepsInfo()
    frame_a = io.StringIO("Kerogen t= 1.00000 step= 1\n1\n")
    frame_b = io.StringIO("Kerogen t= 2.00000 step= 2\n1\n")

    Reader.read_head_struct(frame_a, info)
    Reader.read_head_struct(frame_b, info)

    assert info.steps == [1, 2]
    assert info.times == [1.0, 2.0]

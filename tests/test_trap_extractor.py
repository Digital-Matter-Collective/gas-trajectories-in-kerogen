import numpy as np
import pytest

from processes.trap_extractor import TrapExtractor


def _k_est(times: np.ndarray) -> float:
    n0 = int(np.count_nonzero(times == 0))
    nt = int(np.count_nonzero(times > 0))
    return nt / (n0 + nt)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        TrapExtractor.get_trap_seq(np.array([], dtype=bool), delta_time=1.0)


def test_all_free_trajectory_is_one_zero_event() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([False, False, False]), delta_time=1.0
    )
    assert list(seq.times) == [0.0]
    assert list(seq.traps) == [3]
    assert seq.get_zero_trap_count() == 1
    assert seq.get_non_zero_trap_count() == 0
    assert _k_est(seq.times) == pytest.approx(0.0)


def test_all_trapped_trajectory_has_no_spurious_zero_event() -> None:
    # P1-08 regression: an unconditional artificial (0.0, 1) boundary entry
    # used to make an all-trapped trajectory report k_est = 0.5 instead of 1.0.
    seq = TrapExtractor.get_trap_seq(np.array([True, True, True]), delta_time=1.0)
    assert list(seq.times) == [3.0]
    assert list(seq.traps) == [4]
    assert seq.get_zero_trap_count() == 0
    assert seq.get_non_zero_trap_count() == 1
    assert _k_est(seq.times) == pytest.approx(1.0)


def test_free_run_collapses_to_a_single_event_not_one_per_step() -> None:
    # P1-08 regression: previously every free-free edge pair produced its
    # own (0.0, 1) entry, so N0 scaled with free-run length instead of the
    # number of free episodes.
    seq = TrapExtractor.get_trap_seq(
        np.array([False, False, False, False, False]), delta_time=1.0
    )
    assert list(seq.times) == [0.0]
    assert seq.get_zero_trap_count() == 1


def test_alternating_runs_produce_one_entry_each() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([True, False, True]), delta_time=1.0
    )
    assert list(seq.times) == [1.0, 0.0, 1.0]
    assert list(seq.traps) == [2, 1, 2]
    assert seq.get_zero_trap_count() == 1
    assert seq.get_non_zero_trap_count() == 2
    assert _k_est(seq.times) == pytest.approx(2 / 3)


def test_single_trapped_edge_followed_by_free_run_is_not_misclassified() -> None:
    # P1-08 regression: the original state machine only advanced cur_time
    # when edge_traps[i] was True for i >= 1, so a trap run consisting only
    # of edge 0 recorded duration 0.0 and was miscounted as a free event.
    seq = TrapExtractor.get_trap_seq(
        np.array([True, False, False]), delta_time=1.0
    )
    assert list(seq.times) == [1.0, 0.0]
    assert seq.get_non_zero_trap_count() == 1
    assert seq.get_zero_trap_count() == 1


def test_trapped_run_at_the_end_is_included() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([False, True, True]), delta_time=2.0
    )
    assert list(seq.times) == [0.0, 4.0]
    assert list(seq.traps) == [1, 3]


def test_delta_time_scales_trapped_duration() -> None:
    seq = TrapExtractor.get_trap_seq(np.array([True, True]), delta_time=0.5)
    assert seq.times[0] == pytest.approx(1.0)

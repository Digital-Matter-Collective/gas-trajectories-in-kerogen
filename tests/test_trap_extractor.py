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


def test_all_free_trajectory_contains_only_fully_observed_bypasses() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([False, False, False]), delta_time=1.0
    )
    assert list(seq.times) == [0.0, 0.0]
    assert list(seq.traps) == [1, 1]
    assert seq.get_zero_trap_count() == 2
    assert seq.get_non_zero_trap_count() == 0
    assert _k_est(seq.times) == pytest.approx(0.0)


def test_single_free_boundary_edge_contains_no_complete_visit() -> None:
    seq = TrapExtractor.get_trap_seq(np.array([False]), delta_time=1.0)
    assert len(seq.times) == 0
    assert len(seq.traps) == 0


def test_all_trapped_trajectory_has_no_spurious_zero_event() -> None:
    # P1-08 regression: an unconditional artificial (0.0, 1) boundary entry
    # used to make an all-trapped trajectory report k_est = 0.5 instead of 1.0.
    seq = TrapExtractor.get_trap_seq(
        np.array([True, True, True]), delta_time=1.0
    )
    assert list(seq.times) == [3.0]
    assert list(seq.traps) == [4]
    assert seq.get_zero_trap_count() == 0
    assert seq.get_non_zero_trap_count() == 1
    assert _k_est(seq.times) == pytest.approx(1.0)


def test_free_run_encodes_each_observed_intermediate_bypass() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([False, False, False, False, False]), delta_time=1.0
    )
    assert list(seq.times) == [0.0, 0.0, 0.0, 0.0]
    assert seq.get_zero_trap_count() == 4


def test_single_transition_between_captures_is_not_a_bypass() -> None:
    seq = TrapExtractor.get_trap_seq(
        np.array([True, False, True]), delta_time=1.0
    )
    assert list(seq.times) == [1.0, 1.0]
    assert list(seq.traps) == [2, 2]
    assert seq.get_zero_trap_count() == 0
    assert seq.get_non_zero_trap_count() == 2
    assert _k_est(seq.times) == pytest.approx(1.0)


def test_single_trapped_edge_followed_by_free_run_is_not_misclassified() -> (
    None
):
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
    assert list(seq.times) == [4.0]
    assert list(seq.traps) == [3]


def test_mixed_runs_preserve_event_order_and_ignore_censored_boundaries() -> (
    None
):
    seq = TrapExtractor.get_trap_seq(
        np.array([False, False, True, True, False, False, False]),
        delta_time=2.0,
    )
    assert list(seq.times) == [0.0, 4.0, 0.0, 0.0]
    assert list(seq.traps) == [1, 3, 1, 1]
    assert _k_est(seq.times) == pytest.approx(1 / 4)


def test_delta_time_scales_trapped_duration() -> None:
    seq = TrapExtractor.get_trap_seq(np.array([True, True]), delta_time=0.5)
    assert seq.times[0] == pytest.approx(1.0)

import pickle
from pathlib import Path

import numpy as np
import pytest

from base.trap_sequence import TrapSequence
from scripts.sim_algo_check import (
    ERROR_CORRIDOR_QUANTILES,
    ERROR_SERIES_COLORS,
    FIGURE_8_ANALYZERS,
    FIGURE_13_ANALYZERS,
    _atomic_pickle_dump,
    _benchmark_manifest,
    _checkpoint_metadata,
    _empty_checkpoint,
    _error_band,
    _figure_error_data,
    _load_or_initialize_checkpoint,
    _save_analyzer_checkpoint,
    _shared_error_axis_limits,
    estimate_trapping_probability,
    trajectory_seed,
)


def test_k_est_is_capture_events_divided_by_all_events() -> None:
    sequence = TrapSequence(
        traps=np.ones(5, dtype=np.int32),
        times=np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
    )

    assert estimate_trapping_probability(sequence) == pytest.approx(3 / 5)


def test_trajectory_seeds_are_stable_and_independent() -> None:
    seeds = {
        trajectory_seed(42, k_index, p_index, trajectory_index)
        for k_index in range(3)
        for p_index in range(2)
        for trajectory_index in range(5)
    }

    assert len(seeds) == 30
    assert trajectory_seed(42, 1, 2, 3) == trajectory_seed(42, 1, 2, 3)
    assert trajectory_seed(43, 1, 2, 3) != trajectory_seed(42, 1, 2, 3)


def test_atomic_pickle_uses_a_separate_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pkl"

    _atomic_pickle_dump({"value": 1}, target)
    _atomic_pickle_dump({"value": 2}, target)

    with target.open("rb") as file:
        assert pickle.load(file) == {"value": 2}
    assert not (tmp_path / ".checkpoint.pkl.tmp").exists()


def test_checkpoint_resumes_from_the_next_unprocessed_trajectory(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pkl"
    result_shape = (2, 10)
    step_count = 9
    prob_grid = np.array([0.0, 0.5])
    metadata = _checkpoint_metadata(
        analyzer_name="dm",
        k=0.1,
        prob_grid=prob_grid,
        trajectory_count=10,
        step_count=step_count,
        seed=42,
    )
    state = _empty_checkpoint(result_shape, step_count)
    state["k_est"].reshape(-1)[:10] = 0.5
    state["errors"].reshape(-1)[:10] = 0.1
    state["results"].reshape(-1, step_count)[:10] = 0
    state["next_flat_index"] = 10
    _save_analyzer_checkpoint(checkpoint_path, metadata, state)

    restored = _load_or_initialize_checkpoint(
        checkpoint_path,
        metadata,
        result_shape,
        step_count,
        force_recompute=False,
    )

    assert restored["next_flat_index"] == 10
    assert np.all(restored["results"].reshape(-1, step_count)[:10] == 0)


def test_seed_and_figure_contract_are_saved_without_code_fingerprint() -> None:
    manifest = _benchmark_manifest(
        seed=42,
        trajectory_count=100,
        step_count=3000,
        prob_grid=np.array([0.0, 0.5, 1.0]),
    )

    assert manifest["seed"] == 42
    assert manifest["k_est_definition"] == "N_t / (N_0 + N_t)"
    assert tuple(manifest["figure_8"]["algorithms"]) == FIGURE_8_ANALYZERS
    assert tuple(manifest["figure_13"]["algorithms"]) == FIGURE_13_ANALYZERS
    assert manifest["figure_8"]["corridor"] == "20th-80th percentile"
    assert manifest["figure_13"]["corridor"] == "20th-80th percentile"
    assert "fingerprint" not in manifest
    assert "version" not in manifest


def test_figures_share_dm_and_sib_data_statistics_and_colors() -> None:
    dm = (np.arange(12, dtype=float).reshape(3, 4), 0.4)
    np_summary = (np.ones((3, 4), dtype=float), 0.5)
    sib = (np.full((3, 4), 2.0), 0.6)
    hybrid = (np.full((3, 4), 3.0), 0.7)

    figure_8, figure_13 = _figure_error_data(
        dm,
        np_summary,
        sib,
        hybrid,
    )

    assert figure_8["DM"] is figure_13["DM"]
    assert figure_8["SIB"] is figure_13["NP + Bayesian (SIB)"]
    assert ERROR_CORRIDOR_QUANTILES == (0.2, 0.8)
    assert set(figure_8) | set(figure_13) <= set(ERROR_SERIES_COLORS)
    assert (
        ERROR_SERIES_COLORS["SIB"]
        == ERROR_SERIES_COLORS["NP + Bayesian (SIB)"]
    )

    q_low, q_high = ERROR_CORRIDOR_QUANTILES
    np.testing.assert_array_equal(
        _error_band(
            figure_8["DM"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
        _error_band(
            figure_13["DM"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
    )
    np.testing.assert_array_equal(
        _error_band(
            figure_8["SIB"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
        _error_band(
            figure_13["NP + Bayesian (SIB)"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
    )


def test_shared_error_axis_includes_every_displayed_series() -> None:
    summaries = {
        "DM": (np.array([[0.1, 0.2, 0.3]]), 0.1),
        "SIB": (np.array([[0.0, 0.1, 0.2]]), 0.2),
        "HYB": (np.array([[0.2, 0.3, 0.4]]), 0.3),
        "NP": (np.array([[0.5, 0.6, 0.7]]), 0.4),
    }
    q_low, q_high = ERROR_CORRIDOR_QUANTILES

    lower, upper = _shared_error_axis_limits(
        summaries,
        q_low=q_low,
        q_high=q_high,
        center="mean",
    )

    assert lower == 0.0
    assert upper > 0.66

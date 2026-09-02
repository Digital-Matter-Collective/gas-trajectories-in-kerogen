import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from processes.trajectory_analyzer.dm import DistanceMatrixParams
from scripts.dm_parameter_search import (
    CANDIDATE_GRID,
    CANDIDATE_SHAPE,
    K_VALUES,
    P_VALUES,
    SCALE_SETS,
    expected_result_metadata,
)
from scripts.errors_params import (
    DEFAULT_TRAJECTORY_COUNT,
    _completed_scale_indices,
    _load_cached_result,
    _save_checkpoint,
)
from scripts.find_best_params import (
    LoadedPairErrors,
    load_pair_errors,
    save_table_i_summary,
    select_optimal_parameters,
)


def _loaded(errors: np.ndarray) -> LoadedPairErrors:
    return LoadedPairErrors(
        errors=errors,
        source_format="schema_1_pickle",
        error_metric="mean_relative_classification_error",
        trajectory_count=100,
        trajectory_points=1000,
        base_seed=42,
    )


def test_shared_grid_contains_only_the_twelve_article_scale_sets() -> None:
    assert DEFAULT_TRAJECTORY_COUNT == 100
    assert CANDIDATE_SHAPE == (12, 216)
    assert tuple(row[0].scale_set for row in CANDIDATE_GRID) == SCALE_SETS
    assert (
        len(
            {
                candidate.candidate_id
                for row in CANDIDATE_GRID
                for candidate in row
            }
        )
        == 12 * 216
    )

    for scale_set, candidates in zip(SCALE_SETS, CANDIDATE_GRID):
        original = DistanceMatrixParams.get_params(lmu=scale_set)
        assert len(original) == len(candidates)
        for old, candidate in zip(original, candidates):
            assert (
                old.traj_type,
                old.nu,
                old.diag_percentile,
                old.kernel_size,
                tuple(old.list_mu),
                old.p_value,
            ) == (
                candidate.reference_motion,
                candidate.invariant_threshold,
                candidate.diagonal_percentile,
                candidate.kernel_radius,
                candidate.scale_set,
                candidate.p_value_threshold,
            )


def test_selection_averages_all_p_values_and_exports_table_i(
    tmp_path: Path,
) -> None:
    expected_indices = ((2, 9), (5, 57), (0, 56))
    pair_errors = {}
    for k_index, k in enumerate(K_VALUES):
        for p_index, p in enumerate(P_VALUES):
            errors = np.ones(CANDIDATE_SHAPE, dtype=np.float64)
            errors[expected_indices[k_index]] = 0.2
            if k_index == 0:
                # This distractor wins five individual p values but loses after
                # aggregation over the complete six-value article grid.
                errors[0, 0] = 0.19 if p_index < 5 else 1.0
            pair_errors[(k, p)] = _loaded(errors)

    rows = select_optimal_parameters(pair_errors)
    csv_path, json_path = save_table_i_summary(rows, tmp_path)
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert [row.candidate_id for row in rows] == [
        "scale-02-param-009",
        "scale-05-param-057",
        "scale-00-param-056",
    ]
    assert [row.smoothing_window_size for row in rows] == [3, 3, 1]
    assert [row.diagonal_fill_width for row in rows] == [0, 0, 0]
    assert [row.scale_set for row in rows] == [
        (0.5, 1.0),
        (1.5, 2.0),
        (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    ]
    assert [row.block_invariant_threshold_v_c for row in rows] == [
        0.5,
        0.5,
        0.5,
    ]
    assert [row.minimum_run_threshold_p_val for row in rows] == [
        0.01,
        0.9,
        0.9,
    ]
    assert [row.reference_motion for row in rows] == ["fBm", "fBm", "fBm"]
    assert json_rows[0]["trajectory_count_per_k_p"] == 100
    assert csv_path.is_file()


def test_legacy_loader_rejects_an_uncomputed_candidate(tmp_path: Path) -> None:
    payload = {
        (0, 0, scale_index, parameter_index): 1.0
        for scale_index in range(CANDIDATE_SHAPE[0])
        for parameter_index in range(CANDIDATE_SHAPE[1])
    }
    result_path = tmp_path / "k=0.1_p=0.0.pickle"
    with result_path.open("wb") as file:
        pickle.dump(payload, file)

    loaded = load_pair_errors(
        result_path,
        k=0.1,
        p=0.0,
        k_index=0,
        p_index=0,
    )
    assert loaded.errors.shape == CANDIDATE_SHAPE

    payload.pop((0, 0, CANDIDATE_SHAPE[0] - 1, CANDIDATE_SHAPE[1] - 1))
    with result_path.open("wb") as file:
        pickle.dump(payload, file)
    with pytest.raises(ValueError, match="missing 1 candidates"):
        load_pair_errors(
            result_path,
            k=0.1,
            p=0.0,
            k_index=0,
            p_index=0,
        )


def test_new_result_preserves_publication_metadata(tmp_path: Path) -> None:
    result_path = tmp_path / "k=0.1_p=0.0.pickle"
    metadata = expected_result_metadata(
        k=0.1,
        p=0.0,
        k_index=0,
        p_index=0,
        trajectory_count=100,
        trajectory_points=1000,
        seed=42,
    )
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "metadata": metadata,
                "mean_relative_errors": np.ones(CANDIDATE_SHAPE),
            },
            file,
        )

    loaded = load_pair_errors(
        result_path,
        k=0.1,
        p=0.0,
        k_index=0,
        p_index=0,
    )

    assert loaded.error_metric == "mean_relative_classification_error"
    assert loaded.trajectory_count == 100
    assert loaded.trajectory_points == 1000
    assert loaded.base_seed == 42


def test_scale_checkpoint_round_trip(tmp_path: Path) -> None:
    result_path = tmp_path / "k=0.1_p=0.0.pickle"
    metadata = expected_result_metadata(
        k=0.1,
        p=0.0,
        k_index=0,
        p_index=0,
        trajectory_count=100,
        trajectory_points=1000,
        seed=42,
    )
    errors = np.full(CANDIDATE_SHAPE, np.nan, dtype=np.float64)
    errors[0] = 0.25

    _save_checkpoint(result_path, metadata, errors)
    restored = _load_cached_result(
        result_path,
        metadata,
        allow_legacy=False,
    )

    assert restored is not None
    assert _completed_scale_indices(restored) == (0,)
    np.testing.assert_array_equal(restored[0], errors[0])
    assert np.all(np.isnan(restored[1:]))
    assert not result_path.with_suffix(".pickle.tmp").exists()


def test_resume_replaces_legacy_results_without_deleting_them_first(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "k=0.1_p=0.0.pickle"
    with result_path.open("wb") as file:
        pickle.dump({(0, 0, 0, 0): 1.0}, file)
    metadata = expected_result_metadata(
        k=0.1,
        p=0.0,
        k_index=0,
        p_index=0,
        trajectory_count=100,
        trajectory_points=1000,
        seed=42,
    )

    assert _load_cached_result(result_path, metadata, allow_legacy=True) is None
    assert result_path.is_file()
    with pytest.raises(RuntimeError, match="Legacy result"):
        _load_cached_result(result_path, metadata, allow_legacy=False)


def test_checkpoint_rejects_a_partially_filled_scale_row() -> None:
    errors = np.full(CANDIDATE_SHAPE, np.nan, dtype=np.float64)
    errors[0, 0] = 0.25

    with pytest.raises(RuntimeError, match="partially written scale row: 0"):
        _completed_scale_indices(errors)

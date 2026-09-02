import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

import numpy as np

from processes.trajectory_analyzer.dm import list_vert_median
from scripts.dm_parameter_search import (
    CANDIDATE_GRID,
    CANDIDATE_SHAPE,
    K_VALUES,
    P_VALUES,
    SEARCH_SCHEMA_VERSION,
    DMCandidate,
    grid_fingerprint,
    result_file_name,
    validate_error_matrix_shape,
)
from utils.utils import kprint


@dataclass(frozen=True)
class LoadedPairErrors:
    errors: np.ndarray
    source_format: str
    error_metric: str
    trajectory_count: int | None
    trajectory_points: int | None
    base_seed: int | None


@dataclass(frozen=True)
class TableIRow:
    k: float
    candidate_id: str
    scale_set: tuple[float, ...]
    smoothing_window_size: int
    diagonal_fill_width: int | None
    diagonal_fill_widths: tuple[int, ...]
    diagonal_percentile: int
    block_invariant_threshold_v_c: float
    minimum_run_threshold_p_val: float
    reference_motion: str
    aggregate_mean_error: float
    error_metric: str
    p_values: tuple[float, ...]
    trajectory_count_per_k_p: int | None
    trajectory_points: int | None
    base_seed: int | None
    source_format: str


def _load_legacy_errors(
    payload: dict,
    *,
    expected_k_index: int,
    expected_p_index: int,
    path: Path,
) -> LoadedPairErrors:
    errors = np.full(CANDIDATE_SHAPE, np.nan, dtype=np.float64)
    for key, value in payload.items():
        if not isinstance(key, tuple) or len(key) != 4:
            raise ValueError(f"Unexpected legacy key {key!r} in {path}")
        k_index, p_index, scale_index, parameter_index = key
        if k_index != expected_k_index or p_index != expected_p_index:
            raise ValueError(
                f"Legacy key {key!r} does not match its file {path}"
            )
        if not (
            0 <= scale_index < CANDIDATE_SHAPE[0]
            and 0 <= parameter_index < CANDIDATE_SHAPE[1]
        ):
            raise ValueError(
                f"Candidate index outside the shared grid: {key!r}"
            )
        if np.isfinite(errors[scale_index, parameter_index]):
            raise ValueError(f"Duplicate legacy candidate key: {key!r}")
        errors[scale_index, parameter_index] = float(value)

    if not np.all(np.isfinite(errors)):
        missing = int(np.count_nonzero(~np.isfinite(errors)))
        raise ValueError(
            f"Legacy result {path} is missing {missing} candidates"
        )
    return LoadedPairErrors(
        errors=errors,
        source_format="legacy_pickle_without_metadata",
        error_metric="mean_misclassified_steps",
        trajectory_count=None,
        trajectory_points=None,
        base_seed=None,
    )


def load_pair_errors(
    path: Path,
    *,
    k: float,
    p: float,
    k_index: int,
    p_index: int,
) -> LoadedPairErrors:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary in {path}")

    if "metadata" not in payload:
        return _load_legacy_errors(
            payload,
            expected_k_index=k_index,
            expected_p_index=p_index,
            path=path,
        )

    metadata = payload["metadata"]
    expected_values = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "grid_fingerprint": grid_fingerprint(),
        "k": k,
        "p": p,
        "k_index": k_index,
        "p_index": p_index,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_values.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Metadata mismatch in {path}: {mismatches}")

    if "mean_relative_errors" not in payload:
        raise ValueError(f"Result has no error matrix: {path}")
    errors = np.asarray(payload["mean_relative_errors"], dtype=np.float64)
    validate_error_matrix_shape(errors)
    if not np.all(np.isfinite(errors)):
        raise ValueError(f"Result contains uncomputed candidates: {path}")
    return LoadedPairErrors(
        errors=errors,
        source_format=f"schema_{SEARCH_SCHEMA_VERSION}_pickle",
        error_metric=str(metadata["error_metric"]),
        trajectory_count=int(metadata["trajectory_count"]),
        trajectory_points=int(metadata["trajectory_points"]),
        base_seed=int(metadata["base_seed"]),
    )


def load_all_pair_errors(
    path: Path,
) -> dict[tuple[float, float], LoadedPairErrors]:
    loaded = {}
    for k_index, k in enumerate(K_VALUES):
        for p_index, p in enumerate(P_VALUES):
            result_path = path / result_file_name(k, p)
            if not result_path.is_file():
                raise FileNotFoundError(
                    f"Missing parameter-search result: {result_path}"
                )
            loaded[(k, p)] = load_pair_errors(
                result_path,
                k=k,
                p=p,
                k_index=k_index,
                p_index=p_index,
            )
    return loaded


def _common_optional_value(values: list[int | None], name: str) -> int | None:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"Inconsistent {name} across p values: {unique}")
    return values[0]


def _candidate_to_table_row(
    *,
    k: float,
    candidate: DMCandidate,
    aggregate_mean_error: float,
    pair_results: list[LoadedPairErrors],
) -> TableIRow:
    error_metrics = {result.error_metric for result in pair_results}
    source_formats = {result.source_format for result in pair_results}
    if len(error_metrics) != 1 or len(source_formats) != 1:
        raise ValueError("Cannot aggregate mixed parameter-search formats")

    diagonal_fill_widths = tuple(
        int(
            list_vert_median[
                (candidate.diagonal_percentile, candidate.reference_motion)
            ][int(2 * scale) - 1]
        )
        for scale in candidate.scale_set
    )
    unique_fill_widths = set(diagonal_fill_widths)
    diagonal_fill_width = (
        diagonal_fill_widths[0] if len(unique_fill_widths) == 1 else None
    )
    return TableIRow(
        k=k,
        candidate_id=candidate.candidate_id,
        scale_set=candidate.scale_set,
        smoothing_window_size=2 * candidate.kernel_radius + 1,
        diagonal_fill_width=diagonal_fill_width,
        diagonal_fill_widths=diagonal_fill_widths,
        diagonal_percentile=candidate.diagonal_percentile,
        block_invariant_threshold_v_c=candidate.invariant_threshold,
        minimum_run_threshold_p_val=candidate.p_value_threshold,
        reference_motion=candidate.reference_motion,
        aggregate_mean_error=aggregate_mean_error,
        error_metric=next(iter(error_metrics)),
        p_values=P_VALUES,
        trajectory_count_per_k_p=_common_optional_value(
            [result.trajectory_count for result in pair_results],
            "trajectory_count",
        ),
        trajectory_points=_common_optional_value(
            [result.trajectory_points for result in pair_results],
            "trajectory_points",
        ),
        base_seed=_common_optional_value(
            [result.base_seed for result in pair_results],
            "base_seed",
        ),
        source_format=next(iter(source_formats)),
    )


def select_optimal_parameters(
    pair_errors: Mapping[tuple[float, float], LoadedPairErrors],
) -> list[TableIRow]:
    """Average over all six p values and select only computed candidates."""
    rows = []
    for k in K_VALUES:
        pair_results = [pair_errors[(k, p)] for p in P_VALUES]
        stacked = np.stack([result.errors for result in pair_results], axis=0)
        if stacked.shape[1:] != CANDIDATE_SHAPE:
            raise ValueError(
                f"Expected candidate dimensions {CANDIDATE_SHAPE}, "
                f"got {stacked.shape[1:]}"
            )
        if not np.all(np.isfinite(stacked)):
            raise ValueError(
                f"Cannot select k={k}: some candidates are uncomputed"
            )

        aggregate = np.mean(stacked, axis=0)
        scale_index, parameter_index = np.unravel_index(
            int(np.argmin(aggregate)), aggregate.shape
        )
        candidate = CANDIDATE_GRID[scale_index][parameter_index]
        rows.append(
            _candidate_to_table_row(
                k=k,
                candidate=candidate,
                aggregate_mean_error=float(
                    aggregate[scale_index, parameter_index]
                ),
                pair_results=pair_results,
            )
        )
    return rows


def save_table_i_summary(
    rows: list[TableIRow], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_i_optimized_dm_params.csv"
    json_path = output_dir / "table_i_optimized_dm_params.json"
    json_rows = [asdict(row) for row in rows]

    csv_rows = []
    for row in json_rows:
        csv_row = row.copy()
        for key in ("scale_set", "diagonal_fill_widths", "p_values"):
            csv_row[key] = json.dumps(csv_row[key], separators=(",", ":"))
        csv_rows.append(csv_row)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[field.name for field in fields(TableIRow)],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_rows, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return csv_path, json_path


def run(path: str | Path, output_dir: Path | None = None) -> list[TableIRow]:
    input_dir = Path(path)
    pair_errors = load_all_pair_errors(input_dir)
    rows = select_optimal_parameters(pair_errors)
    if any(row.source_format.startswith("legacy") for row in rows):
        kprint(
            "WARNING: legacy results have no trajectory-count/seed metadata; "
            "rerun scripts.errors_params for a fully reproducible Table I."
        )

    for row in rows:
        kprint(
            f"k={row.k}: scales={row.scale_set}, "
            f"window={row.smoothing_window_size}, "
            f"s={row.diagonal_fill_width}, "
            f"v_c={row.block_invariant_threshold_v_c}, "
            f"p_val={row.minimum_run_threshold_p_val}, "
            f"motion={row.reference_motion}, "
            f"mean_error={row.aggregate_mean_error:.6g}"
        )

    csv_path, json_path = save_table_i_summary(rows, output_dir or input_dir)
    kprint(f"Saved Table I summary: {csv_path}")
    kprint(f"Saved Table I summary: {json_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate the Table I DM parameter search"
    )
    parser.add_argument(
        "path", type=Path, help="Directory with find_best_params data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Table I CSV/JSON directory (default: input directory)",
    )
    args = parser.parse_args()

    run(args.path, args.output_dir)

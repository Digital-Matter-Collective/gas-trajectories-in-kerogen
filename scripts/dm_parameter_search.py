"""Shared, deterministic parameter grid for the Table I DM search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from processes.trajectory_analyzer.dm import DistanceMatrixParams

SEARCH_SCHEMA_VERSION = 1
K_VALUES = (0.1, 0.5, 0.9)
P_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SCALE_SETS = (
    (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    (1.5, 2.0, 2.5),
    (0.5, 1.0),
    (2.5, 3.0),
    (0.5, 3.0),
    (1.5, 2.0),
    (0.5,),
    (1.0,),
    (1.5,),
    (2.0,),
    (2.5,),
    (3.0,),
)

DIAGONAL_PERCENTILES = (0, 10, 50)
P_VALUE_THRESHOLDS = (0.01, 0.1, 0.9)
INVARIANT_THRESHOLDS = (0.1, 0.5, 0.9)
REFERENCE_MOTIONS = ("fBm", "Bm")
KERNEL_RADII = (0, 1, 2, 3)


@dataclass(frozen=True)
class DMCandidate:
    candidate_id: str
    scale_set_index: int
    parameter_index: int
    scale_set: tuple[float, ...]
    reference_motion: str
    invariant_threshold: float
    diagonal_percentile: int
    kernel_radius: int
    p_value_threshold: float

    def to_params(self) -> DistanceMatrixParams:
        return DistanceMatrixParams(
            traj_type=self.reference_motion,
            nu=self.invariant_threshold,
            diag_percentile=self.diagonal_percentile,
            kernel_size=self.kernel_radius,
            list_mu=np.array(self.scale_set),
            p_value=self.p_value_threshold,
        )


def build_candidate_grid() -> tuple[tuple[DMCandidate, ...], ...]:
    """Return the 12 × 216 candidate grid in a stable order."""
    scale_grids = []
    for scale_index, scale_set in enumerate(SCALE_SETS):
        candidates = []
        parameter_index = 0
        for diagonal_percentile in DIAGONAL_PERCENTILES:
            for p_value_threshold in P_VALUE_THRESHOLDS:
                for invariant_threshold in INVARIANT_THRESHOLDS:
                    for reference_motion in REFERENCE_MOTIONS:
                        for kernel_radius in KERNEL_RADII:
                            candidates.append(
                                DMCandidate(
                                    candidate_id=(
                                        f"scale-{scale_index:02d}-"
                                        f"param-{parameter_index:03d}"
                                    ),
                                    scale_set_index=scale_index,
                                    parameter_index=parameter_index,
                                    scale_set=scale_set,
                                    reference_motion=reference_motion,
                                    invariant_threshold=invariant_threshold,
                                    diagonal_percentile=diagonal_percentile,
                                    kernel_radius=kernel_radius,
                                    p_value_threshold=p_value_threshold,
                                )
                            )
                            parameter_index += 1
        scale_grids.append(tuple(candidates))
    return tuple(scale_grids)


CANDIDATE_GRID = build_candidate_grid()
CANDIDATE_SHAPE = (len(CANDIDATE_GRID), len(CANDIDATE_GRID[0]))
TABLE_I_CANDIDATE_INDICES = {
    0.1: (2, 33),
    0.5: (1, 57),
    0.9: (0, 52),
}


def table_i_candidate_for_k(k: float) -> DMCandidate:
    """Return the optimized DM/SIB structural profile reported in Table I."""
    try:
        scale_index, parameter_index = TABLE_I_CANDIDATE_INDICES[k]
    except KeyError as error:
        raise ValueError(f"No Table I parameter profile for k={k}") from error
    return CANDIDATE_GRID[scale_index][parameter_index]


def grid_fingerprint() -> str:
    serialized = json.dumps(
        [[asdict(candidate) for candidate in row] for row in CANDIDATE_GRID],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_search_manifest(
    *,
    trajectory_count: int,
    trajectory_points: int,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "grid_fingerprint": grid_fingerprint(),
        "k_values": list(K_VALUES),
        "p_values": list(P_VALUES),
        "trajectory_count": trajectory_count,
        "trajectory_points": trajectory_points,
        "base_seed": seed,
        "error_metric": "mean_relative_classification_error",
        "scale_sets": [list(scale_set) for scale_set in SCALE_SETS],
        "candidates": [
            asdict(candidate)
            for scale_candidates in CANDIDATE_GRID
            for candidate in scale_candidates
        ],
    }


def save_search_manifest(manifest: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "parameter_search_manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def result_file_name(k: float, p: float) -> str:
    return f"k={k}_p={p}.pickle"


def pair_seed(base_seed: int, k_index: int, p_index: int) -> int:
    return base_seed + k_index * len(P_VALUES) + p_index


def expected_result_metadata(
    *,
    k: float,
    p: float,
    k_index: int,
    p_index: int,
    trajectory_count: int,
    trajectory_points: int,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "grid_fingerprint": grid_fingerprint(),
        "k": k,
        "p": p,
        "k_index": k_index,
        "p_index": p_index,
        "trajectory_count": trajectory_count,
        "trajectory_points": trajectory_points,
        "base_seed": seed,
        "pair_seed": pair_seed(seed, k_index, p_index),
        "error_metric": "mean_relative_classification_error",
    }


def validate_error_matrix_shape(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> None:
    row_count = len(matrix)
    column_counts = {len(row) for row in matrix}
    if row_count != CANDIDATE_SHAPE[0] or column_counts != {CANDIDATE_SHAPE[1]}:
        raise ValueError(
            f"Expected error matrix shape {CANDIDATE_SHAPE}, got "
            f"{row_count} rows with column counts {sorted(column_counts)}"
        )

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from joblib import Parallel, delayed

from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF
from processes.distribution_fitter import WeibullFitter
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from scripts.dm_parameter_search import (
    CANDIDATE_GRID,
    CANDIDATE_SHAPE,
    K_VALUES,
    P_VALUES,
    build_search_manifest,
    evaluate_trajectory_for_scale,
    expected_result_metadata,
    result_file_name,
    save_search_manifest,
    validate_error_matrix_shape,
)
from utils.logging_setup import setup_logging
from utils.utils import create_empirical_cdf, kprint, ps_generate

DEFAULT_TRAJECTORY_COUNT = 100
DEFAULT_TRAJECTORY_POINTS = 1000
DEFAULT_SEED = 42
DEFAULT_N_JOBS = -1


def _load_inputs(path_to_main: Path) -> tuple[np.ndarray, Any]:
    radiuses_path = path_to_main / "radiuses.npy"
    throat_fitter_path = path_to_main / "throat_lengths_weibull_fitter.json"
    if not radiuses_path.is_file():
        raise RuntimeError(f"radiuses not found: {radiuses_path}")
    if not throat_fitter_path.is_file():
        raise RuntimeError(f"throat_lengths not found: {throat_fitter_path}")

    radiuses = np.load(radiuses_path)
    with throat_fitter_path.open() as file:
        throat_fitter = WeibullFitter.from_dict(json.load(file))
    return radiuses, throat_fitter


def _make_simulator(
    radiuses: np.ndarray,
    throat_fitter: Any,
    *,
    k: float,
    p: float,
    seed: int,
) -> KerogenWalkSimulator:
    # KerogenWalkSimulator currently uses both numpy.random and random.
    np.random.seed(seed)
    random.seed(seed)

    psd = create_empirical_cdf(radiuses)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=10_000)
    bs_ptl = BufferedSampler(throat_fitter, "ptl", size=10_000)
    ps = ps_generate("uniform", mean_count=100)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)
    return KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, k, p)


def _completed_scale_indices(errors: np.ndarray) -> tuple[int, ...]:
    """Validate a pair checkpoint and return its completed scale rows."""
    validate_error_matrix_shape(errors)
    if np.any(np.isinf(errors)):
        raise RuntimeError(
            "Parameter-search checkpoint contains infinite errors"
        )

    completed = []
    for scale_index, row in enumerate(errors):
        finite = np.isfinite(row)
        if np.all(finite):
            completed.append(scale_index)
        elif np.any(finite):
            raise RuntimeError(
                "Parameter-search checkpoint contains a partially written "
                f"scale row: {scale_index}"
            )
        elif not np.all(np.isnan(row)):
            raise RuntimeError(
                f"Parameter-search checkpoint has invalid scale row: {scale_index}"
            )
    return tuple(completed)


def _load_cached_result(
    result_path: Path,
    expected_metadata: dict[str, object],
    *,
    allow_legacy: bool,
) -> np.ndarray | None:
    if not result_path.is_file():
        return None

    with np.load(result_path) as payload:
        if "metadata_json" not in payload:
            if allow_legacy:
                kprint(
                    f"Replace legacy result on first checkpoint: {result_path}"
                )
                return None
            raise RuntimeError(
                f"Legacy result without reproducibility metadata: {result_path}. "
                "Rerun with --resume to replace legacy files while preserving "
                "compatible checkpoints, or use --force-recompute to restart all."
            )
        metadata = json.loads(str(payload["metadata_json"]))
        if metadata != expected_metadata:
            raise RuntimeError(
                f"Cached result metadata does not match this run: {result_path}. "
                "Rerun with --force-recompute or use matching CLI parameters."
            )

        if "mean_relative_errors" not in payload:
            raise RuntimeError(
                f"Cached result has no error matrix: {result_path}"
            )
        matrix = np.asarray(payload["mean_relative_errors"], dtype=np.float64)
        completed = _completed_scale_indices(matrix)
        if "completed_scale_indices" in payload:
            recorded_completed = tuple(
                int(v) for v in payload["completed_scale_indices"]
            )
            if recorded_completed != completed:
                raise RuntimeError(
                    "Checkpoint completion metadata does not match its "
                    f"matrix: {result_path}"
                )
    return matrix.copy()


def _save_checkpoint(
    result_path: Path,
    metadata: dict[str, object],
    errors: np.ndarray,
) -> None:
    completed = _completed_scale_indices(errors)

    temporary_path = result_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        # np.savez appends ".npz" to string/Path filenames but not to an
        # already-open file object, so this is what keeps the atomic
        # temp-file-then-rename pattern working.
        np.savez(
            file,
            metadata_json=json.dumps(metadata),
            mean_relative_errors=errors,
            completed_scale_indices=np.array(completed, dtype=np.int64),
        )
    temporary_path.replace(result_path)


def run(
    path_to_main: str | Path,
    *,
    trajectory_count: int = DEFAULT_TRAJECTORY_COUNT,
    trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    force_recompute: bool = False,
    resume: bool = False,
) -> None:
    if trajectory_count <= 0:
        raise ValueError("trajectory_count must be positive")
    if trajectory_points < 2:
        raise ValueError("trajectory_points must be at least 2")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if force_recompute and resume:
        raise ValueError("force_recompute and resume are mutually exclusive")

    main_path = Path(path_to_main)
    output_dir = main_path / "errors" / "find_best_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    radiuses, throat_fitter = _load_inputs(main_path)

    manifest = build_search_manifest(
        trajectory_count=trajectory_count,
        trajectory_points=trajectory_points,
        seed=seed,
    )
    cached_results: dict[Path, np.ndarray] = {}
    if not force_recompute:
        for k_index, k in enumerate(K_VALUES):
            for p_index, p in enumerate(P_VALUES):
                result_path = output_dir / result_file_name(k, p)
                metadata = expected_result_metadata(
                    k=k,
                    p=p,
                    k_index=k_index,
                    p_index=p_index,
                    trajectory_count=trajectory_count,
                    trajectory_points=trajectory_points,
                    seed=seed,
                )
                cached = _load_cached_result(
                    result_path,
                    metadata,
                    allow_legacy=resume,
                )
                if cached is not None:
                    cached_results[result_path] = cached

    manifest_path = save_search_manifest(manifest, output_dir)
    kprint(f"Parameter-grid manifest: {manifest_path}")

    candidates_per_pair = int(np.prod(CANDIDATE_SHAPE))
    total_candidates = len(K_VALUES) * len(P_VALUES) * candidates_per_pair
    completed_candidates = 0
    for k_index, k in enumerate(K_VALUES):
        for p_index, p in enumerate(P_VALUES):
            result_path = output_dir / result_file_name(k, p)
            metadata = expected_result_metadata(
                k=k,
                p=p,
                k_index=k_index,
                p_index=p_index,
                trajectory_count=trajectory_count,
                trajectory_points=trajectory_points,
                seed=seed,
            )
            errors = cached_results.get(result_path)
            if errors is None:
                errors = np.full(CANDIDATE_SHAPE, np.nan, dtype=np.float64)
            completed_scales = set(_completed_scale_indices(errors))
            if len(completed_scales) == CANDIDATE_SHAPE[0]:
                completed_candidates += candidates_per_pair
                kprint(f"Reuse {result_path}")
                continue

            simulator = _make_simulator(
                radiuses,
                throat_fitter,
                k=k,
                p=p,
                seed=int(cast(int, metadata["pair_seed"])),
            )
            trajectories = [
                simulator.run(trajectory_points)
                for _ in range(trajectory_count)
            ]
            ground_truth = []
            for trj in trajectories:
                assert trj.traps is not None
                ground_truth.append(trj.traps.copy())
            for scale_index, scale_candidates in enumerate(CANDIDATE_GRID):
                if scale_index in completed_scales:
                    completed_candidates += len(scale_candidates)
                    kprint(
                        f"Reuse checkpoint; "
                        f"ready {completed_candidates}/{total_candidates}; "
                        f"k={k}, p={p}, scale={scale_index}"
                    )
                    continue

                start_time = time.time()
                trajectory_errors = np.empty(
                    (trajectory_count, len(scale_candidates)), dtype=np.float64
                )
                result_stream = Parallel(
                    n_jobs=n_jobs, return_as="generator_unordered"
                )(
                    delayed(evaluate_trajectory_for_scale)(
                        trajectory_index,
                        trajectory,
                        expected,
                        scale_index,
                    )
                    for trajectory_index, (trajectory, expected) in enumerate(
                        zip(trajectories, ground_truth)
                    )
                )
                progress_interval = max(1, trajectory_count // 10)
                for completed_count, result in enumerate(
                    result_stream, start=1
                ):
                    trajectory_index, candidate_errors = result
                    trajectory_errors[trajectory_index] = candidate_errors
                    if (
                        completed_count == 1
                        or completed_count % progress_interval == 0
                        or completed_count == trajectory_count
                    ):
                        kprint(
                            f"Scale progress {completed_count}/{trajectory_count} "
                            f"trajectories; k={k}, p={p}, scale={scale_index}; "
                            f"time={time.time() - start_time:.1f}s"
                        )

                for parameter_index in range(len(scale_candidates)):
                    # Preserve the original candidate-major floating-point
                    # reduction order even though workers now run by trajectory.
                    ordered_errors = [
                        trajectory_errors[trajectory_index, parameter_index]
                        for trajectory_index in range(trajectory_count)
                    ]
                    errors[scale_index, parameter_index] = float(
                        np.mean(ordered_errors)
                    )

                _save_checkpoint(result_path, metadata, errors)
                completed_candidates += len(scale_candidates)
                kprint(
                    f"Ready {completed_candidates}/{total_candidates}; "
                    f"k={k}, p={p}, scale={scale_index}; "
                    f"checkpoint saved; time={time.time() - start_time:.1f}s"
                )

            kprint(f"Saved {result_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic DM parameter grid"
    )
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument(
        "--trajectory-count",
        type=int,
        default=DEFAULT_TRAJECTORY_COUNT,
        help=(
            "Synthetic trajectories per (k,p) "
            f"(default: {DEFAULT_TRAJECTORY_COUNT})"
        ),
    )
    parser.add_argument(
        "--trajectory-points",
        type=int,
        default=DEFAULT_TRAJECTORY_POINTS,
        help=f"Points per trajectory (default: {DEFAULT_TRAJECTORY_POINTS})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--force-recompute",
        action="store_true",
        help="Discard all compatible checkpoints and recompute every pair",
    )
    cache_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume compatible scale checkpoints and replace legacy result "
            "files as their first new checkpoint is completed"
        ),
    )
    args = parser.parse_args()

    run(
        args.path,
        trajectory_count=args.trajectory_count,
        trajectory_points=args.trajectory_points,
        seed=args.seed,
        n_jobs=args.n_jobs,
        force_recompute=args.force_recompute,
        resume=args.resume,
    )

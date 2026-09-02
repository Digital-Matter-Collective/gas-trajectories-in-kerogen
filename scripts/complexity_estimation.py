import argparse
import json
import os
import pickle
import platform
import random
import time
from os.path import isfile, join
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from processes.trajectory_analyzer.hybrid import HybridAnalyzer, HybridParams
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)
from utils.cache_manifest import check_cache, write_manifest
from utils.utils import create_empirical_cdf, kprint, ps_generate

DEFAULT_SEED = 42
DEFAULT_MIN_LENGTH = 500
DEFAULT_MAX_LENGTH = 8000
DEFAULT_LENGTH_STEP = 500
DEFAULT_REPEATS = 5
WARMUP_LENGTH = 500


def get_struct_params() -> DistanceMatrixParams:
    return DistanceMatrixParams(
        traj_type='fBm',
        nu=0.1,
        diag_percentile=10,
        kernel_size=2,
        list_mu=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        p_value=0.9,
    )


def get_prob_params() -> StructureInformedBayesParams:
    return StructureInformedBayesParams()


def build_analyzers(pil_gamma_fitter, throat_lengths_weibull_fitter):
    """Construct each analyzer once, outside the timed region.

    Construction loads DM thresholds and fits the NP threshold used by SIB;
    the manuscript's protocol times the analysis stage with structural
    distributions already prepared, so this one-time cost must not be
    counted as part of `.run()` timing.
    """
    return [
        (DistanceMatrixAnalyzer(get_struct_params()), "DM"),
        (
            StructureInformedBayesAnalyzer(
                get_prob_params(),
                pil_gamma_fitter,
                throat_lengths_weibull_fitter,
            ),
            "SIB",
        ),
        (
            HybridAnalyzer(
                HybridParams(get_prob_params(), get_struct_params(), 0.1),
                pil_gamma_fitter,
                throat_lengths_weibull_fitter,
            ),
            "HYB",
        ),
    ]


def environment_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }


class WalkSimulator(Protocol):
    """Structural type for `measure_complexity`'s simulator argument: only
    `run(length)` is used, so any object with that method works, including
    test doubles that don't subclass `KerogenWalkSimulator`."""

    def run(self, length: int) -> object: ...


def measure_complexity(
    simulator: WalkSimulator,
    analyzers: list,
    trajectory_lengths: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Return per-repeat run times, shape (len(trajectory_lengths), len(analyzers), repeats)."""
    np.random.seed(seed)
    random.seed(seed)

    # Warm-up: trigger Numba JIT compilation and BLAS first-call overhead
    # once per analyzer before any timed measurement.
    warmup_trajectory = simulator.run(WARMUP_LENGTH)
    for analyzer, _ in analyzers:
        analyzer.run(warmup_trajectory)

    times = np.zeros(
        (len(trajectory_lengths), len(analyzers), repeats), dtype=np.float64
    )
    for j, trj_len in enumerate(trajectory_lengths):
        # Same set of `repeats` trajectories is reused across all analyzers
        # so methods are compared on identical inputs.
        trajectories = [simulator.run(int(trj_len)) for _ in range(repeats)]
        for k, (analyzer, _) in enumerate(analyzers):
            for i, trajectory in enumerate(trajectories):
                start_time = time.perf_counter()
                analyzer.run(trajectory)
                times[j, k, i] = time.perf_counter() - start_time

        kprint(f"End estimation for trj_len = {trj_len}")
    return times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Complexity estimation for trajectory analyzers"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Data directory (contains pi_l_gamma_fitter.pkl etc.)",
    )
    parser.add_argument("output", type=Path, help="Output PDF path")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--length-step", type=int, default=DEFAULT_LENGTH_STEP)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Independent trajectories timed per (analyzer, length)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    path_to_main = str(args.path)

    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.pkl")
    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    path_to_radiuses: str = join(path_to_main, "radiuses.npy")

    if isfile(path_to_pil_gf):
        with open(path_to_pil_gf, "rb") as f:
            pil_gamma_fitter = pickle.load(f)
    else:
        raise RuntimeError("pi_l data not found")

    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    ps_type = 'uniform'  # poisson uniform
    psd = create_empirical_cdf(radiuses)
    ps = ps_generate(ps_type, mean_count=50)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=10_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=10_000)

    trajectory_lengths = np.arange(
        args.min_length, args.max_length, args.length_step
    )
    simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, 0.5, 0.5)
    analyzers = build_analyzers(pil_gamma_fitter, throat_lengths_weibull_fitter)

    times_path = args.path / "times.npy"
    cache_metadata = {
        "trajectory_lengths": trajectory_lengths,
        "repeats": args.repeats,
        "seed": args.seed,
        "struct_params": get_struct_params(),
        "prob_params": get_prob_params(),
        "analyzer_order": [name for _, name in analyzers],
    }
    cache_status = check_cache(times_path, cache_metadata)

    if cache_status == "match":
        times = np.load(times_path)
    elif cache_status == "legacy":
        kprint(
            f"Upgrading legacy cache {times_path} to provenance-tracked "
            "format (trusted as-is, not recomputed)"
        )
        times = np.load(times_path)
        write_manifest(times_path, cache_metadata)
    else:
        if cache_status == "mismatch":
            kprint(
                f"Cache {times_path} does not match current parameters; recomputing"
            )
        times = measure_complexity(
            simulator, analyzers, trajectory_lengths, args.repeats, args.seed
        )
        np.save(times_path, times)
        write_manifest(times_path, cache_metadata)

    mean_times = times.mean(axis=2)[1:, :]
    std_times = times.std(axis=2)[1:, :]
    plot_lengths = trajectory_lengths[1:]

    fit_summary = []

    def add_plot(
        mean_atime: np.ndarray, std_atime: np.ndarray, name: str
    ) -> None:
        logx = np.log(plot_lengths)
        logy = np.log(mean_atime)
        p = np.polyfit(logx, logy, deg=1)
        fit_summary.append(
            {
                "analyzer": name,
                "log_log_slope": float(p[0]),
                "log_log_intercept": float(p[1]),
            }
        )
        print(f"{name}: p = {p}")

        errorbar = plt.errorbar(
            plot_lengths,
            mean_atime,
            yerr=std_atime,
            fmt='o',
            markersize=5,
            alpha=0.35,
            capsize=3,
        )
        color = errorbar.lines[0].get_color()
        plt.plot(
            plot_lengths,
            np.exp(p[1]) * plot_lengths ** p[0],
            label=name,
            color=color,
        )

    for k, (_, name) in enumerate(analyzers):
        add_plot(mean_times[:, k], std_times[:, k], name)

    plt.xlabel('Trajectory length', fontsize=14)
    plt.ylabel('Execution Time, sec', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, fontsize=14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        str(args.output),
        bbox_inches="tight",
        pad_inches=0,
    )

    summary = {
        "trajectory_lengths": plot_lengths.tolist(),
        "repeats": args.repeats,
        "seed": args.seed,
        "environment": environment_metadata(),
        "fits": fit_summary,
    }
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    kprint(f"Saved complexity summary: {summary_path}")

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Patch

from base.trajectory import Trajectory
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from utils.types import f32


def _trap_segments(mask: npt.NDArray[np.bool_]) -> list[Tuple[int, int]]:
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _threshold_crit(
    analyzer: DistanceMatrixAnalyzer,
    mu: float,
    p_value: float,
) -> int:
    ind1 = int(mu * 2) - 1
    ind2 = int((1.0 - p_value) * 100.0) - 1
    return int(analyzer.list_threshold[ind1, ind2])


def _validate_mu(mu: float) -> None:
    supported = np.arange(0.5, 3.0 + 0.5, 0.5)
    if not np.any(np.isclose(mu, supported)):
        values = ", ".join(f"{value:g}" for value in supported)
        raise ValueError(f"Unsupported mu={mu:g}. Expected one of: {values}")


def _filter_short_traps(
    mask: npt.NDArray[np.bool_],
    crit: int,
) -> npt.NDArray[np.bool_]:
    filtered = mask.copy()
    for start, stop in _trap_segments(filtered):
        if stop - start + 1 <= crit:
            filtered[start : stop + 1] = False
    return filtered


def _cut_trajectory(
    trj: Trajectory,
    min_index: int,
    max_index: int | None,
) -> tuple[Trajectory, int, int]:
    count_points = len(trj.points)
    if min_index < 0:
        raise ValueError("--min-index must be >= 0")

    stop = count_points if max_index is None else max_index + 1
    if stop > count_points:
        raise ValueError(
            f"--max-index={max_index} is out of range for "
            f"{count_points} trajectory points"
        )
    if min_index >= stop:
        raise ValueError(
            "--min-index must be smaller than or equal to --max-index"
        )
    if stop - min_index < 2:
        raise ValueError("At least two trajectory points are required")

    trj.cut(min_index, stop)
    for key in (
        "points_without_periodic",
        "count_points",
        "delta_time",
        "delta_time_sec",
    ):
        trj.__dict__.pop(key, None)
    return trj, min_index, stop - 1


def compute_invariant(
    trj: Trajectory,
    analyzer: DistanceMatrixAnalyzer,
    mu: float,
    filter_short_traps: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], int]:
    sq_dist_matrix = analyzer.compute_pairwise_sq_dist(
        trj.points_without_periodic
    )
    diagonal_max = analyzer.diag_fill_list[int(2 * mu) - 1]
    vertical, diagonal, parallel = analyzer.RQA_block_measures(
        sq_dist_matrix,
        mu,
        diagonal_max,
    )

    denominator = parallel + diagonal - 1
    invariant = np.divide(
        vertical,
        denominator,
        out=np.zeros_like(vertical, dtype=np.float64),
        where=denominator != 0,
    )

    crit = _threshold_crit(analyzer, mu, analyzer.params.p_value)
    traps = invariant > analyzer.params.nu
    if filter_short_traps:
        if trj.count_points >= crit:
            traps = _filter_short_traps(traps, crit)
        else:
            traps = np.zeros_like(traps, dtype=np.bool_)

    return invariant, traps, crit


def _x_values(
    trj: Trajectory,
    x_axis: str,
    index_offset: int,
) -> tuple[npt.NDArray[np.float64], str]:
    if x_axis == "time-us":
        return trj.times.astype(np.float64) * 1e-6, r"$t, \mu s$"
    if x_axis == "time-ps":
        return trj.times.astype(np.float64), r"$t, ps$"
    x = np.arange(
        index_offset,
        index_offset + trj.count_points,
        dtype=np.float64,
    )
    return x, "Trajectory point"


def plot_invariant(
    trj: Trajectory,
    invariant: npt.NDArray[np.float64],
    traps: npt.NDArray[np.bool_],
    nu: float,
    mu: float,
    crit: int,
    x_axis: str,
    index_offset: int,
    trap_fill_y_max: float | None,
    filter_short_traps: bool,
    output: Path,
) -> None:
    x, xlabel = _x_values(trj, x_axis, index_offset)

    fig, ax = plt.subplots(figsize=(9, 7))

    # invariant[invariant > trap_fill_y_max] = trap_fill_y_max

    ax.plot(x, invariant, color="black", linewidth=1.8, label=r"$\nu$")
    fill_y_min = min(0.0, float(np.nanmin(invariant)))
    fill_y_max = nu if trap_fill_y_max is None else trap_fill_y_max
    has_traps = False
    for start, stop in _trap_segments(traps):
        has_traps = True
        ax.fill_between(
            [x[start], x[stop]],
            fill_y_min,
            fill_y_max,
            color="tab:red",
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )

    handles, labels = ax.get_legend_handles_labels()
    if has_traps:
        handles.append(Patch(color="tab:red", alpha=0.18))
        labels.append("Trapping")

    title = rf"DM invariant $\nu, \lambda={mu:g}$"
    if filter_short_traps:
        title += rf", filtered trap length $>{crit}$"
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(r"$\nu$", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim(x.min(), x.max())
    ax.legend(handles, labels, frameon=False, fontsize=24)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    trajectory_path: Path,
    trajectory_index: int,
    mu: float,
    nu: float,
    diag_percentile: int,
    kernel_size: int,
    p_value: float,
    traj_type: str,
    x_axis: str,
    min_index: int,
    max_index: int | None,
    trap_fill_y_max: float | None,
    filter_short_traps: bool,
    output: Path | None,
) -> None:
    _validate_mu(mu)
    trajectories = Trajectory.read_trajectoryes(trajectory_path)
    if not 0 <= trajectory_index < len(trajectories):
        raise IndexError(
            f"Trajectory index {trajectory_index} is out of range "
            f"for {len(trajectories)} trajectories"
        )

    trj = trajectories[trajectory_index]
    trj, actual_min_index, actual_max_index = _cut_trajectory(
        trj,
        min_index,
        max_index,
    )
    params = DistanceMatrixParams(
        traj_type=traj_type,
        nu=nu,
        diag_percentile=diag_percentile,
        kernel_size=kernel_size,
        list_mu=np.array([mu], dtype=f32),
        p_value=p_value,
    )
    analyzer = DistanceMatrixAnalyzer(params)
    invariant, traps, crit = compute_invariant(
        trj,
        analyzer,
        mu,
        filter_short_traps,
    )

    if output is None:
        output = trajectory_path.parent / "figs" / (
            f"dm_invariant_trj={trajectory_index}_"
            f"range={actual_min_index}-{actual_max_index}.svg"
        )

    plot_invariant(
        trj,
        invariant,
        traps,
        nu,
        mu,
        crit,
        x_axis,
        actual_min_index,
        trap_fill_y_max,
        filter_short_traps,
        output,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the DistanceMatrix invariant for one trajectory"
    )
    parser.add_argument("trajectory", type=Path, help="Input .gro trajectory")
    parser.add_argument(
        "--trajectory-index",
        "--num",
        type=int,
        default=0,
        help="Trajectory/molecule index in the .gro file",
    )
    parser.add_argument(
        "--mu",
        type=float,
        required=True,
        help="DistanceMatrix mu value",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Invariant threshold for trapping",
    )
    parser.add_argument(
        "--diag-percentile",
        type=int,
        default=5,
        choices=(0, 5, 10, 50),
        help="Reference percentile used to fill matrix diagonals",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="DM convolution kernel radius",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=0.9,
        help="DM p-value used for critical trap length",
    )
    parser.add_argument(
        "--traj-type",
        choices=("fBm", "Bm"),
        default="fBm",
        help="Reference motion type for DM thresholds",
    )
    parser.add_argument(
        "--x-axis",
        choices=("index", "time-ps", "time-us"),
        default="index",
        help="X axis for the plot",
    )
    parser.add_argument(
        "--min-index",
        type=int,
        default=0,
        help="First trajectory point index to include",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="Last trajectory point index to include (inclusive)",
    )
    parser.add_argument(
        "--filter-short-traps",
        action="store_true",
        help="Apply DM critical-length filtering to highlighted regions",
    )
    parser.add_argument(
        "--trap-fill-y-max",
        type=float,
        default=None,
        help="Upper y level for trapping shading (default: nu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path",
    )
    args = parser.parse_args()

    run(
        args.trajectory,
        args.trajectory_index,
        args.mu,
        args.nu,
        args.diag_percentile,
        args.kernel_size,
        args.p_value,
        args.traj_type,
        args.x_axis,
        args.min_index,
        args.max_index,
        args.trap_fill_y_max,
        args.filter_short_traps,
        args.output,
    )

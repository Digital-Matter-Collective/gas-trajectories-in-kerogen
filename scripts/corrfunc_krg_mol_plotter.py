import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

from base.trajectory import Trajectory
from utils.cache_manifest import check_cache, file_fingerprint, write_manifest
from utils.utils import kprint


@dataclass
class RMSDResult:
    rmsd: np.ndarray
    t: np.ndarray


def extract_mean_displacement(trajectories, traj_stride: int = 1) -> RMSDResult:
    a_msd = []
    a_t = []

    for i, trj in enumerate(trajectories[::traj_stride]):
        msd = trj.msd_average_time()
        t = np.asarray(trj.times, dtype=float)

        a_t.append((t * 1e-6).astype(float))
        a_msd.append(msd)

    msd_mean = np.mean(np.stack(a_msd, axis=0), axis=0)
    rmsd = np.sqrt(msd_mean)
    # rmsd = msd_mean

    return RMSDResult(rmsd, a_t[0])


def plot_corrfunc_and_md(
    trj_msd: RMSDResult,
    save_path: Path,
    max_t: float = 2.8,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_msd = default_colors[3]

    r = trj_msd.rmsd
    t = trj_msd.t
    mask = t <= max_t
    mask[0] = False
    t = t[mask]
    r = r[mask]

    plt.plot(
        t,
        r,
        color=color_msd,
        linewidth=3.0,
    )

    plt.yscale("log")
    plt.xscale("log")

    plt.ylim(bottom=0.08, top=1.5)

    ax = plt.gca()

    yticks = [0.1, 0.2, 1.0]
    ylabels = ["0.1", "0.2", "1.0"]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Оставляем рамку целиком
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Название оси Y переносим вправо
    ax.yaxis.set_label_position("right")

    plt.xlabel(r"Time delay, $\mu$s", fontsize=20)
    plt.ylabel(r"$\mathrm{RMSD}(t)$, nm", fontsize=20)

    # Убираем тики и подписи слева
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        labelsize=16,
    )
    ax.tick_params(
        axis="x",
        labelsize=16,
    )

    # plt.legend(fontsize=20, frameon=False)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_kerogen_molecula_coorfunc(input: Path, output: Path, msd_path: Path):
    trajectories = Trajectory.read_trajectoryes(input)

    msd_path.parent.mkdir(parents=True, exist_ok=True)

    traj_stride = 1
    cache_metadata = {
        "input": file_fingerprint(input),
        "traj_stride": traj_stride,
    }
    status = check_cache(msd_path, cache_metadata)
    if status == "match":
        with open(msd_path, "rb") as f:
            trj_msd = pickle.load(f)
    elif status == "legacy":
        kprint(
            f"Upgrading legacy cache {msd_path} to provenance-tracked "
            "format (trusted as-is, not recomputed)"
        )
        with open(msd_path, "rb") as f:
            trj_msd = pickle.load(f)
        write_manifest(msd_path, cache_metadata)
    else:
        if status == "mismatch":
            kprint(
                f"Cache {msd_path} does not match current input; recomputing"
            )
        trj_msd = extract_mean_displacement(
            trajectories, traj_stride=traj_stride
        )
        with open(msd_path, "wb") as f:
            pickle.dump(trj_msd, f)
        write_manifest(msd_path, cache_metadata)

    plot_corrfunc_and_md(
        trj_msd,
        save_path=output,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output figure path")
    parser.add_argument("msd", type=Path, help="Output MSD pickle path")

    args = parser.parse_args()

    plot_kerogen_molecula_coorfunc(
        args.input,
        args.output,
        args.msd,
    )

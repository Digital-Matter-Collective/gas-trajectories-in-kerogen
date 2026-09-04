import argparse
from pathlib import Path

import numpy as np

from base.trajectory import Trajectory
from utils.logging_setup import setup_logging
from utils.utils import kprint
from visualizer.visualizer import Visualizer, WrapMode


def visualize_dist_trajectory(
    traj_path: str, num: int, traps_path: str = ""
) -> None:
    trajectories = Trajectory.read_trajectories(traj_path)
    trj = trajectories[num]
    cp = trj.points.shape[0]
    trj.cut(cp // 2)

    use_clusters = len(traps_path) != 0

    if use_clusters:
        with np.load(traps_path) as data:
            trj.traps = data["traps"]

    kprint(trj.points.shape)
    Visualizer.draw_trajectories(
        [trj],
        wrap_mode=WrapMode.EMPTY,
        periodic=False,
        radius=0.005,
        with_points=True,
        color_type='clusters' if use_clusters else 'dist',
    )
    Visualizer.show()


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Visualize distance trajectory"
    )
    parser.add_argument("trj", type=Path, help="Trajectory file (.gro)")
    parser.add_argument("num", type=int, help="Molecule index")
    parser.add_argument("--traps", type=Path, help="Traps .npz file (optional)")
    args = parser.parse_args()

    traps_path = str(args.traps) if args.traps else ""
    visualize_dist_trajectory(str(args.trj), args.num, traps_path)

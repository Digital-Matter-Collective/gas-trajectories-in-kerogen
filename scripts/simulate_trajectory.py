import argparse
import pickle
import sys
from os.path import isfile, join, realpath
from pathlib import Path

import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.bufferedsampler import BufferedSampler  # noqa: E402
from base.discretecdf import DiscreteCDF  # noqa: E402
from base.empiricalcdf import EmpiricalCDF  # noqa: E402
from processes.kerogen_walk_simulator import KerogenWalkSimulator  # noqa: E402
from utils.utils import create_empirical_cdf, ps_generate  # noqa: E402
from visualizer.visualizer import Visualizer, WrapMode  # noqa: E402


def run(
    path_to_main: str,
    k: float = 0.5,
    p: float = 0.5,
    steps: int = 1000,
    radius: float = 0.02,
) -> None:
    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type, mean_count=50)

    path_to_radiuses: str = join(path_to_main, "radiuses.npy")
    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")

    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    psd = create_empirical_cdf(radiuses)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=100_000)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=100_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=100_000)

    simulator = KerogenWalkSimulator(
        bs_psd, bs_ps, bs_ptl, k, p, with_history=False
    )
    traj = simulator.run(steps)
    Visualizer.draw_trajectoryes(
        [traj],
        radius=radius,
        periodic=False,
        wrap_mode=WrapMode.EMPTY,
        with_points=True,
        color_type='dist',
    )
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulate and visualize one kerogen gas-molecule trajectory"
    )
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument(
        "--k", type=float, default=0.5, help="Trapping probability parameter"
    )
    parser.add_argument(
        "--p", type=float, default=0.5, help="Return probability parameter"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulated steps"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.02,
        help="Rendered trajectory tube/point radius",
    )
    args = parser.parse_args()

    run(
        str(args.path), k=args.k, p=args.p, steps=args.steps, radius=args.radius
    )

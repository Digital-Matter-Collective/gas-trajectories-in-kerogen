from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from base.trajectory import Trajectory


class TrajectoryAnalyzer(ABC):
    """Common interface for trajectory-step classifiers.

    An input trajectory with ``N`` points contains ``N - 1`` directed steps.
    Analyzer outputs therefore contain ``N - 1`` Boolean labels. Trajectories
    shorter than ten points are outside the supported analysis regime: they do
    not contain enough observations for meaningful trapping statistics.
    """

    MIN_TRAJECTORY_POINTS = 10

    @classmethod
    def validate_trajectory(cls, trj: Trajectory) -> None:
        point_count = trj.count_points
        if point_count < cls.MIN_TRAJECTORY_POINTS:
            raise ValueError(
                f"{cls.__name__} requires at least "
                f"{cls.MIN_TRAJECTORY_POINTS} trajectory points; "
                f"got {point_count}"
            )

    @abstractmethod
    def run(self, trj: Trajectory) -> npt.NDArray[np.bool_]:
        pass

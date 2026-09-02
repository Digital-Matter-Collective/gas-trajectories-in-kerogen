from dataclasses import dataclass
from typing import Optional

import numpy as np

from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer
from utils.types import NPBArray, NPFArray, f32


@dataclass
class HybridParams:
    prob_params: StructureInformedBayesParams
    struct_params: DistanceMatrixParams
    prob_diff: float = 0.1


class HybridAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: HybridParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf
        self.trap_approx: Optional[NPBArray] = None
        self.sib_analyzer = StructureInformedBayesAnalyzer(
            params.prob_params,
            pi_l_gf,
            throat_lengthes_wf,
        )

    @staticmethod
    def name() -> str:
        return "hybrid"

    def set_trap_approx(self, trap_approx: NPBArray):
        """Set a DM mask for the next :meth:`run` call only.

        The override is deliberately one-shot: keeping it on the analyzer would
        apply one trajectory's DM labels to all subsequent trajectories.
        """
        self.trap_approx = np.asarray(trap_approx, dtype=np.bool_).copy()

    def get_trap_approx(self, trj: Optional[Trajectory] = None) -> NPBArray:
        trap_approx = self.trap_approx
        self.trap_approx = None

        if trap_approx is None:
            assert trj is not None
            analyzer = DistanceMatrixAnalyzer(self.params.struct_params)
            trap_approx = analyzer.run(trj)
            print(" --- Matrix Algorithm finished")

        if trj is not None:
            expected_shape = (trj.count_points - 1,)
            if trap_approx.shape != expected_shape:
                raise ValueError(
                    "DM trap mask shape does not match trajectory steps: "
                    f"expected {expected_shape}, got {trap_approx.shape}"
                )
        return trap_approx

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        self.validate_trajectory(trj)
        trap_approx = self.get_trap_approx(trj)

        _, probabilityies = self.sib_analyzer.analyze(trj)
        result = probabilityies > 0.5
        struct_mask = np.abs(probabilityies - 0.5) < self.params.prob_diff
        result[struct_mask] = trap_approx[struct_mask]
        assert not np.any(result == -1)
        return result

    @staticmethod
    def distances(points: NPFArray) -> NPFArray:
        dxyz = points[:-1, :] - points[1:, :]
        result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1)).astype(f32)
        return result

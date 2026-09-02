from dataclasses import dataclass
from typing import Optional
import numpy as np

from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer

from utils.types import NPBArray, i32
from processes.trajectory_analyzer.np import (
    NeymanPearsonAnalyzer,
    NeymanPearsonParams,
)
from processes.trajectory_analyzer.bayes import (
    BayesParams,
    BayesAnalyzer,
)


@dataclass
class StructureInformedBayesParams(NeymanPearsonParams, BayesParams):
    pass


class StructureInformedBayesAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: StructureInformedBayesParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaFitter] = None

        self.threshold = NeymanPearsonAnalyzer.calculate_threshold(
            self.pi_l_gf, self.throat_lengthes_wf, self.params.error
        )

    @staticmethod
    def name() -> str:
        return "sib"

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        _, probabilityies = self.analyze(trj)
        return probabilityies > 0.5

    def analyze(self, trj: Trajectory) -> tuple[float, np.ndarray]:
        """Return the NP-initialized Bayesian prior and posteriors."""
        likelihood = NeymanPearsonAnalyzer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
        )
        result = (likelihood < self.threshold).astype(i32)
        return BayesAnalyzer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
            self.params.critical_probability,
            p_trap=np.sum(result) / len(result),
        )

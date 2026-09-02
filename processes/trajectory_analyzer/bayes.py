from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    PdfFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer
from utils.types import NPBArray, NPFArray
from utils.utils import pdistances

DEFAULT_MAX_EM_ITERATIONS = 1000


@dataclass
class BayesParams:
    critical_probability: float = 1e-3


class BayesAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: BayesParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaFitter] = None

    @staticmethod
    def name() -> str:
        return "bayes"

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        _, probabilityies = self.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
            self.params.critical_probability,
        )
        result = probabilityies > 0.5
        return result

    @staticmethod
    def analyze(
        trj: Trajectory,
        transition_step_fitter: PdfFitter,
        trapped_step_fitter: PdfFitter,
        critical_probability: float,
        p_trap: float = 0.5,
        max_iter: int = DEFAULT_MAX_EM_ITERATIONS,
    ) -> Tuple[float, NPFArray]:
        BayesAnalyzer.validate_trajectory(trj)
        points = trj.points_without_periodic
        distances = pdistances(points)

        L_T = trapped_step_fitter.pdf(distances).astype(np.float64)  # p(x|trap)
        L_C = transition_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|not_trap)

        eps = 1e-300
        L_T = np.maximum(L_T, eps)
        L_C = np.maximum(L_C, eps)

        prev = np.inf
        iterations = 0
        gamma = None
        while np.abs(p_trap - prev) > critical_probability:
            if iterations >= max_iter:
                raise RuntimeError(
                    "Bayes EM loop did not converge within "
                    f"{max_iter} iterations (critical_probability="
                    f"{critical_probability}); last |Δp_trap|="
                    f"{abs(p_trap - prev):.3e}. Increase max_iter or "
                    "critical_probability if this is expected."
                )
            prev = p_trap

            denom = p_trap * L_T + (1.0 - p_trap) * L_C
            denom = np.maximum(denom, eps)

            gamma = (p_trap * L_T) / denom  # P(trap | x)
            # EM update: ожидаемая доля
            p_trap = float(gamma.mean())
            iterations += 1

        assert trapped_step_fitter is not None
        assert transition_step_fitter is not None
        assert gamma is not None

        return (p_trap, gamma)

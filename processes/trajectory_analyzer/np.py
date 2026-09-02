from dataclasses import dataclass
from typing import Optional

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


@dataclass
class NeymanPearsonParams:
    error: float = 0.01


class NeymanPearsonAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: NeymanPearsonParams,
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
        return "np"

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        likelihood = self.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
        )
        result = likelihood < self.threshold
        return result

    @staticmethod
    def analyze(
        trj: Trajectory,
        transition_step_fitter: PdfFitter,
        trapped_step_fitter: PdfFitter,
    ) -> NPFArray:
        NeymanPearsonAnalyzer.validate_trajectory(trj)
        points = trj.points_without_periodic
        distances = pdistances(points)

        L_T = trapped_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|trap) H_1
        L_C = transition_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|~trap) H_2

        # Floor both densities away from zero (same eps convention as
        # BayesAnalyzer) so a step outside either fitted support divides to
        # a large-but-finite ratio instead of inf/nan and a RuntimeWarning.
        eps = 1e-300
        L_T = np.maximum(L_T, eps)
        L_C = np.maximum(L_C, eps)

        return np.divide(L_C, L_T).astype(np.float64)

    @staticmethod
    def calculate_threshold(
        f_distr, g_distr, epsilon, x_max: float = 1.0
    ) -> float:
        x = np.linspace(0, x_max, 1_000_000)
        f = f_distr.pdf(x)
        g = g_distr.pdf(x)
        # likelihood ratio
        likelihood = np.divide(g, f, out=np.zeros_like(g), where=f > 0)

        # нормируем веса (аппроксимация интеграла)
        weights = f / np.sum(f)

        # сортируем по убыванию l
        idx = np.argsort(likelihood)[::-1]
        l_sorted = likelihood[idx]
        w_sorted = weights[idx]

        # накопленная масса
        cumulative = np.cumsum(w_sorted)

        # находим порог
        mask = cumulative >= epsilon
        threshold = l_sorted[mask][0]
        return float(threshold)

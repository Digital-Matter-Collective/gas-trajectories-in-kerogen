from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.types import NPFArray, NPIArray


@dataclass
class TrapSequence:
    traps: NPIArray
    times: NPFArray

    def get_zero_trap_probability(self) -> float:
        return np.count_nonzero(self.times == 0) / len(self.times)

    def get_zero_trap_count(self) -> int:
        return int(np.count_nonzero(self.times == 0))

    def get_non_zero_trap_count(self) -> int:
        """Оценка вероятности трепинга k"""
        return int(np.count_nonzero(self.times > 0))

    def get_count_zero_steps_inside(self) -> int:
        mask = self.times > 0
        return int(np.sum(self.traps[mask]))

    def get_non_count_zero_steps_inside(self) -> int:
        mask = self.times == 0
        return int(np.sum(self.traps[mask]))

    def save_npz(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            np.savez(f, traps=self.traps, times=self.times)

    @classmethod
    def load_npz(cls, path: str | Path) -> "TrapSequence":
        with np.load(path) as data:
            return cls(traps=data["traps"], times=data["times"])

from base.trap_sequence import TrapSequence
import numpy as np
from utils.types import NPBArray

# Bump when get_trap_seq's run-encoding changes, so callers that cache its
# output (e.g. scripts/trap_distr_builder.py) invalidate stale seq/traps
# caches computed with a previous, possibly buggy, version instead of
# silently trusting them (P1-08: fixed a boundary-counting bug that biased
# N0/Nt/k_est).
TRAP_EXTRACTOR_VERSION = 2


class TrapExtractor:
    @staticmethod
    def get_trap_seq(edge_traps: NPBArray, delta_time: float) -> TrapSequence:
        """Collapse `edge_traps` into one (duration, count) entry per run.

        Each maximal run of consecutive trapped edges becomes one non-zero
        entry (duration = run_length * delta_time, count = run_length + 1,
        keeping the pre-existing "count the exit edge too" convention for
        trap-duration statistics used by the power-law fit). Each maximal
        run of consecutive free edges becomes exactly one zero-duration
        entry. There is no artificial boundary entry: `N0`/`Nt` (derived
        from how many entries have times == 0 vs > 0 — see
        `TrapSequence.get_zero_trap_count`/`get_non_zero_trap_count`) equal
        the number of free/trapped runs actually observed, so an all-trapped
        or all-free trajectory yields `k_est` of exactly 1.0 or 0.0.
        """
        if len(edge_traps) == 0:
            raise ValueError("edge_traps must not be empty")

        times: list[float] = []
        traps: list[int] = []

        run_value = bool(edge_traps[0])
        run_length = 1

        def close_run(value: bool, length: int) -> None:
            if value:
                times.append(length * delta_time)
                traps.append(length + 1)
            else:
                times.append(0.0)
                traps.append(length)

        for i in range(1, len(edge_traps)):
            value = bool(edge_traps[i])
            if value == run_value:
                run_length += 1
                continue
            close_run(run_value, run_length)
            run_value, run_length = value, 1

        close_run(run_value, run_length)

        return TrapSequence(np.array(traps), np.array(times))

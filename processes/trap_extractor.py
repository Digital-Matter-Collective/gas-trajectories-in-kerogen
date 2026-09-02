import numpy as np

from base.trap_sequence import TrapSequence
from utils.types import NPBArray

# Bump when get_trap_seq's event encoding changes, so callers invalidate
# stale derived values. Version 3 fixes the version-2 regression that counted
# free runs rather than fully observed bypass visits and forced k_est to 0.5.
TRAP_EXTRACTOR_VERSION = 3


class TrapExtractor:
    @staticmethod
    def get_trap_seq(edge_traps: NPBArray, delta_time: float) -> TrapSequence:
        """Collapse `edge_traps` into one (duration, count) entry per run.

        Each maximal run of consecutive trapped edges becomes one non-zero
        entry (duration = run_length * delta_time, count = run_length + 1).

        A run of ``L`` free/inter-trap edges contains ``L - 1`` fully
        observed zero-duration visits: every adjacent free/free pair means
        that the intermediate trap was bypassed.  The trajectory boundaries
        are censored, so no artificial zero-duration event is added at either
        end.  Collapsing a free run to one event would make free and trapped
        run counts alternate and force ``k_est`` towards 0.5 independently
        of the actual capture frequency.
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
                # L transition edges delimit L - 1 complete intermediate
                # trap visits.  The visits outside the observed trajectory
                # are unknown and must not be invented as boundary events.
                times.extend([0.0] * (length - 1))
                traps.extend([1] * (length - 1))

        for i in range(1, len(edge_traps)):
            value = bool(edge_traps[i])
            if value == run_value:
                run_length += 1
                continue
            close_run(run_value, run_length)
            run_value, run_length = value, 1

        close_run(run_value, run_length)

        return TrapSequence(np.array(traps), np.array(times))

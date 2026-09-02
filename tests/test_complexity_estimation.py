import numpy as np

from scripts.complexity_estimation import WARMUP_LENGTH, measure_complexity


class _FakeTrajectory:
    def __init__(self, length: int, draw_index: int) -> None:
        self.length = length
        self.draw_index = draw_index


class _FakeSimulator:
    """Records every requested length and returns a distinct trajectory."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, length: int) -> _FakeTrajectory:
        self.calls.append(length)
        return _FakeTrajectory(length, len(self.calls))


class _FakeAnalyzer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen_trajectories: list[_FakeTrajectory] = []

    def run(self, trajectory: _FakeTrajectory) -> None:
        self.seen_trajectories.append(trajectory)


def test_same_trajectories_are_shared_across_analyzers_per_length() -> None:
    simulator = _FakeSimulator()
    dm, sib = _FakeAnalyzer("DM"), _FakeAnalyzer("SIB")
    analyzers = [(dm, "DM"), (sib, "SIB")]
    trajectory_lengths = np.array([100, 200])

    measure_complexity(
        simulator, analyzers, trajectory_lengths, repeats=3, seed=1
    )

    # One warm-up call plus 3 repeats per length.
    assert simulator.calls == [WARMUP_LENGTH, 100, 100, 100, 200, 200, 200]

    # Both analyzers must see the identical trajectory objects per length,
    # not independently drawn ones (P1-07: same inputs across methods).
    dm_by_length = [t for t in dm.seen_trajectories if t.length == 100]
    sib_by_length = [t for t in sib.seen_trajectories if t.length == 100]
    assert [t.draw_index for t in dm_by_length] == [
        t.draw_index for t in sib_by_length
    ]


def test_warmup_runs_once_per_analyzer_before_timed_region() -> None:
    simulator = _FakeSimulator()
    dm = _FakeAnalyzer("DM")

    measure_complexity(
        simulator, [(dm, "DM")], np.array([50]), repeats=2, seed=1
    )

    # First trajectory seen by the analyzer is the untimed warm-up one.
    assert dm.seen_trajectories[0].length == WARMUP_LENGTH
    assert len(dm.seen_trajectories) == 1 + 2  # warm-up + repeats


def test_output_shape_is_lengths_by_analyzers_by_repeats() -> None:
    simulator = _FakeSimulator()
    analyzers = [(_FakeAnalyzer("DM"), "DM"), (_FakeAnalyzer("SIB"), "SIB")]
    trajectory_lengths = np.array([10, 20, 30])

    times = measure_complexity(
        simulator, analyzers, trajectory_lengths, repeats=4, seed=1
    )

    assert times.shape == (3, 2, 4)
    assert np.all(times >= 0)


def test_seed_makes_simulation_reproducible() -> None:
    class _SeededFakeSimulator:
        def __init__(self) -> None:

            self.draws: list[float] = []

        def run(self, length: int):
            import random

            self.draws.append(random.random())
            return _FakeTrajectory(length, len(self.draws))

    sim_a = _SeededFakeSimulator()
    sim_b = _SeededFakeSimulator()
    analyzers_a = [(_FakeAnalyzer("DM"), "DM")]
    analyzers_b = [(_FakeAnalyzer("DM"), "DM")]

    measure_complexity(sim_a, analyzers_a, np.array([100]), repeats=2, seed=7)
    measure_complexity(sim_b, analyzers_b, np.array([100]), repeats=2, seed=7)

    assert sim_a.draws == sim_b.draws

import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.trajectory import Trajectory
from processes.pnm_walk_simulator import PnmWalkSimulator, load_pnm_graph
from utils.utils import ps_generate


def _write_pnm_fixture(prefix: Path) -> None:
    """A tiny 4-pore network: pore1(r=0.10) -- pore2(r=0.20) -- pore3(r=0.15),
    with pore2 also -- pore4(r=0.25); plus one throat to the boundary
    reservoir (pore index 0) that `read_pnm_ext_data` must drop."""
    (prefix.parent / f"{prefix.name}_node1.dat").write_text(
        "4    0    0    0\n"
        "1    0.0    0.0    0.0\n"
        "2    1.0    0.0    0.0\n"
        "3    2.0    0.0    0.0\n"
        "4    1.0    1.0    0.0\n"
    )
    (prefix.parent / f"{prefix.name}_node2.dat").write_text(
        "1    0.0    0.10\n"
        "2    0.0    0.20\n"
        "3    0.0    0.15\n"
        "4    0.0    0.25\n"
    )
    (prefix.parent / f"{prefix.name}_link1.dat").write_text(
        "4\n"
        "1 1 2 0.05 0.6 0.5\n"
        "2 2 3 0.05 0.6 0.6\n"
        "3 2 4 0.05 0.6 0.7\n"
        "4 1 0 0.05 0.6 0.3\n"
    )


def test_load_pnm_graph_reads_real_pore_radii_and_throat_lengths(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)

    graph = load_pnm_graph(str(prefix), scale=1.0)

    assert graph.number_of_nodes() == 4
    assert graph.nodes[0]["size"] == pytest.approx(0.10)
    assert graph.nodes[1]["size"] == pytest.approx(0.20)
    assert graph.nodes[2]["size"] == pytest.approx(0.15)
    assert graph.nodes[3]["size"] == pytest.approx(0.25)

    # The boundary throat (pore1 -> reservoir 0) must not create a node.
    assert -1 not in graph.nodes
    assert graph.number_of_edges() == 3
    assert graph[0][1]["weight"] == pytest.approx(0.5)
    assert graph[1][2]["weight"] == pytest.approx(0.6)
    assert graph[1][3]["weight"] == pytest.approx(0.7)


def test_load_pnm_graph_drops_pores_at_or_below_min_radius(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)

    graph = load_pnm_graph(str(prefix), scale=1.0, min_radius=0.12)

    assert set(graph.nodes) == {1, 2, 3}
    assert not graph.has_edge(0, 1)
    assert graph.has_edge(1, 2)
    assert graph.has_edge(1, 3)


def _bs_ps(mean_count: int = 5) -> BufferedSampler:
    return BufferedSampler(
        DiscreteCDF(ps_generate("uniform", mean_count=mean_count)),
        "ps",
        size=1_000,
    )


def test_pnm_walk_simulator_rejects_a_graph_with_no_throats() -> None:
    graph = nx.Graph()
    graph.add_node(0, size=0.1)
    graph.add_node(1, size=0.1)

    with pytest.raises(ValueError, match="no throats"):
        PnmWalkSimulator(graph, _bs_ps(), k=0.5, p=0.5)


def test_pnm_walk_simulator_start_node_defaults_to_a_connected_pore(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)
    graph = load_pnm_graph(str(prefix), scale=1.0)
    graph.add_node(99, size=0.05)  # isolated pore, no throats

    simulator = PnmWalkSimulator(graph, _bs_ps(), k=0.5, p=0.5)

    assert simulator.start_node != 99
    assert graph.degree(simulator.start_node) > 0


def test_pnm_walk_simulator_rejects_an_isolated_explicit_start_node(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)
    graph = load_pnm_graph(str(prefix), scale=1.0)
    graph.add_node(99, size=0.05)

    with pytest.raises(ValueError, match="start_node=99"):
        PnmWalkSimulator(graph, _bs_ps(), k=0.5, p=0.5, start_node=99)


def test_pnm_walk_simulator_produces_a_trajectory_of_the_requested_length(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)
    graph = load_pnm_graph(str(prefix), scale=1.0)

    np.random.seed(42)
    random.seed(42)
    simulator = PnmWalkSimulator(
        graph, _bs_ps(), k=0.5, p=0.5, with_history=False, start_node=1
    )
    trajectory = simulator.run(30)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.points.shape[0] >= 30
    assert trajectory.traps is not None
    assert trajectory.traps.shape[0] == trajectory.points.shape[0] - 1


def test_pnm_walk_simulator_is_deterministic_given_the_same_seed(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "pnm"
    _write_pnm_fixture(prefix)
    graph = load_pnm_graph(str(prefix), scale=1.0)

    def _run() -> np.ndarray:
        np.random.seed(7)
        random.seed(7)
        simulator = PnmWalkSimulator(
            graph, _bs_ps(), k=0.5, p=0.5, with_history=False, start_node=1
        )
        return simulator.run(40).points

    np.testing.assert_array_equal(_run(), _run())

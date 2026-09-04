import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from base.boundingbox import BoundingBox, Range
from base.bufferedsampler import BufferedSampler
from base.reader import Reader
from base.trajectory import Trajectory
from processes.kerogen_walk_simulator import (
    KerogenWalkSimulator,
    Uniform01,
    UnitVector3,
)
from utils.types import NPFArray, f32


def load_pnm_graph(
    pnm_prefix: str,
    scale: float = Reader.PNM_M_TO_NM,
    min_radius: float = 0.0,
) -> nx.Graph:
    """Build the real pore network as a graph: nodes are pores (`size` =
    pore radius), edges are throats (`weight` = throat length). Node/edge
    values are read directly from the PNM files, not fit to a distribution.

    Pores with `size <= min_radius` are dropped (their incident throats go
    with them) — Statoil-format PNMs can contain zero/negative-radius or
    boundary-reservoir entries that are not real pores; `read_pnm_data`
    elsewhere in this codebase filters the same way for distribution
    fitting. Positions are not attached to nodes: `PnmWalkSimulator` places
    points in its own running trajectory frame, not in the PNM's absolute
    coordinate frame (see its docstring).
    """
    radiuses, throat_lengths, linked_list, _positions = (
        Reader.read_pnm_ext_data(pnm_prefix, scale=scale)
    )

    graph = nx.Graph()
    graph.add_nodes_from(
        (i, {"size": float(radiuses[i])}) for i in range(len(radiuses))
    )
    graph.add_weighted_edges_from(
        (int(n1), int(n2), float(length))
        for (n1, n2), length in zip(linked_list, throat_lengths[:, 2])
    )
    graph.remove_nodes_from(
        [n for n, size in graph.nodes(data="size") if size <= min_radius]
    )
    return graph


class PnmWalkSimulator:
    """Same trapping/return random walk as `KerogenWalkSimulator`, but
    walking the real, finite pore network instead of drawing trap size and
    inter-trap step length from independently fit P(r)/P(h) distributions.

    Trap size is the current pore's real radius, and the step between traps
    is the real length of the throat connecting them, both read from `pnm`.
    The count of jitter steps taken inside a trap still comes from `bs_ps`
    (dwell time isn't a property of the pore network itself). Points are
    placed in the walk's own running 3D frame (a random direction times the
    real throat length, exactly like `KerogenWalkSimulator`) rather than at
    the PNM's absolute pore coordinates — the two frames are never mixed.

    Because the real network is finite, `move_next`'s "explore a fresh
    trap" step falls back to revisiting an already-visited neighbor once
    every real neighbor of the current pore has been visited — there is no
    always-available fresh trap to invent, unlike the synthetic simulator.
    """

    def __init__(
        self,
        pnm: nx.Graph,
        bs_ps: BufferedSampler,
        k: float,
        p: float,
        with_history: bool = True,
        start_node: Optional[int] = None,
    ) -> None:
        """
        :param pnm: pore network graph from `load_pnm_graph`: nodes carry a
            `size` (pore radius) attribute, edges carry a `weight` (throat
            length) attribute.
        :param bs_ps: count-of-steps-inside-a-trap distribution.
        :param p: probability of moving to an already-visited adjacent trap,
            vs. 1 - p to move toward a (if available) fresh one.
        :param k: probability of staying in the current trap (more jitter
            steps) vs. 1 - k stepping to a neighboring pore.
        :param start_node: PNM node id to start the walk from. Must have at
            least one throat. Defaults to a random node with a throat.
        """
        if pnm.number_of_edges() == 0:
            raise ValueError(
                "PNM graph has no throats to walk on (all pores are "
                "isolated after filtering) — check the PNM prefix and "
                "min_radius"
            )
        self.pnm = pnm
        self.bs_ps = bs_ps
        self.p = p
        self.k = k
        self.bs_dir = BufferedSampler(UnitVector3(), "dir", size=100_000)
        self.bs_el = BufferedSampler(Uniform01(), "el", size=100_000)
        self.with_history = with_history

        if start_node is None:
            candidates = [n for n in pnm.nodes if pnm.degree(n) > 0]
            start_node = random.choice(candidates)
        elif pnm.degree(start_node) == 0:
            raise ValueError(
                f"start_node={start_node} has no throats to walk on"
            )
        self.start_node = start_node

    def run(self, count_points: int) -> Trajectory:
        traps = np.zeros(shape=(count_points - 1,), dtype=np.bool_)
        points = np.zeros(shape=(count_points, 3), dtype=f32)

        cur_pos_ind = 0
        cur_trap_ind = self.start_node

        visited: Set[int] = {cur_trap_ind}
        walk_pos: Dict[int, NPFArray] = {cur_trap_ind: np.zeros(3, dtype=f32)}

        def move_towards(
            cur_pos_ind: int, cur_trap_ind: int, node_num: int
        ) -> Tuple[int, int]:
            if node_num not in visited:
                length = self.pnm[cur_trap_ind][node_num]["weight"]
                dir = self.bs_dir.get()
                pos = points[cur_pos_ind, :] + dir * length
                points[cur_pos_ind + 1, :] = pos
                visited.add(node_num)
                walk_pos[node_num] = pos
            else:
                size = self.pnm.nodes[node_num]["size"]
                trap_pos = walk_pos[node_num]
                new_pos = KerogenWalkSimulator.gen_new_pos(1, size, trap_pos)
                points[cur_pos_ind + 1, :] = new_pos
            traps[cur_pos_ind] = False
            return cur_pos_ind + 1, node_num

        def move_next(cur_pos_ind: int, cur_trap_ind: int) -> Tuple[int, int]:
            neighbors: List[int] = list(self.pnm.neighbors(cur_trap_ind))
            unvisited = [n for n in neighbors if n not in visited]
            node_num = random.choice(unvisited if unvisited else neighbors)
            return move_towards(cur_pos_ind, cur_trap_ind, node_num)

        def move_to_adjacent(
            cur_pos_ind: int, cur_trap_ind: int
        ) -> Tuple[int, int]:
            visited_neighbors: List[int] = [
                n for n in self.pnm.neighbors(cur_trap_ind) if n in visited
            ]
            node_num = random.choice(visited_neighbors)
            return move_towards(cur_pos_ind, cur_trap_ind, node_num)

        def steps_inside(
            cur_pos_ind: int,
            cur_trap_ind: int,
            ps: BufferedSampler,
            max_count_steps: int,
        ) -> Tuple[int, int]:
            count_steps = int(ps.get())
            if count_steps == 0:
                return cur_pos_ind, cur_trap_ind
            count_steps = min(max_count_steps - cur_pos_ind - 1, count_steps)
            size = self.pnm.nodes[cur_trap_ind]["size"]
            trap_pos = walk_pos[cur_trap_ind]

            new_pos = KerogenWalkSimulator.gen_new_pos(
                count_steps, size, trap_pos
            )
            points[(cur_pos_ind + 1) : (cur_pos_ind + count_steps + 1)] = (
                new_pos
            )
            traps[cur_pos_ind : (cur_pos_ind + count_steps)] = True
            return cur_pos_ind + count_steps, cur_trap_ind

        count_history_steps = 10 if self.with_history else 1
        for _ in range(count_history_steps):
            cur_pos_ind, cur_trap_ind = steps_inside(
                cur_pos_ind, cur_trap_ind, self.bs_ps, count_points
            )
            if cur_pos_ind + 1 == count_points:
                cur_pos_ind = 0
                points[0, :] = points[-1, :]
                points[1:, :] = 0.0
                traps[:] = False

            cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)

        cur_pos_ind, cur_trap_ind = steps_inside(
            cur_pos_ind, cur_trap_ind, self.bs_ps, count_points
        )

        point_start_ind = cur_pos_ind

        tmp_points = points
        points = np.zeros(shape=(cur_pos_ind + count_points, 3), dtype=f32)
        points[:count_points, :] = tmp_points

        tmp_traps = traps
        traps = np.zeros(
            shape=(cur_pos_ind + count_points - 1,), dtype=np.bool_
        )
        traps[: (count_points - 1)] = tmp_traps

        links_start_ind = cur_pos_ind
        max_count_points = count_points + point_start_ind

        while cur_pos_ind + 1 < max_count_points:
            if self.bs_el.get() <= self.p:
                cur_pos_ind, cur_trap_ind = move_to_adjacent(
                    cur_pos_ind, cur_trap_ind
                )
            else:
                cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)

            if cur_pos_ind >= max_count_points:
                break

            if self.bs_el.get() <= self.k:
                cur_pos_ind, cur_trap_ind = steps_inside(
                    cur_pos_ind, cur_trap_ind, self.bs_ps, max_count_points
                )

        points = points[point_start_ind:, :]
        traps = traps[links_start_ind:]
        mmin = points.min(axis=0) - 1e6
        mmax = points.max(axis=0) + 1e6
        df = mmax - mmin
        bbox = BoundingBox(
            *tuple(Range(k - f, mx + f) for k, mx, f in zip(mmin, mmax, df))
        )

        return Trajectory(
            points, np.arange(count_points, dtype=f32), bbox, traps=traps
        )

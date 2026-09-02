from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from base.boundingbox import BoundingBox, Range
from utils.types import NPBArray, NPFArray

_CACHED_PROPERTY_NAMES = (
    "points_without_periodic",
    "count_points",
    "delta_time",
    "delta_time_sec",
)


@dataclass
class Trajectory:
    points: NPFArray  # coords of atom position, nm for .gro input
    times: NPFArray
    box: BoundingBox  # its box of cell
    atom_size: float = 0.19
    traps: Optional[NPBArray] = None
    # non_periodic_points: Optional[npt.NDArray[np.float64]] = None
    start_dist: float = 0.0

    def _invalidate_cached_properties(self) -> None:
        """Drop memoized `@cached_property` values after `points`/`times`
        change in place, so they are recomputed from the new arrays instead
        of silently returning stale results computed before `cut()`."""
        for name in _CACHED_PROPERTY_NAMES:
            self.__dict__.pop(name, None)

    def cut(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        save_dists: bool = False,
    ):
        if stop is None:
            stop = len(self.points)

        if save_dists:
            dist = Trajectory.extractDists(self.points_without_periodic)
            self.start_dist = dist[start]

        start = max(start, 0)
        stop = min(stop, len(self.points))
        self.points = self.points[start:stop]
        self.times = self.times[start:stop]
        if self.traps is not None:
            self.traps = self.traps[start:stop]
        self._invalidate_cached_properties()

    def dists(self) -> NPFArray:
        dist = Trajectory.extractDists(self.points_without_periodic)
        return dist + self.start_dist

    def is_intersect_borders(self) -> np.bool_:
        ppoints = self.points_without_periodic
        xmask = np.logical_or(
            ppoints[:, 0] > self.box.xb_.max_, ppoints[:, 0] < 0
        )
        ymask = np.logical_or(
            ppoints[:, 1] > self.box.yb_.max_, ppoints[:, 1] < 0
        )
        zmask = np.logical_or(
            ppoints[:, 2] > self.box.zb_.max_, ppoints[:, 2] < 0
        )
        return np.any(xmask) or np.any(ymask) or np.any(zmask)

    def trjbox(self, periodic: bool) -> BoundingBox:
        ppoints = self.points_without_periodic if not periodic else self.points
        mmin = ppoints.min(axis=0)
        mmax = ppoints.max(axis=0)
        tmp = [Range(i, j) for i, j in zip(mmin, mmax)]
        return BoundingBox(*tmp)

    @staticmethod
    def extractDists(
        points: NPFArray,
    ) -> NPFArray:
        diff = points[1:,] - points[:-1,]
        sq_diff = diff * diff
        sq_dist = np.sum(sq_diff, axis=1)
        dist = np.sqrt(sq_dist)
        return np.array(dist, dtype=np.float32)

    @cached_property
    def points_without_periodic(self) -> npt.NDArray[np.float32]:
        borders = self.box.max()

        npoints = np.zeros(shape=self.points.shape, dtype=np.float32)
        npoints[0, :] = self.points[0, :]

        shift = np.zeros(shape=(3,), dtype=np.float32)
        for i in range(1, self.points.shape[0]):
            prev_pos = self.points[i - 1, :]
            next_pos = self.points[i, :]

            for a in range(3):
                delta = next_pos[a] - prev_pos[a]
                b = np.array([0, borders[a]])
                s = delta - b * (-1 if delta < 0 else 1)
                shift[a] = s[np.argmin(np.abs(s))]

            npoints[i, :] = npoints[i - 1, :] + shift

        return npoints

    @cached_property
    def count_points(self) -> int:
        return int(self.points.shape[0])

    @cached_property
    def delta_time(self) -> float:
        """time step

        Returns:
            float: picoseconds
        """
        assert len(self.times) >= 2
        return float(self.times[1] - self.times[0])

    @cached_property
    def delta_time_sec(self) -> float:
        """time step

        Returns:
            float: seconds
        """
        assert len(self.times) >= 2
        return self.delta_time * 1e-12

    @staticmethod
    def read_trajectoryes(file_name: str | Path) -> List['Trajectory']:
        ax = []
        ay = []
        az = []

        time_steps = []
        with open(file_name) as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                time_start = line.find('t=')
                # step_start = line.find('step=')
                t = line[(time_start + 2) : (time_start + 12)]
                # step = line[step_start:]
                count = int(f.readline())

                time_steps.append(float(t))

                for i in range(count):
                    line = f.readline()
                    x, y, z = line[20:28], line[28:36], line[36:44]
                    ax.append(float(x))
                    ay.append(float(y))
                    az.append(float(z))
                if len(ax) == count:
                    line = f.readline()
                    box = BoundingBox(
                        Range(0, float(line[:10])),
                        Range(0, float(line[10:20])),
                        Range(0, float(line[20:])),
                    )
                else:
                    next(f)

        count_step = int(len(ax) / count)
        # count_step = count_step // 2
        trajectories = []
        for i in range(count):
            points = np.zeros(shape=(count_step, 3), dtype=np.float32)
            points[:, 0] = [ax[i + j * count] for j in range(count_step)]
            points[:, 1] = [ay[i + j * count] for j in range(count_step)]
            points[:, 2] = [az[i + j * count] for j in range(count_step)]

            trajectories.append(Trajectory(points, np.array(time_steps), box))

        return trajectories

    @staticmethod
    def read_trajectories_fast(file_name: str | Path) -> List['Trajectory']:
        frames = []
        time_steps = []
        box = None
        count = None

        with open(file_name, "rb") as f:
            while True:
                header = f.readline()

                if not header:
                    break

                if b"t=" not in header:
                    raise RuntimeError(f"Unexpected frame header: {header!r}")

                t = float(header.split(b"t=", 1)[1].split()[0])
                time_steps.append(t)

                count_line = f.readline()
                if not count_line:
                    raise RuntimeError("Unexpected EOF after header")

                current_count = int(count_line)

                if count is None:
                    count = current_count
                elif current_count != count:
                    raise RuntimeError(
                        f"Atom count changed between frames: {count} -> {current_count}"
                    )

                atom_lines = [f.readline() for _ in range(count)]

                coord_block = b" ".join(line[20:44] for line in atom_lines)

                xyz = np.fromstring(
                    coord_block,
                    sep=" ",
                    dtype=np.float32,
                )

                expected_size = count * 3
                if xyz.size != expected_size:
                    raise RuntimeError(
                        f"Failed to parse coordinates: got {xyz.size}, expected {expected_size}"
                    )

                xyz = xyz.reshape(count, 3)
                frames.append(xyz)

                box_line = f.readline()
                if not box_line:
                    raise RuntimeError("Unexpected EOF while reading box line")

                if box is None:
                    box_values = np.fromstring(
                        box_line, sep=" ", dtype=np.float32
                    )

                    if box_values.size < 3:
                        raise RuntimeError(
                            f"Cannot parse box line: {box_line!r}"
                        )

                    box = BoundingBox(
                        Range(0, float(box_values[0])),
                        Range(0, float(box_values[1])),
                        Range(0, float(box_values[2])),
                    )

        coords = np.stack(frames, axis=0)
        # shape: (count_step, count_atoms, 3)

        time_steps_arr = np.array(time_steps, dtype=np.float32)
        count_atoms = coords.shape[1]
        assert box is not None

        trajectories = []

        for i in range(count_atoms):
            points = coords[:, i, :]

            # Если дальше код меняет points inplace, лучше сделать copy:
            # points = coords[:, i, :].copy()

            trajectories.append(Trajectory(points, time_steps_arr, box))

        return trajectories

    def msd(self):
        return Trajectory.msd_by_points(self.points_without_periodic)

    def msd_average_time(self):
        points = self.points_without_periodic
        n = points.shape[0]

        msd = np.zeros(n, dtype=np.float64)

        for lag in range(n):
            dr = points[lag:] - points[: n - lag]
            sq_disp = np.sum(dr * dr, axis=1)
            msd[lag] = np.mean(sq_disp)

        return msd.astype(np.float32)

    @staticmethod
    def msd_by_points(points):
        p = points - points[0, :]
        return p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2

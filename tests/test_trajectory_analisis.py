import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.io import loadmat

from base.trajectory import Trajectory
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)


@pytest.fixture
def test_path() -> Path:
    return Path(__file__).parent


@pytest.fixture
def trajectory(test_path: Path) -> Trajectory:
    with (test_path / "trajectory.pickle").open("rb") as file:
        return pickle.load(file)  # type: ignore[no-any-return]


@pytest.fixture
def expected_result(test_path: Path) -> npt.NDArray[np.bool_]:
    matlab_res = loadmat(test_path / "result_list_trapped.mat")
    return np.asarray(matlab_res["list_trapped"][0], dtype=np.bool_)


def _assert_dm_regression(
    trajectory: Trajectory,
    expected_point_labels: npt.NDArray[np.bool_],
    params: DistanceMatrixParams,
) -> None:
    assert expected_point_labels.shape == (trajectory.count_points,)
    traps_before = trajectory.traps

    actual = DistanceMatrixAnalyzer(params).run(trajectory)

    assert actual.shape == (trajectory.count_points - 1,)
    assert actual.dtype == np.bool_
    assert trajectory.traps is traps_before
    np.testing.assert_array_equal(actual, expected_point_labels[1:])


def test_dm_regression_against_matlab_fixture(
    trajectory: Trajectory, expected_result: npt.NDArray[np.bool_]
) -> None:
    # The MATLAB reference was calculated without smoothing. The historical
    # Python implementation declared kernel_size=2 by default but did not
    # apply that kernel, so kernel_size=0 states the reference calculation
    # explicitly instead of relying on the obsolete default.
    _assert_dm_regression(
        trajectory,
        expected_result,
        DistanceMatrixParams(kernel_size=0),
    )

import csv
import json
from pathlib import Path

import numpy as np

from scripts.stationarity import (
    KSTestResult,
    holm_correction,
    save_stationarity_summary,
    stationarity_summary_record,
)
from scripts.structure_image_utils import (
    StepTimeMapping,
    infer_step_time_mapping,
    resolve_step_time_mapping,
)


def _gro_frame(time_ps: float, step: int) -> str:
    return (
        f"Kerogen t= {time_ps:.5f} step= {step}\n"
        "1\n"
        "    1KRG      C    1   0.000   0.000   0.000\n"
        "   1.00000   1.00000   1.00000\n"
    )


def test_holm_correction_known_values() -> None:
    reject, adjusted = holm_correction(np.array([0.5, 0.01, 0.02]))

    np.testing.assert_allclose(adjusted, [0.5, 0.03, 0.04])
    np.testing.assert_array_equal(reject, [False, True, True])


def test_holm_correction_empty_input() -> None:
    reject, adjusted = holm_correction(np.array([]))

    assert reject.dtype == np.bool_
    assert adjusted.dtype == np.float64
    assert reject.size == adjusted.size == 0


def test_step_time_mapping_is_inferred_from_two_gro_frames(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trj.gro"
    trajectory.write_text(
        _gro_frame(12.5, 100) + _gro_frame(13.25, 130),
        encoding="utf-8",
    )

    mapping = infer_step_time_mapping(trajectory)
    overridden = resolve_step_time_mapping(
        trajectory,
        time_delta_ps=1.5,
    )

    assert mapping == StepTimeMapping(100, 12.5, 30, 0.75)
    assert mapping.time_ps(160) == 14.0
    assert overridden == StepTimeMapping(100, 12.5, 30, 1.5)


def test_explicit_step_time_mapping_does_not_require_trajectory() -> None:
    mapping = resolve_step_time_mapping(
        Path("does-not-exist.gro"),
        anchor_step=100,
        anchor_time_ps=12.5,
        step_delta=30,
        time_delta_ps=0.75,
    )

    assert mapping.time_ps(160) == 14.0


def test_table_iv_summary_is_saved_as_csv_and_json(tmp_path: Path) -> None:
    result = KSTestResult(
        pairs=[(0, 1), (0, 2)],
        times=np.array([1.5, 2.5, 3.5]),
        D=np.array([0.1, 0.2]),
        p=np.array([0.01, 0.5]),
        p_adj=np.array([0.02, 0.5]),
        reject=np.array([True, False]),
    )
    record = stationarity_summary_record(
        result,
        distribution="P(h)",
        comparison="Baseline",
    )

    csv_path, json_path = save_stationarity_summary([record], tmp_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        csv_records = list(csv.DictReader(file))
    json_records = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_records[0]["n_rejections"] == "1"
    assert json_records == [record]
    assert json_records[0]["D_max"] == 0.2

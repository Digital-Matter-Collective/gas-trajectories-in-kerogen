import json
from pathlib import Path

import numpy as np
import pytest

import scripts.pil_plotter as pil_plotter


class _StopAfterFit(Exception):
    """Raised from a monkeypatched `exponweib.fit` to short-circuit the
    rest of `plot_distributions` (heatmap generation/plotting), which is
    expensive and irrelevant to the `--fit-sample-stride` behavior under
    test."""


def _make_data_dir(tmp_path: Path, n_radiuses: int = 1000) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pnm_distribution_units.json").write_text(
        json.dumps({"length_unit": "nm"})
    )
    radiuses = np.linspace(0.05, 2.0, n_radiuses).astype(np.float64)
    np.save(data_dir / "radiuses.npy", radiuses)
    return data_dir


def test_fit_sample_stride_subsamples_radii_before_fitting(
    tmp_path, monkeypatch
) -> None:
    data_dir = _make_data_dir(tmp_path, n_radiuses=1000)
    captured = {}

    def fake_fit(sample):
        captured["size"] = len(sample)
        raise _StopAfterFit

    monkeypatch.setattr(pil_plotter.exponweib, "fit", fake_fit)

    with pytest.raises(_StopAfterFit):
        pil_plotter.plot_distributions(
            str(data_dir),
            radius_min=0.0,
            r_points=10,
            r_step=1,
            l_bins=10,
            heatmap_mode="smooth",
            heatmap_interpolation="bicubic",
            heatmap_dpi=100,
            fit_sample_stride=25,
        )

    assert captured["size"] == len(np.load(data_dir / "radiuses.npy")[::25])
    assert captured["size"] == 40


def test_fit_sample_stride_default_is_ten(tmp_path, monkeypatch) -> None:
    data_dir = _make_data_dir(tmp_path, n_radiuses=1000)
    captured = {}

    def fake_fit(sample):
        captured["size"] = len(sample)
        raise _StopAfterFit

    monkeypatch.setattr(pil_plotter.exponweib, "fit", fake_fit)

    with pytest.raises(_StopAfterFit):
        pil_plotter.plot_distributions(
            str(data_dir),
            radius_min=0.0,
            r_points=10,
            r_step=1,
            l_bins=10,
            heatmap_mode="smooth",
            heatmap_interpolation="bicubic",
            heatmap_dpi=100,
        )

    assert captured["size"] == 100


def test_fit_sample_stride_rejects_non_positive_values(tmp_path) -> None:
    data_dir = _make_data_dir(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        pil_plotter.plot_distributions(
            str(data_dir),
            radius_min=0.0,
            r_points=10,
            r_step=1,
            l_bins=10,
            heatmap_mode="smooth",
            heatmap_interpolation="bicubic",
            heatmap_dpi=100,
            fit_sample_stride=0,
        )

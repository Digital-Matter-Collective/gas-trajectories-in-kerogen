from pathlib import Path

import numpy as np
import vtk

from base.boundingbox import BoundingBox, Range
from visualizer.visualizer import Visualizer


def _make_bbox() -> BoundingBox:
    return BoundingBox(Range(0, 10), Range(0, 10), Range(0, 10))


def _make_float_img() -> np.ndarray:
    img = np.zeros((6, 6, 6), dtype=np.float32)
    img[2:4, 2:4, 2:4] = 1.0
    return img


# --- create_img_data dtype check (isinstance bug) ---------------------------


def test_create_img_data_selects_int_scalars_for_uint8_images() -> None:
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    image_data = Visualizer.create_img_data(img, bbox=None)
    assert image_data.GetScalarType() == vtk.VTK_INT


def test_create_img_data_selects_float_scalars_for_float_images() -> None:
    img = _make_float_img()
    image_data = Visualizer.create_img_data(img, bbox=None)
    assert image_data.GetScalarType() == vtk.VTK_FLOAT


# --- draw_img / add_img_actor call contract (P1-10) --------------------------


def test_draw_img_offscreen_does_not_raise_and_writes_a_file(
    tmp_path: Path,
) -> None:
    img = np.zeros((6, 6, 6), dtype=np.int8)
    img[2:4, 2:4, 2:4] = 1
    output = tmp_path / "render.png"

    Visualizer.draw_img(
        img,
        volume_mode=True,
        bbox=_make_bbox(),
        output_path=str(output),
        window_size=(64, 64),
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_float_img_offscreen_does_not_raise_and_writes_a_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "render.png"

    Visualizer.draw_float_img(
        _make_float_img(),
        bbox=_make_bbox(),
        output_path=str(output),
        window_size=(64, 64),
        isovalue=0.5,
        img_opacity=0.5,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_save_offscreen_render_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    import pytest

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    with pytest.raises(ValueError, match="Unsupported offscreen"):
        Visualizer.save_offscreen_render(renWin, str(tmp_path / "out.svg"))


# --- draw_graph colors_data/nscales NameError (P1-10) -------------------------


def test_draw_graph_without_colors_data_kwargs_does_not_raise() -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, color_id=0, scale_id=0)
    graph.add_node(1, color_id=0, scale_id=0)
    graph.add_edge(0, 1)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    corr = {0: 0, 1: 1}

    renderer = vtk.vtkRenderer()
    # No 'colors_data'/'scales_data' kwargs: used to raise NameError for
    # `colors_data`/`nscales` before both were referenced further down.
    Visualizer.draw_graph(renderer, (graph, positions, corr))


def test_draw_graph_with_colors_data_kwargs_does_not_raise() -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, color_id=0, scale_id=0)
    graph.add_node(1, color_id=1, scale_id=0)
    graph.add_edge(0, 1)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    corr = {0: 0, 1: 1}

    renderer = vtk.vtkRenderer()
    Visualizer.draw_graph(
        renderer,
        (graph, positions, corr),
        colors_data={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)},
        scales_data={0: 0.5},
    )

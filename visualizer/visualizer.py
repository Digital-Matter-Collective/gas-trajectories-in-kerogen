from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import vtk
from skimage import measure
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkCommand, vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersSources import (
    vtkCubeSource,
    vtkSphereSource,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkTIFFWriter,
)
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer

from base.boundingbox import BoundingBox
from base.trajectory import Trajectory
from utils.utils import kprint
from visualizer.timer_callback_camera import vtkTimerCallbackCamera
from visualizer.win_struct_collection import WinStructCollection


def visualize_trajectory(
    traj: Trajectory, color_type: str = 'dist', win_name: str = ""
) -> None:
    Visualizer.draw_trajectories(
        [traj],
        color_type=color_type,
        wrap_mode=WrapMode.EMPTY,
        window_name=win_name,
        with_points=True,
        radius=0.1,
    )


class WrapMode(Enum):
    EMPTY = 0
    BOX = 1
    AXES = 2


collection: List[WinStructCollection] = []

# Off-screen image writers keyed by output file extension, used by draw
# functions that support a non-interactive `output_path` render (P1-10).
_OFFSCREEN_IMAGE_WRITERS = {
    ".png": vtkPNGWriter,
    ".jpg": vtkJPEGWriter,
    ".jpeg": vtkJPEGWriter,
    ".tif": vtkTIFFWriter,
    ".tiff": vtkTIFFWriter,
    ".bmp": vtkBMPWriter,
}


class Visualizer:

    @staticmethod
    def draw_graph_and_img(
        G: Any,
        img: npt.NDArray[Any],
        node_pos_corr: Any,
        bbox: BoundingBox,
        size_node: float = 0.25,
        size_edge: float = 0.02,
        save_pos_path: str = '',
        scale: str = "full_by_1",
        plot_box: bool = True,
        **kwargs: Any,
    ) -> None:
        renderer = vtkRenderer()

        positions = None
        corr = None
        if isinstance(node_pos_corr, dict):
            nums = np.array(list(node_pos_corr.keys()), dtype=int)
            positions = np.zeros(shape=(nums.shape[0], 3), dtype=float)
            corr = np.zeros(shape=(nums.max() + 1,), dtype=int)
            i = 0
            for k in node_pos_corr.keys():
                positions[i, :] = node_pos_corr[k]
                corr[int(k)] = i
                i = i + 1
        else:
            positions = node_pos_corr[0]
            corr = node_pos_corr[1]

        data_for_vis = (G, positions, corr)

        focal_pos, camera_pos = Visualizer.draw_graph(
            renderer,
            data_for_vis,
            size_node,
            size_edge,
            save_pos_path,
            scale,
            **kwargs,
        )

        Visualizer.add_img_actor(
            renderer,
            img,
            bbox,
            **kwargs,
        )

        if plot_box:
            outfit_actor = Visualizer.create_box_actor(bbox)
            renderer.AddActor(outfit_actor)

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)

        style = vtkInteractorStyleTrackballCamera()
        # style = vtkInteractorStyleFlight()
        # style = vtkInteractorStyleTrackballActor()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        # Add the actors

        camera_y_shift = 3.0e-10  # подбери значение в единицах твоей сцены
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_pos[0], focal_pos[1], focal_pos[2])
        camera.SetPosition(
            camera_pos[0], camera_pos[1] - camera_y_shift, camera_pos[2]
        )
        # renWin.SetSize(640, 640)

        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver(vtkCommand.TimerEvent, cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        # renWin.FullScreenOn()
        renWin.SetSize(1900, 1080)
        iren.Start()

    @staticmethod
    def draw_nxvtk(
        G: Any,
        node_pos_corr: Any,
        size_node: float = 0.25,
        size_edge: float = 0.02,
        save_pos_path: str = '',
        scale: str = "full_by_1",
        **kwargs: Any,
    ) -> None:
        """
        Draw networkx graph in 3d with nodes at node_pos.

        See layout.py for functions that compute node positions.

        node_pos is a dictionary keyed by vertex with a three-tuple
        of x-y positions as the value.

        The node color is plum.
        The edge color is banana.

        All the nodes are the same size.

        @todo to enumerate
        Scale: full_by_1, one_ax_by_1, no

        """

        # Now create the RenderWindow, Renderer and Interactor
        renderer = vtkRenderer()

        positions = None
        corr = None
        if isinstance(node_pos_corr, dict):
            nums = np.array(list(node_pos_corr.keys()), dtype=int)
            positions = np.zeros(shape=(nums.shape[0], 3), dtype=float)
            corr = np.zeros(shape=(nums.max() + 1,), dtype=int)
            i = 0
            for k in node_pos_corr.keys():
                positions[i, :] = node_pos_corr[k]
                corr[int(k)] = i
                i = i + 1
        else:
            positions = node_pos_corr[0]
            corr = node_pos_corr[1]

        data_for_vis = (G, positions, corr)

        focal_pos, camera_pos = Visualizer.draw_graph(
            renderer,
            data_for_vis,
            size_node,
            size_edge,
            save_pos_path,
            scale,
            **kwargs,
        )

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)

        style = vtkInteractorStyleTrackballCamera()
        # style = vtkInteractorStyleFlight()
        # style = vtkInteractorStyleTrackballActor()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        # Add the actors

        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(focal_pos[0], focal_pos[1], focal_pos[2])
        camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
        # renWin.SetSize(640, 640)

        renWin.Render()
        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver(vtkCommand.TimerEvent, cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        # renWin.FullScreenOn()
        renWin.SetSize(1900, 1080)
        iren.Start()

    @staticmethod
    def draw_graph(
        ren: vtkRenderer,
        graph_pos_corr: Any,
        size_node: float = 0.25,
        size_edge: float = 0.02,
        save_pos_path: str = '',
        scale: str = "full_by_1",
        **kwargs: Any,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mrange = 1e2
        i = 0

        ndcolors = None
        tdcolors = None
        # `colors_data`/`nscales` must be bound on every path below: they
        # are read unconditionally further down (P1-10), but were only
        # ever assigned inside the nested branch that requires both
        # `colors_data` and `scales_data` kwargs.
        colors_data = kwargs.get('colors_data')
        nscales = None
        if len(graph_pos_corr) == 3:
            graph, positions, corr = graph_pos_corr
            if colors_data is not None and 'scales_data' in kwargs:
                scales_data = kwargs['scales_data']
                ndcolors = [n[1]["color_id"] for n in graph.nodes(data=True)]
                nscales = [
                    scales_data[n[1]["scale_id"]]
                    for n in graph.nodes(data=True)
                ]

        else:
            graph, positions, corr, ndcolors, tdcolors = graph_pos_corr

        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        dmax = diff.max()

        if scale == "full_by_1" or scale == "one_ax_by_1":
            if scale == "full_by_1":
                for i in range(3):
                    positions[:, i] = (
                        positions[:, i] - a_min[i]
                    ) * 2 * mrange / diff[i] - mrange
            elif scale == "one_ax_by_1":
                for i in range(3):
                    d = (positions[:, i] - a_min[i]) / dmax
                    positions[:, i] = ((d - d.max() / 2)) * mrange

        kprint(f"min = {positions.min(axis=0)}, max = {positions.max(axis=0)}")

        if len(save_pos_path) != 0:
            with open(save_pos_path[0], 'wb') as f:
                np.save(f, positions)
            with open(save_pos_path[1], 'wb') as f:
                np.save(f, corr)

        # set node positions
        colors = vtkNamedColors()
        nodePoints = vtkPoints()
        color_transfer = None
        if colors_data is not None:
            color_transfer = vtkColorTransferFunction()
            for cd, color in colors_data.items():
                color_transfer.AddRGBPoint(cd, color[0], color[1], color[2])

        i = 0
        count_nodes = positions.shape[0]
        pore_data = vtkDoubleArray()
        pore_data.SetNumberOfValues(count_nodes)
        pore_data.SetName("color_data")

        scales = vtkDoubleArray()
        scales.SetNumberOfValues(count_nodes)
        scales.SetName("scales")
        for x, y, z in positions:
            nodePoints.InsertPoint(i, x, y, z)
            if ndcolors is not None:
                pore_data.SetValue(i, ndcolors[i])
            if nscales is not None:
                scales.SetValue(i, nscales[i])
            i = i + 1

        # Create a polydata to be glyphed.
        inputData = vtkPolyData()
        inputData.SetPoints(nodePoints)
        inputData.GetPointData().AddArray(pore_data)
        inputData.GetPointData().AddArray(scales)
        inputData.GetPointData().SetActiveScalars(scales.GetName())

        # Use sphere as glyph source.
        balls = vtkSphereSource()
        balls.SetRadius(size_node)
        balls.SetPhiResolution(20)
        balls.SetThetaResolution(20)

        glyphPoints = vtkGlyph3D()
        glyphPoints.SetInputData(inputData)
        glyphPoints.SetScaleModeToScaleByScalar()
        glyphPoints.SetSourceConnection(balls.GetOutputPort())

        glyphMapper = vtkPolyDataMapper()
        glyphMapper.SetInputConnection(glyphPoints.GetOutputPort())
        glyphMapper.SetScalarModeToUsePointFieldData()
        glyphMapper.SelectColorArray(pore_data.GetName())
        glyphMapper.SetLookupTable(color_transfer)
        glyphMapper.Update()

        glyph = vtkActor()
        glyph.SetMapper(glyphMapper)
        glyph.GetProperty().SetDiffuseColor(0.36, 0.16, 0.53)
        glyph.GetProperty().SetSpecular(0.3)
        glyph.GetProperty().SetSpecularPower(30)

        # Generate the polyline for the spline.
        points = vtkPoints()
        edgeData = vtkPolyData()

        # Edges

        lines = vtkCellArray()
        i = 0
        for u, v in graph.edges():
            # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
            lines.InsertNextCell(2)
            for n in (u, v):
                ni = corr[int(n)]
                x, y, z = positions[ni, :]
                points.InsertPoint(i, x, y, z)
                lines.InsertCellPoint(i)
                i = i + 1

        edgeData.SetPoints(points)
        edgeData.SetLines(lines)

        # Add thickness to the resulting line.
        Tubes = vtkTubeFilter()
        Tubes.SetNumberOfSides(16)
        Tubes.SetInputData(edgeData)
        Tubes.SetRadius(size_edge)
        # Tubes.SetVaryRadiusToVaryRadiusByScalar()
        # Tubes.SetRadiusFactor(size_edge)
        # Tubes.SetRadiusFactor(1.0)
        #
        throat_mapper = vtkPolyDataMapper()
        throat_mapper.SetInputConnection(Tubes.GetOutputPort())

        colors = vtkNamedColors()

        #
        throat_actor = vtkActor()
        throat_actor.SetMapper(throat_mapper)
        # throat_actor.GetProperty().SetColor(colorsGetColor3d("Blue"))
        throat_actor.GetProperty().SetDiffuseColor(0.18, 0.53, 0.67)
        throat_actor.GetProperty().SetSpecular(0.15)
        throat_actor.GetProperty().SetSpecularPower(20)

        ren.AddActor(glyph)
        ren.AddActor(throat_actor)
        ren.SetBackground(colors.GetColor3d("White"))

        mid = positions.mean(axis=0)
        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        return mid, mid + 2 * diff

    @staticmethod
    def save_offscreen_render(
        renWin: vtkRenderWindow,
        output_path: str,
        window_size: Tuple[int, int] = (1900, 1080),
    ) -> None:
        """Render `renWin` off-screen at a fixed size and save it to a file.

        For publication figures that need a non-interactive, reproducible
        render instead of an interactive VTK window with an unrecorded
        camera/resolution.
        """
        suffix = Path(output_path).suffix.lower()
        writer_cls = _OFFSCREEN_IMAGE_WRITERS.get(suffix)
        if writer_cls is None:
            raise ValueError(
                f"Unsupported offscreen output extension: {suffix!r}; "
                f"use one of {sorted(_OFFSCREEN_IMAGE_WRITERS)}"
            )

        renWin.SetOffScreenRendering(1)
        renWin.SetSize(*window_size)
        renWin.Render()

        to_image = vtkWindowToImageFilter()
        to_image.SetInput(renWin)
        to_image.Update()

        writer = writer_cls()
        writer.SetFileName(str(output_path))
        writer.SetInputConnection(to_image.GetOutputPort())
        writer.Write()

    @staticmethod
    def create_img_data(
        img: npt.NDArray[Any], bbox: Optional[BoundingBox]
    ) -> vtk.vtkImageData:
        image_data = vtk.vtkImageData()
        size = img.shape
        image_data.SetDimensions(size[0], size[1], size[2])
        if img.dtype == np.uint8:
            image_data.AllocateScalars(vtk.VTK_INT, 1)
        else:
            image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

        if bbox is not None:
            spacing = np.asarray(bbox.size(), dtype=float) / (
                np.asarray(img.shape, dtype=float) - 1.0
            )
            image_data.SetSpacing(*spacing)
        else:
            image_data.SetSpacing(1, 1, 1)
        for ix in range(size[0]):
            for iy in range(size[1]):
                for iz in range(size[2]):
                    image_data.SetScalarComponentFromDouble(
                        ix, iy, iz, 0, img[ix, iy, iz]
                    )
        return image_data

    @staticmethod
    def create_volume_img(image_data: "vtk.vtkImageData") -> vtk.vtkVolume:
        composite_opacity = vtk.vtkPiecewiseFunction()
        composite_opacity.AddPoint(0.0, 0.3)
        composite_opacity.AddPoint(0.98, 0.3)
        composite_opacity.AddPoint(0.995, 0.5)
        composite_opacity.AddPoint(1.0, 1)

        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(
            1, 205.0 / 255.0, 164.0 / 255.0, 52.0 / 255.0
        )

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(composite_opacity)
        volume_property.ShadeOff()
        # volume_property.SetInterpolationTypeToLinear()
        volume_property.SetInterpolationTypeToNearest()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        return volume

    @staticmethod
    def create_actor_img(
        image_data: "vtk.vtkImageData", **kwargs: Any
    ) -> vtkActor:
        if image_data.GetScalarType() == vtk.VTK_INT:
            marchingcube = vtk.vtkDiscreteFlyingEdges3D()
        else:
            marchingcube = vtk.vtkFlyingEdges3D()
        marchingcube.SetInputData(image_data)
        marchingcube.ComputeNormalsOn()
        marchingcube.ComputeScalarsOn()
        marchingcube.SetNumberOfContours(1)
        marchingcube.SetValue(0, kwargs["isovalue"])

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(1)
        lut.SetTableRange(0, 1)
        lut.SetScaleToLinear()
        lut.Build()
        lut.SetTableValue(0, 0.75, 0.64, 0.48, 1)
        lut.SetTableValue(1, 1, 0, 1, 1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marchingcube.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 2)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(kwargs["img_opacity"])
        return actor

    @staticmethod
    def add_img_actor(
        ren: vtkRenderer,
        img: npt.NDArray[Any],
        bbox: Optional[BoundingBox],
        **kwargs: Any,
    ) -> None:
        image_data = Visualizer.create_img_data(img, bbox)

        mactor = None
        if kwargs["volume_mode"]:
            mactor = Visualizer.create_volume_img(image_data)
            ren.AddVolume(mactor)
        else:
            mactor = Visualizer.create_actor_img(image_data, **kwargs)
            ren.AddActor(mactor)

        if bbox is not None:
            pcenter = bbox.min()
            mactor.SetPosition(*pcenter)

    @staticmethod
    def draw_img(
        img: npt.NDArray[np.int8],
        volume_mode: bool,
        bbox: BoundingBox,
        *,
        output_path: Optional[str] = None,
        window_size: Tuple[int, int] = (1900, 1080),
        **kwargs: Any,
    ) -> None:
        ren = vtkRenderer()

        Visualizer.add_img_actor(
            ren, img, bbox, volume_mode=volume_mode, **kwargs
        )

        colors = vtkNamedColors()
        ren.SetBackground(colors.GetColor3d("White"))

        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)

        # Add the actors
        center = bbox.center()
        cm_pos = center + bbox.size()

        camera = ren.GetActiveCamera()
        # size = img.shape
        camera.SetFocalPoint(*center)
        camera.SetPosition(*cm_pos)

        if output_path is not None:
            Visualizer.save_offscreen_render(renWin, output_path, window_size)
            return

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver(vtkCommand.TimerEvent, cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        # renWin.FullScreenOn()
        renWin.SetSize(*window_size)
        iren.Start()

    @staticmethod
    def draw_float_img(
        img: npt.NDArray[np.float32],
        bbox: BoundingBox,
        *,
        output_path: Optional[str] = None,
        window_size: Tuple[int, int] = (1900, 1080),
        **kwargs: Any,
    ) -> None:
        renderer = vtkOpenGLRenderer()

        Visualizer.add_img_actor(
            renderer, img, bbox, volume_mode=False, **kwargs
        )

        colors = vtkNamedColors()
        renderer.SetBackground(colors.GetColor3d("White"))

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)

        # Add the actors
        center = bbox.center()
        cm_pos = center + bbox.size()

        camera = renderer.GetActiveCamera()
        # size = img.shape
        camera.SetFocalPoint(*center)
        camera.SetPosition(*cm_pos)

        if output_path is not None:
            Visualizer.save_offscreen_render(renWin, output_path, window_size)
            return

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver(vtkCommand.TimerEvent, cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        win_col = WinStructCollection(iren)
        collection.append(win_col)

    @staticmethod
    def create_trajectory_actor(
        trj: Trajectory,
        **kwargs: Any,
    ) -> vtkActor:
        points = (
            trj.points_without_periodic
            if not kwargs["periodic"]
            else trj.points
        )
        if kwargs["color_type"] == 'dist':
            colors = np.cumsum(trj.dists())
            colors = np.append(0, colors)
            colors /= colors[-1]
        elif kwargs["color_type"] == 'clusters':
            assert trj.traps is not None
            clusters = trj.traps

            # colors = ndimage.binary_erosion(clusters).astype(clusters.dtype)
            colors = measure.label(clusters, connectivity=1).astype(np.float32)
            # print(colors.max())
            colors /= colors.max()

        return Visualizer.create_polyline_actor(
            points,
            colors,
            kwargs['radius'],
        )[0]

    @staticmethod
    def draw_trajectories(
        trjs: List[Trajectory],
        color_type: str = 'dist',
        periodic: bool = False,
        wrap_mode: WrapMode = WrapMode.EMPTY,
        with_points: bool = False,
        window_name: str = 'Trajectory',
        radius: float = 1,
    ) -> None:
        # renderer = vtkRenderer()
        renderer = vtkOpenGLRenderer()

        bbox = BoundingBox()
        for trj in trjs:
            actor = Visualizer.create_trajectory_actor(
                trj,
                periodic=periodic,
                color_type=color_type,
                radius=radius * 0.75,
            )
            trjbox: BoundingBox = trj.trjbox(periodic)
            bbox.update_by_box(trjbox)
            renderer.AddActor(actor)
            if with_points:
                actor = Visualizer.create_trj_points_actor(
                    trj,
                    periodic=periodic,
                    color_type=color_type,
                    radius=radius,
                )
                renderer.AddActor(actor)

        colors = vtkNamedColors()

        if wrap_mode == WrapMode.BOX:
            outfit_actor = Visualizer.create_box_actor(trjs[0].box)
            renderer.AddActor(outfit_actor)
        elif wrap_mode == WrapMode.AXES:
            actor = Visualizer.create_axes_actor(
                bbox, renderer.GetActiveCamera()
            )
            renderer.AddActor(actor)
        else:
            assert wrap_mode == WrapMode.EMPTY

        renderer.SetBackground(colors.GetColor3d("White"))
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(135)

        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(renderer)

        ren_win.SetSize(640, 512)
        ren_win.SetWindowName(window_name)

        style = vtkInteractorStyleTrackballCamera()

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        iren.SetInteractorStyle(style)

        irradiance = renderer.GetEnvMapIrradiance()
        irradiance.SetIrradianceStep(0.3)

        win_col = WinStructCollection(iren)
        collection.append(win_col)

    @staticmethod
    def show() -> None:
        if len(collection) == 0:
            return
        for col in collection:
            col.interactor.Initialize()
        cont_flag = True
        while cont_flag:
            to_rm = []
            for i, col in enumerate(collection):
                col.running = col.kpis.status
                if col.running:
                    col.interactor.ProcessEvents()
                    col.interactor.Render()
                else:
                    to_rm.append((col, col.renWin.GetWindowName()))
            for col, win_name in to_rm:
                col.clear()
                collection.remove(col)
                kprint('Window', win_name, 'has stopped running.')
            if len(collection) == 0:
                break
            cont_flag = all(x.running is True for x in collection)

    @staticmethod
    def create_box_actor(box: BoundingBox) -> vtkActor:
        cube = vtkCubeSource()
        cube.SetBounds(
            box.xb_.min_,
            box.xb_.max_,
            box.yb_.min_,
            box.yb_.max_,
            box.zb_.min_,
            box.zb_.max_,
        )

        outline = vtkOutlineFilter()
        outline.SetInputConnection(cube.GetOutputPort())

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        actor.GetProperty().SetLineWidth(3.0)
        actor.GetProperty().LightingOff()

        return actor

    @staticmethod
    def create_axes_actor(bbox: BoundingBox, camera: vtkCamera) -> vtkActor:
        colors = vtkNamedColors()
        data = colors.GetColor3d("Black")

        axes = vtkCubeAxesActor()
        axes.SetUseTextActor3D(True)
        axes.SetBounds(bbox.aminmax())
        axes.SetCamera(camera)
        axes.GetXAxesLinesProperty().SetColor(data)
        axes.GetYAxesLinesProperty().SetColor(data)
        axes.GetZAxesLinesProperty().SetColor(data)
        axes.GetXAxesGridlinesProperty().SetColor(data)
        axes.GetYAxesGridlinesProperty().SetColor(data)
        axes.GetZAxesGridlinesProperty().SetColor(data)

        # axes.XAxisLabelVisibilityOff()
        # axes.YAxisLabelVisibilityOff()
        # axes.ZAxisLabelVisibilityOff()

        axes.SetXTitle("")
        axes.SetYTitle("")
        axes.SetZTitle("")

        axes.GetTitleTextProperty(0).SetColor(data)
        axes.GetTitleTextProperty(1).SetColor(data)
        axes.GetTitleTextProperty(2).SetColor(data)
        axes.GetLabelTextProperty(0).SetColor(data)
        axes.GetLabelTextProperty(1).SetColor(data)
        axes.GetLabelTextProperty(2).SetColor(data)
        # axes.SetXUnits("A")
        # axes.SetYUnits("A")
        # axes.SetZUnits("A")
        # axes.DrawXGridlinesOn()
        # axes.DrawYGridlinesOn()
        # axes.DrawZGridlinesOn()
        axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)
        axes.XAxisMinorTickVisibilityOff()
        axes.YAxisMinorTickVisibilityOff()
        axes.ZAxisMinorTickVisibilityOff()
        # axes.SetFlyModeToOuterEdges()
        axes.SetFlyMode(axes.VTK_FLY_FURTHEST_TRIAD)
        return axes

    @staticmethod
    def create_sphere_actor(
        pos: npt.NDArray[np.float32], radius: float
    ) -> vtkActor:
        colors = vtkNamedColors()
        sphereSource = vtkSphereSource()
        sphereSource.SetCenter(0.0, 0.0, 0.0)
        sphereSource.SetRadius(radius)
        sphereSource.SetPhiResolution(30)
        sphereSource.SetThetaResolution(30)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())
        actor = vtkActor()
        actor.GetProperty().SetColor(colors.GetColor3d("Gray"))
        actor.GetProperty().SetSpecular(0.6)
        actor.GetProperty().SetSpecularPower(30)
        actor.SetMapper(mapper)
        return actor

    @staticmethod
    def create_polyline_actor(
        points: npt.NDArray[np.float32],
        colors: npt.NDArray[np.float32],
        radius: float,
    ) -> Tuple[vtkActor, vtkPolyData, vtkTubeFilter]:
        count_points = points.shape[0]

        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0, 0, 0, 0.0)
        ctf.AddRGBPoint(1e-6, 0, 0, 1.0)
        ctf.AddRGBPoint(0.25, 0, 1.0, 1)
        ctf.AddRGBPoint(0.5, 0, 1, 0)
        ctf.AddRGBPoint(0.9, 1, 1, 0)
        ctf.AddRGBPoint(1.0, 1, 0, 0)

        color_data = vtkDoubleArray()
        color_data.SetName("saturation")

        assert ~np.any(colors > 1)
        assert ~np.any(colors < 0)

        vpoints = vtkPoints()
        lines = vtkCellArray()

        active_throat = 0
        for i in range(1, count_points):
            vpoints.InsertNextPoint(
                points[i - 1, 0], points[i - 1, 1], points[i - 1, 2]
            )
            vpoints.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])
            line = vtkLine()
            line.GetPointIds().SetId(0, 2 * active_throat)
            line.GetPointIds().SetId(1, 2 * active_throat + 1)
            active_throat += 1

            lines.InsertNextCell(line)
            if len(colors) == count_points:
                c1, c2 = colors[i - 1], colors[i]
            else:
                c1, c2 = colors[i - 1], colors[i - 1]

            color_data.InsertNextValue(c1)
            color_data.InsertNextValue(c2)

        poly_data = vtkPolyData()
        poly_data.SetPoints(vpoints)
        poly_data.SetLines(lines)
        poly_data.GetPointData().AddArray(color_data)

        tube_filter = vtkTubeFilter()
        tube_filter.SetNumberOfSides(16)
        tube_filter.SetInputData(poly_data)
        tube_filter.SetRadius(radius)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(color_data.GetName())
        mapper.SetLookupTable(ctf)
        mapper.Update()

        vtk_colors = vtkNamedColors()
        color = vtk_colors.GetColor3d("hotpink")

        actor = vtkActor()
        actor.SetMapper(mapper)

        actor.GetProperty().SetOpacity(0.8)  # 0.1
        actor.GetProperty().SetSpecular(0.1)
        actor.GetProperty().SetSpecularPower(80)
        actor.GetProperty().SetDiffuse(0.9)
        actor.GetProperty().SetAmbient(0.1)
        actor.GetProperty().SetDiffuseColor(color)

        return actor, poly_data, tube_filter

    @staticmethod
    def create_trj_points_actor(
        trj: Trajectory,
        **kwargs: Any,
    ) -> vtkActor:
        tp = (
            trj.points_without_periodic
            if not kwargs["periodic"]
            else trj.points
        )

        pcount = tp.shape[0]

        points = vtkPoints()
        points.Resize(pcount)
        points.SetNumberOfPoints(pcount)

        pdata = vtkDoubleArray()
        pdata.SetName("clusters")
        pdata.SetNumberOfValues(pcount)

        scales = vtkDoubleArray()
        scales.SetName("scales")
        scales.SetNumberOfValues(pcount)

        for i in range(pcount):
            points.SetPoint(i, tp[i, 0], tp[i, 1], tp[i, 2])
            if kwargs["color_type"] == 'clusters':
                assert trj.traps is not None
                zero_trap = (
                    (i > 0 and i < pcount - 2)
                    and not trj.traps[i]
                    and not trj.traps[i + 1]
                )
                pdata.SetValue(i, 0.0 if zero_trap else 1.0)
                scales.SetValue(i, 2.0 if zero_trap else 1.0)
            else:
                scales.SetValue(i, 1.0)
                pdata.SetValue(i, 1.0)

        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(pdata)
        polydata.GetPointData().AddArray(scales)
        polydata.GetPointData().SetActiveScalars(scales.GetName())

        # const double pore_scale = vis_set->poreScaleRadius();
        sphere_source = vtkSphereSource()
        # sphere_source.SetRadius(kwargs["radius"])
        glyph = vtkGlyph3D()
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(2 * kwargs["radius"])

        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputData(polydata)

        colors = vtkNamedColors()
        color = colors.GetColor3d("gray")

        ctf = vtkColorTransferFunction()
        if kwargs["color_type"] == 'clusters':
            assert trj.traps is not None
            zero_trap_color = colors.GetColor3d("red")
            ctf.AddRGBPoint(0, *zero_trap_color)
        # if trj.traps is not None:
        #     count_clusters = trj.traps.max() + 1
        #     for i in range(count_clusters):
        #         ctf.AddRGBPoint(
        #             float(i) / trj.traps.max(),
        #             random.uniform(0, 1),
        #             random.uniform(0, 1),
        #             random.uniform(0, 1),
        #         )

        ctf.AddRGBPoint(1, *color)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(pdata.GetName())
        mapper.SetLookupTable(ctf)
        mapper.Update()

        actor = vtkActor()
        actor.SetMapper(mapper)
        return actor

    @staticmethod
    def draw_img_trj(
        img: npt.NDArray[Any],
        bbox: BoundingBox,
        trj: Trajectory,
        **kwargs: Any,
    ) -> None:
        renderer = vtkRenderer()

        Visualizer.add_img_actor(renderer, img, bbox, **kwargs)

        actor = Visualizer.create_trajectory_actor(trj, **kwargs)
        renderer.AddActor(actor)
        if kwargs["with_points"]:
            actor = Visualizer.create_trj_points_actor(trj, **kwargs)
            renderer.AddActor(actor)

        colors = vtkNamedColors()
        renderer.SetBackground(colors.GetColor3d("White"))

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        if kwargs["wrap_mode"] == WrapMode.BOX:
            outfit_actor = Visualizer.create_box_actor(bbox)
            renderer.AddActor(outfit_actor)
        elif kwargs["wrap_mode"] == WrapMode.AXES:
            actor = Visualizer.create_axes_actor(
                bbox, renderer.GetActiveCamera()
            )
            renderer.AddActor(actor)
        else:
            assert kwargs["wrap_mode"] == WrapMode.EMPTY

        # Add the actors
        size = img.shape
        fpos = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
        cpos = np.array([size[0] * 2, size[1] * 2, size[2] * 2])
        if bbox is not None:
            fpos = bbox.center()
            cpos = bbox.center() + np.array([*(bbox.size())])

        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(*fpos)
        camera.SetPosition(*cpos)
        # renWin.SetSize(640, 640)

        iren.Initialize()
        renWin.Render()
        renWin.SetSize(1900, 1080)
        iren.Start()

import argparse
import os
import sys
from os.path import realpath
from pathlib import Path

import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox, Range  # noqa: E402
from utils.utils import get_float_img_pattern, kprint  # noqa: E402
from visualizer.visualizer import Visualizer  # noqa: E402


def extanded_struct_extr(
    float_image_path: Path,
    isovalue: float,
    img_opacity: float,
) -> None:
    if os.path.isfile(float_image_path):
        with open(float_image_path, 'rb') as f:  # type: ignore
            img = np.load(f)  # type: ignore
    else:
        kprint("--- Error: no such file")
        return

    pattern = get_float_img_pattern()

    match = pattern.match(float_image_path.name)
    if not match:
        kprint("--- Error: no match")
        return

    data = match.groupdict()
    x_min = float(data["x_min"])
    x_max = float(data["x_max"])
    y_min = float(data["y_min"])
    y_max = float(data["y_max"])
    z_min = float(data["z_min"])
    z_max = float(data["z_max"])

    bbox = BoundingBox(
        Range(x_min, x_max),
        Range(y_min, y_max),
        Range(z_min, z_max),
    )

    img = np.pad(img, [(1, 1), (1, 1), (1, 1)], 'maximum')
    Visualizer.draw_float_img(
        img, bbox, isovalue=isovalue, img_opacity=img_opacity
    )
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Interactively display a distance-field image as an isosurface"
    )
    parser.add_argument(
        '--float_image_path',
        type=Path,
        required=True,
        help="Float image .npy file produced by distance_map_structs",
    )
    parser.add_argument(
        '--isovalue',
        type=float,
        default=0.11,
        help="Isosurface threshold",
    )
    parser.add_argument(
        '--img-opacity',
        type=float,
        default=0.5,
        help="Rendered surface opacity",
    )

    args = parser.parse_args()

    extanded_struct_extr(args.float_image_path, args.isovalue, args.img_opacity)

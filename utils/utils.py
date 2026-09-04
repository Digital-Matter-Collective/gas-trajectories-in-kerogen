import logging
import re
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import poisson

from utils.types import NPFArray


def get_pattern_bbox() -> "re.Pattern[str]":
    return re.compile(
        r"bbox=\("
        r"x=\((?P<x_min>[\d.]+)-(?P<x_max>[\d.]+)\)_"
        r"y=\((?P<y_min>[\d.]+)-(?P<y_max>[\d.]+)\)_"
        r"z=\((?P<z_min>[\d.]+)-(?P<z_max>[\d.]+)\)"
        r"\)_resolution=(?P<resolution>[\d.]+).npy"
    )


def get_float_img_pattern() -> "re.Pattern[str]":
    return re.compile(
        r"result-img-num=(?P<step>\d+)"
        r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?)"
        r"_bbox=\(x=\((?P<x_min>-?\d+(?:\.\d+)?)-(?P<x_max>-?\d+(?:\.\d+)?)\)"
        r"_y=\((?P<y_min>-?\d+(?:\.\d+)?)-(?P<y_max>-?\d+(?:\.\d+)?)\)"
        r"_z=\((?P<z_min>-?\d+(?:\.\d+)?)-(?P<z_max>-?\d+(?:\.\d+)?)\)\)"
        r"_resolution=(?P<resolution>\d+(?:\.\d+)?)"
    )


def create_empirical_cdf(
    vals: npt.NDArray[Any], n: int = 30
) -> npt.NDArray[np.float64]:
    p, bb = np.histogram(vals, bins=n)
    xdel = bb[1] - bb[0]
    x = (bb[:-1] + xdel * 0.5).reshape(n, 1)
    pn = (np.cumsum(p) / np.sum(p)).reshape(n, 1)
    pn[0] = 0
    return np.hstack((x, pn))


def pdistances(points: NPFArray) -> NPFArray:
    dxyz = points[:-1, :] - points[1:, :]
    result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1))
    return result


def ps_generate(type: str, mean_count: int = 50) -> npt.NDArray[np.float32]:
    steps = np.arange(0, 2 * mean_count, dtype=np.float32)

    ps = np.zeros((len(steps), 2), dtype=np.float32)
    ps[:, 0] = steps

    if type == "poisson":
        # ВАЖНО: loc=-50 сдвигает распределение. это может давать "странные" формы на малом диапазоне steps.
        raw = poisson.cdf(steps.astype(int), mean_count).astype(np.float32)

        denom = raw[-1] - raw[0]
        if denom <= 0:
            # на этом диапазоне CDF почти не растёт -> деградируем в ступеньку
            prob = np.zeros_like(raw)
            prob[-1] = 1.0
        else:
            prob = (raw - raw[0]) / denom
            prob[0] = 0.0
            prob[-1] = 1.0
            prob = np.maximum.accumulate(prob)

        ps[:, 1] = prob

    else:
        # "uniform" ветка: линейная CDF
        prob = steps / steps[-1] if steps[-1] != 0 else np.zeros_like(steps)
        prob[0] = 0.0
        prob[-1] = 1.0
        ps[:, 1] = prob.astype(np.float32)

    return ps


def write_binary_file(array: npt.NDArray[np.int8], file_name: str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


_kprint_logger = logging.getLogger("kerogen")


def kprint(*args: Any) -> None:
    """print(...)-style logging: joins args with a space and logs at INFO,
    through the standard `logging` module instead of writing to stdout
    directly (configure handlers/level/format via
    utils.logging_setup.setup_logging())."""
    _kprint_logger.info(" ".join(map(str, args)))

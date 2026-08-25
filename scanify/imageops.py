"""Low level array helpers shared by the effect stages.

Images travel through the pipeline as ``float32`` arrays of shape ``(H, W, 3)``
holding reflectance in the ``0..1`` range, which keeps every stage a plain
multiply or add.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Rows processed at a time by :func:`remap`, to bound peak memory on big pages.
_CHUNK = 256


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_image(arr: np.ndarray) -> Image.Image:
    data = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(data.astype(np.uint8), "RGB")


def blur(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return arr
    return to_array(to_image(arr).filter(ImageFilter.GaussianBlur(radius)))


def value_noise(height: int, width: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth ``0..1`` noise whose features are roughly ``cell`` pixels wide."""
    cell = max(1, int(cell))
    grid_h = max(2, height // cell + 2)
    grid_w = max(2, width // cell + 2)
    grid = (rng.random((grid_h, grid_w)) * 255.0).astype(np.uint8)
    smooth = Image.fromarray(grid, "L").resize((width, height), Image.BICUBIC)
    return np.asarray(smooth, dtype=np.float32) / 255.0


def remap(arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, fill) -> np.ndarray:
    """Bilinear resample of ``arr`` at the given source coordinates.

    Samples that fall outside the source get ``fill``, so a rotated page shows
    paper rather than black in the corners.
    """
    height, width = arr.shape[:2]
    fill_rgb = np.asarray(fill, dtype=np.float32)
    out = np.empty((map_x.shape[0], map_x.shape[1], arr.shape[2]), dtype=np.float32)

    for start in range(0, map_x.shape[0], _CHUNK):
        stop = min(start + _CHUNK, map_x.shape[0])
        mx = map_x[start:stop]
        my = map_y[start:stop]

        x0 = np.floor(mx).astype(np.int32)
        y0 = np.floor(my).astype(np.int32)
        x1, y1 = x0 + 1, y0 + 1
        fx = (mx - x0)[..., None]
        fy = (my - y0)[..., None]

        inside = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
        x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
        y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)

        top = arr[y0c, x0c] + (arr[y0c, x1c] - arr[y0c, x0c]) * fx
        bottom = arr[y1c, x0c] + (arr[y1c, x1c] - arr[y1c, x0c]) * fx
        out[start:stop] = np.where(inside[..., None], top + (bottom - top) * fy, fill_rgb)

    return out


def luminance(arr: np.ndarray) -> np.ndarray:
    return arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

"""The individual stages that turn a crisp page render into a scan.

Each stage takes the working image, the :class:`~scanify.settings.Settings` and a
random generator, and returns a new image. They are deliberately independent so
they can be reordered or switched off one at a time.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .imageops import blur, luminance, remap, to_array, to_image, value_noise
from .settings import Settings


def ink_bleed(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Let dark strokes creep outwards, the way toner and ink do on fibre."""
    if s.bleed <= 0.0:
        return arr
    spread = to_array(to_image(arr).filter(ImageFilter.MinFilter(3)))
    softened = arr + (spread - arr) * float(s.bleed)
    return blur(softened, 0.3 * s.scale())


def paper(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Multiply the page by the reflectance of a real sheet of paper."""
    height, width = arr.shape[:2]
    field = np.ones((height, width), dtype=np.float32)

    if s.paper_blotch > 0.0:
        stains = value_noise(height, width, int(s.paper_blotch_scale * s.scale()), rng)
        # Centred on zero: paper mottles around its own tone instead of only
        # getting darker, which is what turns a sheet into a grey wash.
        field *= 1.0 + s.paper_blotch * (stains - 0.5)
    if s.paper_fiber > 0.0:
        grain = rng.normal(0.0, s.paper_fiber, (height, width)).astype(np.float32)
        field *= 1.0 + grain

    tint = np.asarray(s.paper_tint, dtype=np.float32)
    return np.clip(arr * field[..., None] * tint, 0.0, 1.0)


def geometry(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Rotate, offset and gently deform the sheet as it lay on the glass."""
    height, width = arr.shape[:2]
    scale = s.scale()

    angle = float(rng.uniform(-s.rotate, s.rotate))
    dx = float(rng.uniform(-s.shift, s.shift)) * scale
    dy = float(rng.uniform(-s.shift, s.shift)) * scale
    bow = float(rng.uniform(-s.bow, s.bow)) * height
    wave = float(s.wave) * scale

    if not (angle or dx or dy or bow or wave):
        return arr

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    x, y = xx - cx, yy - cy

    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    # Inverse rotation: for each output pixel, where did it come from?
    src_x = cos * x + sin * y
    src_y = -sin * x + cos * y

    if bow:
        across = src_x / max(cx, 1.0)
        src_y = src_y + bow * (1.0 - across * across)
    if wave:
        src_x = src_x + wave * np.sin(2.0 * np.pi * src_y / (s.wave_period * scale))

    return remap(arr, src_x + cx - dx, src_y + cy - dy, s.paper_tint)


def lighting(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Uneven lamp: corner falloff, an optional spine shadow, exposure drift."""
    height, width = arr.shape[:2]
    field = np.ones((height, width), dtype=np.float32)

    if s.vignette > 0.0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / max(width - 1, 1) - 0.5) * 2.0
        ny = (yy / max(height - 1, 1) - 0.5) * 2.0
        radius = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
        # Cubic falloff keeps the darkening in the corners; a quadratic one
        # dims the middle of the page as well.
        field *= 1.0 - s.vignette * radius ** 3

    if s.fold_shadow > 0.0:
        columns = np.arange(width, dtype=np.float32) / max(width - 1, 1)
        if rng.random() < 0.5:
            columns = 1.0 - columns
        falloff = np.exp(-columns / max(s.fold_width, 1e-3))
        field *= 1.0 - s.fold_shadow * falloff[None, :]

    exposure = 1.0 + float(rng.uniform(-s.brightness, s.brightness))
    return np.clip(arr * field[..., None] * exposure, 0.0, 1.0)


def optics(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    return blur(arr, s.blur * s.scale())


def tone(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Gamma, contrast and the compressed dynamic range typical of a scan."""
    out = arr
    if s.gamma != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), s.gamma)
    if s.contrast != 1.0:
        out = (out - 0.5) * s.contrast + 0.5
    out = np.clip(out, 0.0, 1.0) * (s.white_point - s.black_point) + s.black_point
    return np.clip(out, 0.0, 1.0)


def sensor_noise(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    if s.noise <= 0.0:
        return arr
    grain = rng.normal(0.0, s.noise, arr.shape).astype(np.float32)
    return np.clip(arr + grain, 0.0, 1.0)


def specks(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Dust on the platen and hairline scratches on the glass."""
    if s.dust <= 0.0 and s.scratches <= 0:
        return arr

    height, width = arr.shape[:2]
    img = to_image(arr)
    draw = ImageDraw.Draw(img)

    count = int(height * width * s.dust)
    for _ in range(count):
        x = float(rng.integers(0, width))
        y = float(rng.integers(0, height))
        radius = float(rng.uniform(0.5, 2.2)) * s.scale()
        shade = int(rng.integers(20, 110))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=(shade, shade, shade))

    for _ in range(int(s.scratches)):
        x = float(rng.integers(0, width))
        y = float(rng.integers(0, height))
        length = float(rng.uniform(0.1, 0.45)) * height
        drift = float(rng.uniform(-0.04, 0.04)) * width
        shade = int(rng.integers(150, 215))
        draw.line([x, y, x + drift, y + length], fill=(shade, shade, shade), width=1)

    return to_array(img)


def colorize(arr: np.ndarray, s: Settings, rng: np.random.Generator) -> np.ndarray:
    """Collapse to the requested output colour space."""
    if s.mode == "color":
        return arr
    gray = luminance(arr)
    if s.mode == "gray":
        return np.repeat(gray[..., None], 3, axis=2)
    if s.mode == "bw":
        # Snap the extremes first. Without this the sensor noise on blank paper
        # survives into the dither as an all-over diagonal moire.
        span = max(s.bw_white_cut - s.bw_black_cut, 1e-3)
        gray = np.clip((gray - s.bw_black_cut) / span, 0.0, 1.0)
        source = to_image(np.repeat(gray[..., None], 3, axis=2)).convert("L")
        bilevel = source.convert("1", dither=Image.FLOYDSTEINBERG if s.dither else Image.NONE)
        return to_array(bilevel.convert("RGB"))
    raise ValueError(f"unknown mode {s.mode!r}; expected color, gray or bw")


#: Applied in this order — ink first, then the sheet, the glass, the lamp, the
#: lens and finally the sensor.
STAGES = (
    ("bleed", ink_bleed),
    ("paper", paper),
    ("geometry", geometry),
    ("lighting", lighting),
    ("optics", optics),
    ("tone", tone),
    ("noise", sensor_noise),
    ("specks", specks),
    ("colorize", colorize),
)

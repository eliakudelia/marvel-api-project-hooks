"""Photograph the printed page instead of scanning it.

The sheet is laid on a lit surface, seen slightly off-axis, and shot with a
phone: perspective, a cast shadow, uneven light, and the softness and noise a
small sensor adds. The page itself reuses the scan pipeline's ink and paper
stages, because a printed sheet is a printed sheet either way.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

from .effects import ink_bleed, paper
from .imageops import blur, remap, to_array, to_image, value_noise
from .settings import Settings


@dataclass
class PhotoSettings:
    """Everything about the shot. Lengths are fractions of the frame width."""

    width: int = 1600  # long side of the finished photo, px
    quality: int = 88

    # --- the sheet ---------------------------------------------------------
    paper_tint: tuple = (1.0, 0.988, 0.968)
    paper_blotch: float = 0.03
    paper_fiber: float = 0.012
    bleed: float = 0.2

    # --- framing -----------------------------------------------------------
    margin: float = 0.09  # free space around the sheet
    tilt: float = 4.5  # in-plane rotation, degrees
    perspective: float = 0.075  # keystone, as a fraction of the sheet width
    offset: float = 0.015  # how far off-centre the sheet sits

    # --- surface -----------------------------------------------------------
    surface: tuple = (0.90, 0.888, 0.865)  # a light desk
    surface_blotch: float = 0.07
    surface_scale: int = 90
    surface_streak: float = 0.03  # drawn-out grain, like wood or linen
    surface_streak_scale: int = 16  # how smooth those streaks run
    surface_grain: float = 0.012

    # --- light -------------------------------------------------------------
    light_angle: float = 215.0  # degrees; where the light comes from
    light_falloff: float = 0.26  # brightness drop across the frame
    curl: float = 0.05  # soft bend of the sheet catching the light
    edge_light: float = 0.06  # the paper's own thickness at the rim
    sheet_gain: float = 1.0  # exposure of the paper relative to the surface

    # --- cast shadow -------------------------------------------------------
    shadow: float = 0.4  # ambient shadow strength
    shadow_offset: float = 0.016
    shadow_blur: float = 0.022
    contact: float = 0.3  # tight dark line where paper meets surface
    contact_blur: float = 0.004

    # --- camera ------------------------------------------------------------
    defocus: float = 0.0018  # blur at the frame edges
    vignette: float = 0.2
    chromatic: float = 0.0009  # lateral colour fringing
    noise: float = 0.008
    warmth: float = 0.05  # white balance drift, positive is warmer
    exposure: float = 0.0

    seed: int | None = None

    def as_page_settings(self, dpi: int) -> Settings:
        """The subset the scan pipeline needs to draw ink on paper."""
        return replace(Settings(), dpi=dpi, paper_tint=self.paper_tint,
                       paper_blotch=self.paper_blotch, paper_fiber=self.paper_fiber,
                       bleed=self.bleed)


PHOTO_PRESETS: dict[str, dict[str, Any]] = {
    "desk": dict(),  # the defaults: a light desk, daylight from the upper left
    "table": dict(
        surface=(0.945, 0.937, 0.922), surface_blotch=0.035, surface_streak=0.012,
        margin=0.11, tilt=2.5, perspective=0.035, light_falloff=0.18,
        shadow=0.33, shadow_offset=0.012, warmth=0.03,
    ),
    "wood": dict(
        surface=(0.83, 0.735, 0.62), surface_blotch=0.10, surface_scale=140,
        light_angle=200.0,
        surface_streak=0.09, surface_streak_scale=26, surface_grain=0.018,
        tilt=5.5, perspective=0.08, light_falloff=0.3,
        shadow=0.45, shadow_offset=0.02, warmth=0.09,
    ),
    "handheld": dict(
        margin=0.06, tilt=7.5, perspective=0.13, offset=0.035,
        light_falloff=0.34, curl=0.09,
        shadow=0.5, shadow_offset=0.024, shadow_blur=0.03, contact=0.36,
        defocus=0.0042, vignette=0.28, chromatic=0.0016, noise=0.016,
        warmth=0.07, exposure=-0.02, quality=80,
    ),
    "overhead": dict(
        margin=0.1, tilt=1.2, perspective=0.012, offset=0.006,
        light_falloff=0.13, curl=0.02,
        shadow=0.26, shadow_offset=0.007, shadow_blur=0.014, contact=0.22,
        defocus=0.0008, vignette=0.12, chromatic=0.0004, noise=0.005,
        warmth=0.02, quality=92,
    ),
    "evening": dict(
        surface=(0.87, 0.83, 0.77), light_angle=255.0, light_falloff=0.38,
        tilt=5.0, perspective=0.07, curl=0.08,
        shadow=0.55, shadow_offset=0.026, shadow_blur=0.034, contact=0.34,
        vignette=0.3, noise=0.014, warmth=0.09, exposure=-0.04, quality=84,
    ),
}


def build_photo(preset: str = "desk", **overrides: Any) -> PhotoSettings:
    if preset not in PHOTO_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; available: {', '.join(sorted(PHOTO_PRESETS))}"
        )
    settings = replace(PhotoSettings(), **PHOTO_PRESETS[preset])
    clean = {k: v for k, v in overrides.items() if v is not None}
    unknown = set(clean) - {f.name for f in fields(PhotoSettings)}
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    return replace(settings, **clean)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 3x3 matrix mapping the four ``src`` corners onto ``dst``."""
    rows, targets = [], []
    for (sx, sy), (dx, dy) in zip(src, dst):
        rows.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx])
        targets.append(dx)
        rows.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy])
        targets.append(dy)
    solved = np.linalg.solve(np.asarray(rows, dtype=np.float64),
                             np.asarray(targets, dtype=np.float64))
    return np.append(solved, 1.0).reshape(3, 3)


def sheet_quad(frame_w: float, frame_h: float, sheet_w: float, sheet_h: float,
               s: PhotoSettings, rng: np.random.Generator) -> np.ndarray:
    """Where the four corners of the sheet land in the photo.

    Built in the sheet's own space — keystone first, then in-plane rotation —
    and finally dropped into the frame, so the two never fight each other.
    """
    half_w, half_h = sheet_w / 2.0, sheet_h / 2.0
    corners = np.array([[-half_w, -half_h], [half_w, -half_h],
                        [half_w, half_h], [-half_w, half_h]], dtype=np.float64)

    # Camera held slightly off-axis: the far edge of the sheet is narrower.
    # Draw from the outer half of the range, so a shot is never accidentally
    # dead-on when the preset asked for an angle.
    def swing(amount: float) -> float:
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.0) * amount)

    keystone = swing(s.perspective)
    lean = swing(s.perspective) * 0.6
    depth = corners[:, 1] / max(sheet_h, 1.0)
    across = corners[:, 0] / max(sheet_w, 1.0)
    corners[:, 0] *= 1.0 + keystone * depth * 2.0
    corners[:, 1] *= 1.0 + lean * across * 2.0

    angle = np.deg2rad(swing(s.tilt))
    cos, sin = np.cos(angle), np.sin(angle)
    rotated = np.stack([corners[:, 0] * cos - corners[:, 1] * sin,
                        corners[:, 0] * sin + corners[:, 1] * cos], axis=1)

    centre = np.array([frame_w / 2.0, frame_h / 2.0])
    centre += np.array([float(rng.uniform(-s.offset, s.offset)) * frame_w,
                        float(rng.uniform(-s.offset, s.offset)) * frame_h])
    return rotated + centre


def quad_mask(height: int, width: int, quad: np.ndarray, feather: float = 1.0) -> np.ndarray:
    """A soft-edged mask of a convex quadrilateral, 1 inside."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    centre = quad.mean(axis=0)
    inside = np.full((height, width), np.inf, dtype=np.float32)
    for i in range(4):
        x0, y0 = quad[i]
        x1, y1 = quad[(i + 1) % 4]
        edge_x, edge_y = x1 - x0, y1 - y0
        length = float(np.hypot(edge_x, edge_y)) or 1.0
        distance = ((xx - x0) * edge_y - (yy - y0) * edge_x) / length
        if (centre[0] - x0) * edge_y - (centre[1] - y0) * edge_x < 0:
            distance = -distance
        inside = np.minimum(inside, distance)
    return np.clip(inside / max(feather, 1e-3) + 0.5, 0.0, 1.0)


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return mask
    image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0


def scale_quad(quad: np.ndarray, factor: float) -> np.ndarray:
    centre = quad.mean(axis=0)
    return (quad - centre) * factor + centre


# --------------------------------------------------------------------------
# the shot
# --------------------------------------------------------------------------

def light_field(height: int, width: int, s: PhotoSettings) -> np.ndarray:
    """A brightness ramp across the frame, running away from the light."""
    if s.light_falloff <= 0.0:
        return np.ones((height, width), dtype=np.float32)
    angle = np.deg2rad(s.light_angle)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = xx / max(width - 1, 1) - 0.5
    ny = yy / max(height - 1, 1) - 0.5
    # Normalised by its own reach, so a diagonal lamp spans the same range as a
    # straight-on one and the ramp peaks at exactly 1.0 either way. Without this
    # the bright corner lands above white and the paper clips.
    reach = abs(np.cos(angle)) + abs(np.sin(angle))
    along = (nx * np.cos(angle) + ny * np.sin(angle)) / reach  # -0.5 .. +0.5
    return (1.0 + s.light_falloff * (along - 0.5)).astype(np.float32)


def surface_field(height: int, width: int, s: PhotoSettings,
                  rng: np.random.Generator) -> np.ndarray:
    """The table: base colour, soft blotches, drawn-out grain, fine noise."""
    field = np.ones((height, width), dtype=np.float32)
    if s.surface_blotch > 0.0:
        blotches = value_noise(height, width, max(4, s.surface_scale), rng)
        field *= 1.0 + s.surface_blotch * (blotches - 0.5)
    if s.surface_streak > 0.0:
        # Stretched noise reads as wood or woven cloth rather than dirt.
        narrow = value_noise(height, max(4, width // 60),
                             max(2, s.surface_streak_scale), rng)
        streaks = np.asarray(
            Image.fromarray((narrow * 255).astype(np.uint8), "L").resize(
                (width, height), Image.BICUBIC), np.float32) / 255.0
        field *= 1.0 + s.surface_streak * (streaks - 0.5)
    if s.surface_grain > 0.0:
        field *= 1.0 + rng.normal(0.0, s.surface_grain, (height, width)).astype(np.float32)
    return field


def cast_shadow(height: int, width: int, quad: np.ndarray, s: PhotoSettings) -> np.ndarray:
    """How much light the sheet takes away from the surface under and beside it."""
    scale = width
    angle = np.deg2rad(s.light_angle)
    # The shadow falls on the side opposite the light.
    drift = np.array([-np.cos(angle), -np.sin(angle)]) * s.shadow_offset * scale

    near = blur_mask(quad_mask(height, width, scale_quad(quad + drift, 1.005)),
                     s.shadow_blur * scale) * s.shadow * 0.65
    far = blur_mask(quad_mask(height, width, scale_quad(quad + drift * 2.2, 1.02)),
                    s.shadow_blur * scale * 2.6) * s.shadow * 0.5
    contact = blur_mask(quad_mask(height, width, quad), s.contact_blur * scale) * s.contact
    return np.clip(near + far + contact, 0.0, 1.0)


def sheet_shading(map_x: np.ndarray, map_y: np.ndarray, page_w: int, page_h: int,
                  s: PhotoSettings) -> np.ndarray:
    """A soft bright band where the sheet lifts off the surface."""
    if s.curl <= 0.0:
        return np.ones_like(map_x)
    across = np.clip(map_x / max(page_w - 1, 1), 0.0, 1.0)
    down = np.clip(map_y / max(page_h - 1, 1), 0.0, 1.0)
    bend = np.sin(np.pi * across) * 0.7 + np.sin(np.pi * down) * 0.3
    return (1.0 + s.curl * (bend - 0.5)).astype(np.float32)


def chromatic_aberration(arr: np.ndarray, amount: float) -> np.ndarray:
    """Push red out and blue in, the way a cheap lens splits colour."""
    if amount <= 0.0:
        return arr
    height, width = arr.shape[:2]
    # Pillow puts pixel centres at integer + 0.5, so the frame centre is w/2,
    # not (w-1)/2. Half a pixel out and every edge fringes.
    cx, cy = width / 2.0, height / 2.0
    out = arr.copy()
    for channel, direction in ((0, 1.0), (2, -1.0)):
        factor = 1.0 + amount * direction
        plane = Image.fromarray((np.clip(arr[..., channel], 0, 1) * 255).astype(np.uint8), "L")
        # Inverse affine about the centre: exact, sub-pixel, no cropping.
        inverse = 1.0 / factor
        scaled = plane.transform(
            (width, height), Image.AFFINE,
            (inverse, 0.0, cx * (1.0 - inverse), 0.0, inverse, cy * (1.0 - inverse)),
            resample=Image.BICUBIC)
        out[..., channel] = np.asarray(scaled, np.float32) / 255.0
    return out


def depth_of_field(arr: np.ndarray, s: PhotoSettings) -> np.ndarray:
    """Sharp in the middle, softer towards the corners."""
    radius = s.defocus * arr.shape[1]
    if radius <= 0.05:
        return arr
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx / max(width - 1, 1) - 0.5) * 2.0
    ny = (yy / max(height - 1, 1) - 0.5) * 2.0
    weight = np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)[..., None]
    return arr * (1.0 - weight) + blur(arr, radius) * weight


def camera(arr: np.ndarray, s: PhotoSettings, rng: np.random.Generator) -> np.ndarray:
    """Everything that happens between the desk and the JPEG."""
    out = chromatic_aberration(arr, s.chromatic)
    out = depth_of_field(out, s)

    if s.vignette > 0.0:
        height, width = out.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / max(width - 1, 1) - 0.5) * 2.0
        ny = (yy / max(height - 1, 1) - 0.5) * 2.0
        radius = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
        out = out * (1.0 - s.vignette * radius ** 3)[..., None]

    if s.warmth:
        balance = np.array([1.0, 1.0 - s.warmth * 0.22, 1.0 - s.warmth * 0.62], dtype=np.float32)
        out = out * balance
    out = out * (1.0 + s.exposure)

    if s.noise > 0.0:
        out = out + rng.normal(0.0, s.noise, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _seed_for(s: PhotoSettings, index: int) -> np.random.Generator:
    if s.seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(s.seed), index])


def shoot(page: Image.Image, s: PhotoSettings, rng: np.random.Generator) -> Image.Image:
    """Compose one photo from an already rendered page."""
    page_arr = to_array(page)
    page_settings = s.as_page_settings(200)
    page_arr = ink_bleed(page_arr, page_settings, rng)
    page_arr = paper(page_arr, page_settings, rng)
    page_h, page_w = page_arr.shape[:2]

    # The frame is the sheet plus a margin, in the sheet's own proportions.
    sheet_w = s.width / (1.0 + 2.0 * s.margin)
    sheet_h = sheet_w * page_h / page_w
    frame_w = int(round(s.width))
    frame_h = int(round(sheet_h * (1.0 + 2.0 * s.margin)))
    quad = sheet_quad(frame_w, frame_h, sheet_w, sheet_h, s, rng)

    light = light_field(frame_h, frame_w, s)
    surface = np.asarray(s.surface, dtype=np.float32)
    frame = (surface[None, None, :]
             * surface_field(frame_h, frame_w, s, rng)[..., None]
             * light[..., None])
    frame *= (1.0 - cast_shadow(frame_h, frame_w, quad, s))[..., None]

    matrix = np.linalg.inv(homography(
        np.array([[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]],
                 dtype=np.float64), quad))
    yy, xx = np.mgrid[0:frame_h, 0:frame_w].astype(np.float64)
    denominator = matrix[2, 0] * xx + matrix[2, 1] * yy + matrix[2, 2]
    denominator[np.abs(denominator) < 1e-12] = 1e-12
    map_x = ((matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]) / denominator)
    map_y = ((matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]) / denominator)
    # The mask decides what is sheet; clamping keeps the sampler on real pixels.
    map_x = np.clip(map_x, 0, page_w - 1.001).astype(np.float32)
    map_y = np.clip(map_y, 0, page_h - 1.001).astype(np.float32)

    sheet = remap(page_arr, map_x, map_y, s.paper_tint)
    sheet *= sheet_shading(map_x, map_y, page_w, page_h, s)[..., None]
    sheet *= light[..., None] * s.sheet_gain

    mask = quad_mask(frame_h, frame_w, quad)
    if s.edge_light > 0.0:
        # The cut edge of the stack catches the light on one side and falls into
        # shade on the other, which is what makes the sheet read as an object.
        rim = np.clip(mask - quad_mask(frame_h, frame_w, scale_quad(quad, 0.995)), 0.0, 1.0)
        sheet *= (1.0 + s.edge_light * rim * (light - light.mean()) * 8.0)[..., None]

    mask = mask[..., None]
    composed = frame * (1.0 - mask) + sheet * mask
    return to_image(camera(composed, s, rng))


def photograph_pdf(src: str, settings: PhotoSettings, pages: Sequence[int] | None = None,
                   progress=None) -> list[Image.Image]:
    """Render the selected pages of ``src`` as photographs."""
    import pymupdf

    from .pipeline import render_page

    document = pymupdf.open(src)
    try:
        if document.page_count == 0:
            raise ValueError(f"{src}: the document has no pages")
        indices = list(range(document.page_count)) if pages is None else list(pages)
        outside = [i for i in indices if not 0 <= i < document.page_count]
        if outside:
            raise ValueError(
                f"page(s) {outside} outside the document (1..{document.page_count})")

        shots = []
        for position, index in enumerate(indices):
            page = document.load_page(index)
            # Render just fine enough for the size the sheet takes in the frame.
            sheet_w = settings.width / (1.0 + 2.0 * settings.margin)
            dpi = int(np.clip(round(sheet_w / max(page.rect.width / 72.0, 1e-3)), 72, 400))
            shots.append(shoot(render_page(page, dpi), settings, _seed_for(settings, index)))
            if progress:
                progress(position + 1, len(indices))
        return shots
    finally:
        document.close()

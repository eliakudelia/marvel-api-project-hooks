"""Tunable parameters for the scan simulation and the built-in presets."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass
class Settings:
    """Every knob of the pipeline.

    Distances are given in pixels at 200 dpi and are rescaled automatically when
    a different rendering resolution is used, so a preset looks the same at any
    ``dpi``.
    """

    # --- rendering -------------------------------------------------------
    dpi: int = 200
    mode: str = "gray"  # color | gray | bw
    quality: int = 70  # JPEG quality of the embedded page image

    # --- paper -----------------------------------------------------------
    paper_tint: tuple = (1.0, 0.985, 0.955)  # reflectance of blank paper
    paper_blotch: float = 0.06  # large soft stains
    paper_blotch_scale: int = 40  # size of one stain cell, px
    paper_fiber: float = 0.02  # fine grain of the sheet

    # --- geometry --------------------------------------------------------
    rotate: float = 0.45  # max |angle|, degrees
    shift: float = 4.0  # max translation, px
    bow: float = 0.0  # page sag, fraction of height
    wave: float = 0.0  # horizontal ripple amplitude, px
    wave_period: float = 900.0  # ripple period, px

    # --- lighting --------------------------------------------------------
    vignette: float = 0.10  # corner falloff of the scanner lamp
    fold_shadow: float = 0.0  # darkening along one edge (book spine)
    fold_width: float = 0.12  # width of that shadow, fraction of page width
    brightness: float = 0.02  # random exposure offset

    # --- optics ----------------------------------------------------------
    blur: float = 0.6  # defocus radius, px

    # --- ink -------------------------------------------------------------
    bleed: float = 0.25  # how much dark strokes spread into the paper

    # --- tone curve ------------------------------------------------------
    gamma: float = 1.0
    contrast: float = 1.05
    black_point: float = 0.06  # scans rarely reach pure black
    white_point: float = 0.99

    # --- sensor ----------------------------------------------------------
    noise: float = 0.012  # gaussian sensor noise, sigma
    dust: float = 0.00002  # fraction of pixels covered by specks
    scratches: int = 0  # hairline scratches on the glass

    # --- 1-bit output ----------------------------------------------------
    dither: bool = True
    bw_white_cut: float = 0.78  # anything lighter is forced to clean white
    bw_black_cut: float = 0.16  # anything darker is forced to solid black

    seed: int | None = None

    def scale(self) -> float:
        """Factor converting the 200 dpi reference units to the current dpi."""
        return self.dpi / 200.0


PRESETS: dict[str, dict[str, Any]] = {
    "clean": dict(
        dpi=300, quality=85, mode="gray",
        paper_tint=(1.0, 0.995, 0.985), paper_blotch=0.025, paper_fiber=0.012,
        rotate=0.12, shift=2.0, vignette=0.05, blur=0.45, bleed=0.12,
        contrast=1.03, black_point=0.03, noise=0.006, dust=0.000004,
    ),
    "office": dict(),  # the dataclass defaults are the everyday office scanner
    "worn": dict(
        dpi=200, quality=48, mode="gray",
        paper_tint=(1.0, 0.955, 0.878), paper_blotch=0.13, paper_blotch_scale=55,
        paper_fiber=0.035, rotate=1.1, shift=9.0, bow=0.004, wave=1.6,
        vignette=0.2, brightness=0.05, blur=1.0, bleed=0.42,
        gamma=1.08, contrast=1.12, black_point=0.11, white_point=0.96,
        noise=0.026, dust=0.00012, scratches=3,
    ),
    "book": dict(
        dpi=200, quality=62, mode="gray",
        paper_tint=(1.0, 0.972, 0.925), paper_blotch=0.07, paper_fiber=0.022,
        rotate=0.7, shift=7.0, bow=0.006, wave=1.0,
        vignette=0.13, fold_shadow=0.42, fold_width=0.16, blur=0.9, bleed=0.3,
        contrast=1.06, black_point=0.09, noise=0.016, dust=0.00003,
    ),
    "photocopy": dict(
        dpi=200, quality=55, mode="gray",
        paper_tint=(0.97, 0.97, 0.968), paper_blotch=0.10, paper_blotch_scale=70,
        paper_fiber=0.03, rotate=0.9, shift=8.0,
        vignette=0.16, blur=0.8, bleed=0.55,
        gamma=0.88, contrast=1.5, black_point=0.03, white_point=0.94,
        noise=0.03, dust=0.00009, scratches=2,
    ),
    "fax": dict(
        dpi=150, mode="bw", dither=True,
        paper_tint=(1.0, 1.0, 1.0), paper_blotch=0.0, paper_fiber=0.0,
        rotate=1.4, shift=10.0, wave=2.2, wave_period=500.0,
        vignette=0.0, blur=1.1, bleed=0.6,
        gamma=0.85, contrast=1.6, black_point=0.0, white_point=1.0,
        noise=0.035, dust=0.00006, scratches=1,
        bw_white_cut=0.72, bw_black_cut=0.2,
    ),
}


def build(preset: str = "office", **overrides: Any) -> Settings:
    """Return the settings for ``preset`` with ``overrides`` applied on top."""
    if preset not in PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; available: {', '.join(sorted(PRESETS))}"
        )
    settings = replace(Settings(), **PRESETS[preset])
    clean = {k: v for k, v in overrides.items() if v is not None}
    unknown = set(clean) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    return replace(settings, **clean)

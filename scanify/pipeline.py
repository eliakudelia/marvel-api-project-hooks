"""Drive the whole conversion: render the PDF, age each page, rebuild the file."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pymupdf
from PIL import Image

from .effects import STAGES
from .imageops import to_array, to_image
from .settings import Settings

ProgressHook = Callable[[int, int], None]


@dataclass
class Result:
    pages: int
    width: int
    height: int
    out_bytes: int


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def process_image(img: Image.Image, settings: Settings, rng: np.random.Generator,
                  skip: Iterable[str] = ()) -> Image.Image:
    """Run every enabled stage over one rendered page."""
    skipped = set(skip)
    arr = to_array(img)
    for name, stage in STAGES:
        if name not in skipped:
            arr = stage(arr, settings, rng)
    return to_image(arr)


def encode(img: Image.Image, settings: Settings) -> bytes:
    """Compress the page the way a scanner would store it."""
    buffer = io.BytesIO()
    if settings.mode == "bw":
        # JPEG would smear a bilevel image; keep it 1-bit and lossless instead.
        img.convert("L").point(lambda v: 255 if v > 127 else 0, mode="1").save(
            buffer, format="PNG", optimize=True
        )
    else:
        source = img.convert("L") if settings.mode == "gray" else img
        source.save(buffer, format="JPEG", quality=int(settings.quality),
                    subsampling=0 if settings.quality >= 80 else 2, optimize=True)
    return buffer.getvalue()


def _page_seed(settings: Settings, index: int) -> np.random.Generator:
    """A generator per page, so page N always looks the same for a given seed."""
    if settings.seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(settings.seed), index])


def scanify_pdf(src: str, dst: str, settings: Settings,
                pages: Sequence[int] | None = None,
                progress: ProgressHook | None = None,
                preview: str | None = None) -> Result:
    """Convert ``src`` into a scanned-looking ``dst``.

    ``pages`` selects zero-based page indices; ``None`` means the whole document.
    """
    source = pymupdf.open(src)
    try:
        if source.page_count == 0:
            raise ValueError(f"{src}: the document has no pages")
        indices = list(range(source.page_count)) if pages is None else list(pages)
        out_of_range = [i for i in indices if not 0 <= i < source.page_count]
        if out_of_range:
            raise ValueError(
                f"page(s) {out_of_range} outside the document (1..{source.page_count})"
            )

        out = pymupdf.open()
        try:
            last_size = (0, 0)
            for position, index in enumerate(indices):
                page = source.load_page(index)
                rendered = render_page(page, settings.dpi)
                scanned = process_image(rendered, settings, _page_seed(settings, index))
                last_size = scanned.size

                if preview and position == 0:
                    scanned.save(preview)

                rect = page.rect
                new_page = out.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(new_page.rect, stream=encode(scanned, settings))

                if progress:
                    progress(position + 1, len(indices))

            out.set_metadata({})
            out.save(dst, garbage=4, deflate=True)
        finally:
            out.close()
    finally:
        source.close()

    return Result(pages=len(indices), width=last_size[0], height=last_size[1],
                  out_bytes=os.path.getsize(dst))

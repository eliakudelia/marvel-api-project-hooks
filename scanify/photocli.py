"""Command line front end for the photo mode."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from .cli import coerce, parse_pages
from .photo import PHOTO_PRESETS, PhotoSettings, build_photo, photograph_pdf

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def output_paths(target: str, count: int) -> list[Path]:
    """Work out where each photo goes.

    A directory (or a trailing slash) collects numbered files; a plain name is
    used as-is for a single page and numbered for several.
    """
    path = Path(target)
    if target.endswith(("/", "\\")) or path.is_dir():
        return [path / f"page-{i + 1:02d}.jpg" for i in range(count)]
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(
            f"{path.name}: expected a .jpg, .png or .webp name, a directory, or a .pdf")
    if count == 1:
        return [path]
    return [path.with_name(f"{path.stem}-{i + 1:02d}{path.suffix}") for i in range(count)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanify-photo",
        description="Photograph a PDF as a printed sheet lying on a lit surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets:\n  " + "\n  ".join(sorted(PHOTO_PRESETS)) + "\n\n"
            "examples:\n"
            "  scanify-photo report.pdf -o photo.jpg\n"
            "  scanify-photo report.pdf -o shot.jpg --preset wood --seed 7\n"
            "  scanify-photo report.pdf -o shots/ --pages 1-3\n"
            "  scanify-photo report.pdf -o album.pdf\n"
            "  scanify-photo report.pdf -o p.jpg --set tilt=9 --set shadow=0.6\n"
        ),
    )
    parser.add_argument("input", help="source PDF")
    parser.add_argument("-o", "--output", required=True,
                        help="image name, directory, or a .pdf to collect the photos")
    parser.add_argument("--preset", default="desk", choices=sorted(PHOTO_PRESETS),
                        help="surface and lighting to start from (default: desk)")
    parser.add_argument("--width", type=int, help="long side of the photo, px")
    parser.add_argument("--quality", type=int, help="JPEG quality, 1-95")
    parser.add_argument("--seed", type=int, help="fix the randomness")
    parser.add_argument("--pages", help="pages to shoot, e.g. 1-3,7")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="NAME=VALUE", help="override any setting; repeatable")
    parser.add_argument("--list-settings", action="store_true",
                        help="print every setting with its preset value and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        overrides: dict[str, Any] = {}
        for item in args.overrides:
            name, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"--set expects NAME=VALUE, got {item!r}")
            overrides[name.strip()] = coerce(name.strip(), raw.strip(), PhotoSettings)

        settings = build_photo(args.preset, width=args.width, quality=args.quality,
                               seed=args.seed, **overrides)

        if args.list_settings:
            for field in fields(PhotoSettings):
                print(f"{field.name:22} {getattr(settings, field.name)}")
            return 0

        if not 1 <= settings.quality <= 95:
            raise ValueError("quality must be between 1 and 95")
        if not 200 <= settings.width <= 6000:
            raise ValueError("width must be between 200 and 6000 px")
        if not 0.0 <= settings.margin <= 0.45:
            raise ValueError("margin must be between 0 and 0.45")

        pages = parse_pages(args.pages) if args.pages else None
    except ValueError as exc:
        print(f"scanify-photo: {exc}", file=sys.stderr)
        return 2

    def report(done: int, total: int) -> None:
        print(f"\r  page {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        shots = photograph_pdf(args.input, settings, pages=pages, progress=report)
        as_pdf = args.output.lower().endswith(".pdf")
        if as_pdf:
            written = [Path(args.output)]
            written[0].parent.mkdir(parents=True, exist_ok=True)
            shots[0].save(written[0], format="PDF", save_all=True,
                          append_images=shots[1:], resolution=150.0)
        else:
            written = output_paths(args.output, len(shots))
            written[0].parent.mkdir(parents=True, exist_ok=True)
            for image, path in zip(shots, written):
                image.save(path, quality=settings.quality)
    except ValueError as exc:
        print(f"\rscanify-photo: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as exc:
        print(f"\rscanify-photo: {exc}", file=sys.stderr)
        return 1

    size = shots[0].size
    print(f"\r{written[0]}{'' if len(written) == 1 else f' … {written[-1]}'}: "
          f"{len(shots)} photo(s), {size[0]}x{size[1]} px", file=sys.stderr)
    return 0

"""Command line front end."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from typing import Any, Sequence

from .pipeline import scanify_pdf
from .settings import PRESETS, Settings, build

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_pages(spec: str) -> list[int]:
    """Turn ``"1-3,7,10-"`` into zero-based indices; ``-`` means open ended."""
    indices: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, _, end_text = chunk.partition("-")
            start = int(start_text) if start_text.strip() else 1
            if not end_text.strip():
                raise ValueError(
                    f"open-ended range {chunk!r} needs an end, e.g. '{start}-12'"
                )
            end = int(end_text)
            if end < start:
                raise ValueError(f"range {chunk!r} ends before it starts")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(chunk))
    if not indices:
        raise ValueError(f"no pages selected by {spec!r}")
    if min(indices) < 1:
        raise ValueError("page numbers start at 1")
    return [i - 1 for i in sorted(dict.fromkeys(indices))]


def coerce(name: str, raw: str, model: type = Settings) -> Any:
    """Cast a ``--set name=value`` pair to the type declared on ``model``."""
    declared = {f.name: f for f in fields(model)}
    if name not in declared:
        raise ValueError(f"unknown setting {name!r}")
    field = declared[name]
    kind = field.type
    if isinstance(field.default, tuple):
        parts = [float(v) for v in raw.split(",")]
        if len(parts) != len(field.default):
            raise ValueError(
                f"{name} needs {len(field.default)} comma separated values")
        return tuple(parts)
    if kind is bool or kind == "bool":
        low = raw.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{name}: expected a boolean, got {raw!r}")
    if kind is int or kind == "int":
        return int(raw)
    if "int | None" in str(kind):
        return int(raw)
    if kind is str or kind == "str":
        return raw
    return float(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanify",
        description="Make a PDF look like it was printed and scanned again.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets:\n  " + "\n  ".join(sorted(PRESETS)) + "\n\n"
            "examples:\n"
            "  scanify report.pdf -o scanned.pdf\n"
            "  scanify report.pdf -o worn.pdf --preset worn --seed 7\n"
            "  scanify report.pdf -o fax.pdf --preset fax --pages 1-3\n"
            "  scanify report.pdf -o out.pdf --set noise=0.04 --set rotate=1.5\n"
        ),
    )
    parser.add_argument("input", help="source PDF")
    parser.add_argument("-o", "--output", required=True, help="destination PDF")
    parser.add_argument("--preset", default="office", choices=sorted(PRESETS),
                        help="starting point for the look (default: office)")
    parser.add_argument("--dpi", type=int, help="rendering resolution")
    parser.add_argument("--mode", choices=("color", "gray", "bw"),
                        help="output colour space")
    parser.add_argument("--quality", type=int, help="JPEG quality, 1-95")
    parser.add_argument("--seed", type=int,
                        help="fix the randomness so runs are reproducible")
    parser.add_argument("--pages", help="pages to convert, e.g. 1-3,7")
    parser.add_argument("--preview", metavar="PNG",
                        help="also write the first converted page as an image")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="override any setting; repeatable")
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
            overrides[name.strip()] = coerce(name.strip(), raw.strip())

        settings = build(
            args.preset, dpi=args.dpi, mode=args.mode, quality=args.quality,
            seed=args.seed, **overrides,
        )

        if args.list_settings:
            for field in fields(Settings):
                print(f"{field.name:20} {getattr(settings, field.name)}")
            return 0

        if not 1 <= settings.quality <= 95:
            raise ValueError("quality must be between 1 and 95")
        if settings.dpi < 36:
            raise ValueError("dpi below 36 will not produce a readable page")

        pages = parse_pages(args.pages) if args.pages else None
    except ValueError as exc:
        print(f"scanify: {exc}", file=sys.stderr)
        return 2

    def report(done: int, total: int) -> None:
        print(f"\r  page {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        result = scanify_pdf(args.input, args.output, settings, pages=pages,
                             progress=report, preview=args.preview)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"\rscanify: {exc}", file=sys.stderr)
        return 1

    print(
        f"\r{args.output}: {result.pages} page(s), "
        f"{result.width}x{result.height} px, {result.out_bytes / 1024:.0f} KiB",
        file=sys.stderr,
    )
    return 0

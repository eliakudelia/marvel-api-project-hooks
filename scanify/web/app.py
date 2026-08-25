"""A small local web UI: upload a PDF, tweak the look, download the scan.

The server only ever listens on the loopback interface unless told otherwise,
and every uploaded file stays in a temporary directory that is wiped on exit.
"""

from __future__ import annotations

import argparse
import io
import ntpath
import threading
import webbrowser
from dataclasses import asdict, fields, replace
from typing import Any

import numpy as np
import pymupdf
from flask import Flask, jsonify, request, send_file, send_from_directory

from ..cli import coerce
from ..photo import PHOTO_PRESETS, PhotoSettings, build_photo, photograph_pdf
from ..pipeline import _page_seed, process_image, render_page, scanify_pdf
from ..settings import PRESETS, Settings, build
from .store import DocumentStore

MAX_UPLOAD = 64 * 1024 * 1024
PREVIEW_DPI = 120  # a full 200 dpi render is too slow to drag a slider against
PREVIEW_MAX_SIDE = 1500
PHOTO_PREVIEW_WIDTH = 760  # every photo length is a fraction of the frame width,
                           # so a narrower preview shows exactly the same shot


def _clean(payload: dict[str, Any], model: type) -> dict[str, Any]:
    """Validate what a browser sent, reusing the CLI's own coercion."""
    known = {f.name for f in fields(model)}
    overrides: dict[str, Any] = {}
    for name, value in (payload or {}).items():
        if name not in known or value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        overrides[name] = coerce(name, str(value), model)
    return overrides


def _photo_from(payload: dict[str, Any]) -> PhotoSettings:
    settings = build_photo("desk", **_clean(payload, PhotoSettings))
    if not 1 <= settings.quality <= 95:
        raise ValueError("quality must be between 1 and 95")
    if not 200 <= settings.width <= 4000:
        raise ValueError("width must be between 200 and 4000 px")
    if not 0.0 <= settings.margin <= 0.45:
        raise ValueError("margin must be between 0 and 0.45")
    return settings


def _settings_from(payload: dict[str, Any]) -> Settings:
    settings = build("office", **_clean(payload, Settings))
    if settings.mode not in {"color", "gray", "bw"}:
        raise ValueError(f"unknown mode {settings.mode!r}; expected color, gray or bw")
    if not 1 <= settings.quality <= 95:
        raise ValueError("quality must be between 1 and 95")
    if not 36 <= settings.dpi <= 600:
        raise ValueError("dpi must be between 36 and 600")
    return settings


def _clamp_page(path, requested: Any) -> int:
    """Keep a page number inside the document rather than erroring on it."""
    doc = pymupdf.open(path)
    try:
        return max(0, min(int(requested or 0), doc.page_count - 1))
    finally:
        doc.close()


def create_app(store: DocumentStore | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    app.extensions["scanify_store"] = store or DocumentStore()

    def documents() -> DocumentStore:
        return app.extensions["scanify_store"]

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/presets")
    def presets():
        return jsonify({
            "default": "office",
            "presets": {name: asdict(build(name)) for name in sorted(PRESETS)},
            "preview_dpi": PREVIEW_DPI,
            "photo_default": "desk",
            "photo_presets": {name: asdict(build_photo(name))
                              for name in sorted(PHOTO_PRESETS)},
            "photo_preview_width": PHOTO_PREVIEW_WIDTH,
        })

    @app.post("/api/documents")
    def upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="no file was sent"), 400
        # Browsers may send a full path; keep only the leaf, for display alone.
        name = ntpath.basename(uploaded.filename.replace("/", "\\"))
        try:
            doc_id, pages = documents().add(uploaded.read())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(id=doc_id, pages=pages, name=name)

    @app.post("/api/preview")
    def preview():
        payload = request.get_json(silent=True) or {}
        try:
            settings = _settings_from(payload.get("settings", {}))
            path = documents().path(payload.get("id", ""))
        except KeyError:
            return jsonify(error="unknown document, upload it again"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        doc = pymupdf.open(path)
        try:
            index_ = max(0, min(int(payload.get("page", 0)), doc.page_count - 1))
            # Preview at a lower resolution; every distance scales with dpi, so
            # the look is the same as the full render, only cheaper.
            preview_settings = build("office", **{**asdict(settings), "dpi": PREVIEW_DPI})
            rendered = render_page(doc.load_page(index_), PREVIEW_DPI)
        finally:
            doc.close()

        image = process_image(rendered, preview_settings, _page_seed(preview_settings, index_))
        image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))

        buffer = io.BytesIO()
        image.convert("L" if settings.mode != "color" else "RGB").save(
            buffer, format="JPEG", quality=82
        )
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")

    @app.post("/api/convert")
    def convert():
        payload = request.get_json(silent=True) or {}
        try:
            settings = _settings_from(payload.get("settings", {}))
            path = documents().path(payload.get("id", ""))
        except KeyError:
            return jsonify(error="unknown document, upload it again"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        output = path.with_name(f"{path.stem}-scan.pdf")
        try:
            scanify_pdf(str(path), str(output), settings)
        except (ValueError, RuntimeError, OSError) as exc:
            return jsonify(error=str(exc)), 400

        data = output.read_bytes()
        output.unlink(missing_ok=True)
        name = str(payload.get("name") or "document.pdf")
        stem = name[:-4] if name.lower().endswith(".pdf") else name
        return send_file(io.BytesIO(data), mimetype="application/pdf",
                         as_attachment=True, download_name=f"{stem}-scan.pdf")

    @app.post("/api/photo/preview")
    def photo_preview():
        payload = request.get_json(silent=True) or {}
        try:
            settings = _photo_from(payload.get("settings", {}))
            path = documents().path(payload.get("id", ""))
        except KeyError:
            return jsonify(error="unknown document, upload it again"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        preview = replace(settings, width=PHOTO_PREVIEW_WIDTH)
        try:
            shot = photograph_pdf(str(path), preview,
                                  pages=[_clamp_page(path, payload.get("page", 0))])[0]
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        buffer = io.BytesIO()
        shot.save(buffer, format="JPEG", quality=82)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")

    @app.post("/api/photo/convert")
    def photo_convert():
        payload = request.get_json(silent=True) or {}
        try:
            settings = _photo_from(payload.get("settings", {}))
            path = documents().path(payload.get("id", ""))
        except KeyError:
            return jsonify(error="unknown document, upload it again"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        whole = str(payload.get("scope", "page")) == "all"
        pages = None if whole else [_clamp_page(path, payload.get("page", 0))]
        try:
            shots = photograph_pdf(str(path), settings, pages=pages)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        name = str(payload.get("name") or "document.pdf")
        stem = name[:-4] if name.lower().endswith(".pdf") else name
        buffer = io.BytesIO()
        if len(shots) > 1:
            shots[0].save(buffer, format="PDF", save_all=True,
                          append_images=shots[1:], resolution=150.0)
            mimetype, suffix = "application/pdf", "-photo.pdf"
        else:
            shots[0].save(buffer, format="JPEG", quality=settings.quality)
            mimetype, suffix = "image/jpeg", "-photo.jpg"
        buffer.seek(0)
        return send_file(buffer, mimetype=mimetype, as_attachment=True,
                         download_name=f"{stem}{suffix}")

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"the file is larger than {MAX_UPLOAD // 1024 // 1024} MB"), 413

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scanify-ui", description="Run the local scanify web interface.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (default: loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on start")
    args = parser.parse_args(argv)

    store = DocumentStore()
    app = create_app(store)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print(f"scanify UI on {url}  (ctrl+c to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        store.close()
    return 0

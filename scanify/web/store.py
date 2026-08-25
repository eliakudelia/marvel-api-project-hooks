"""Temporary storage for the PDFs a browser session is working on."""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pymupdf

_ID = re.compile(r"\A[0-9a-f]{32}\Z")


class DocumentStore:
    """Keeps uploaded PDFs on disk for a while, then throws them away.

    Identifiers are plain hex so they can never escape the storage directory,
    and anything older than ``ttl`` is swept on the next access.
    """

    def __init__(self, ttl: float = 3600.0, root: str | None = None) -> None:
        self.ttl = ttl
        self._owned = root is None
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="scanify-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, data: bytes) -> tuple[str, int]:
        """Store ``data`` and return its id together with the page count."""
        self.sweep()
        doc_id = uuid.uuid4().hex
        path = self.root / f"{doc_id}.pdf"
        path.write_bytes(data)
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # pymupdf raises a plain Exception here
            path.unlink(missing_ok=True)
            raise ValueError("this file is not a PDF we can read") from exc
        try:
            pages = doc.page_count
        finally:
            doc.close()
        if pages == 0:
            path.unlink(missing_ok=True)
            raise ValueError("the document has no pages")
        return doc_id, pages

    def path(self, doc_id: str) -> Path:
        if not _ID.match(str(doc_id or "")):
            raise KeyError(doc_id)
        path = self.root / f"{doc_id}.pdf"
        if not path.is_file():
            raise KeyError(doc_id)
        path.touch()  # keep documents alive while they are being worked on
        return path

    def sweep(self) -> int:
        """Delete documents nobody has touched for ``ttl`` seconds."""
        cutoff = time.time() - self.ttl
        removed = 0
        with self._lock:
            for path in self.root.glob("*.pdf"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    pass
        return removed

    def close(self) -> None:
        if self._owned:
            shutil.rmtree(self.root, ignore_errors=True)

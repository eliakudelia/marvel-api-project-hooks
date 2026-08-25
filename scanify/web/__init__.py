"""Local browser interface for scanify."""

from .app import create_app, main
from .store import DocumentStore

__all__ = ["create_app", "main", "DocumentStore"]

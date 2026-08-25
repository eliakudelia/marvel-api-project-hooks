"""Turn a PDF into something that looks printed and scanned again."""

from .photo import PHOTO_PRESETS, PhotoSettings, build_photo, photograph_pdf, shoot
from .pipeline import Result, process_image, scanify_pdf
from .settings import PRESETS, Settings, build

__version__ = "0.1.0"
__all__ = [
    "Result", "Settings", "PRESETS", "build", "process_image", "scanify_pdf",
    "PhotoSettings", "PHOTO_PRESETS", "build_photo", "photograph_pdf", "shoot",
]

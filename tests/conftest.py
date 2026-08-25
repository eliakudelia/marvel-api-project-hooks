import numpy as np
import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    """A small three page document to run the pipeline against."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for number in range(3):
        page = doc.new_page()
        page.insert_text((72, 90), f"Quarterly Report - page {number + 1}",
                         fontsize=18, fontname="helv")
        page.draw_line(pymupdf.Point(72, 104), pymupdf.Point(523, 104))
        page.insert_textbox(pymupdf.Rect(72, 120, 523, 300),
                            "The quick brown fox jumps over the lazy dog. " * 12,
                            fontsize=10.5, fontname="tiro")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def rng():
    return np.random.default_rng(1234)

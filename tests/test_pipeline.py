import numpy as np
import pymupdf
import pytest
from PIL import Image

from scanify.pipeline import encode, process_image, render_page, scanify_pdf
from scanify.settings import PRESETS, build


def open_pages(path):
    doc = pymupdf.open(path)
    try:
        return doc.page_count, [doc.load_page(i).rect for i in range(doc.page_count)]
    finally:
        doc.close()


def test_all_pages_are_converted(sample_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    result = scanify_pdf(str(sample_pdf), str(out), build("office", seed=1, dpi=72))
    assert result.pages == 3
    assert open_pages(out)[0] == 3


def test_page_geometry_is_preserved(sample_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    scanify_pdf(str(sample_pdf), str(out), build("office", seed=1, dpi=72))
    before = open_pages(sample_pdf)[1]
    after = open_pages(out)[1]
    for a, b in zip(before, after):
        assert a.width == pytest.approx(b.width, abs=0.01)
        assert a.height == pytest.approx(b.height, abs=0.01)


def test_page_selection(sample_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    result = scanify_pdf(str(sample_pdf), str(out), build("office", seed=1, dpi=72), pages=[0, 2])
    assert result.pages == 2 and open_pages(out)[0] == 2


def test_out_of_range_page_is_reported(sample_pdf, tmp_path):
    with pytest.raises(ValueError, match="outside the document"):
        scanify_pdf(str(sample_pdf), str(tmp_path / "o.pdf"), build("office"), pages=[7])


def test_progress_hook_is_called_once_per_page(sample_pdf, tmp_path):
    seen = []
    scanify_pdf(str(sample_pdf), str(tmp_path / "o.pdf"), build("office", seed=1, dpi=72),
                progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def page_image(path):
    """The bytes of the scan embedded in page 1, ignoring the PDF wrapper.

    The wrapper itself is never byte-identical: PyMuPDF stamps a fresh trailer
    /ID on every save.
    """
    doc = pymupdf.open(path)
    try:
        xref = doc.get_page_images(0)[0][0]
        return doc.extract_image(xref)["image"]
    finally:
        doc.close()


def test_a_seed_makes_the_result_reproducible(sample_pdf, tmp_path):
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    for target in (first, second):
        scanify_pdf(str(sample_pdf), str(target), build("worn", seed=42, dpi=72), pages=[0])
    assert page_image(first) == page_image(second)


def test_different_seeds_give_different_pages(sample_pdf, tmp_path):
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    scanify_pdf(str(sample_pdf), str(first), build("worn", seed=1, dpi=72), pages=[0])
    scanify_pdf(str(sample_pdf), str(second), build("worn", seed=2, dpi=72), pages=[0])
    assert page_image(first) != page_image(second)


def test_without_a_seed_runs_differ(sample_pdf, tmp_path):
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    for target in (first, second):
        scanify_pdf(str(sample_pdf), str(target), build("worn", dpi=72), pages=[0])
    assert page_image(first) != page_image(second)


def test_each_page_gets_its_own_randomness(sample_pdf, tmp_path):
    """Page 2 must not be an exact copy of page 1's grain and skew."""
    doc = pymupdf.open(sample_pdf)
    settings = build("worn", seed=5)
    rendered = [render_page(doc.load_page(i), 72) for i in range(2)]
    doc.close()
    from scanify.pipeline import _page_seed
    a = np.asarray(process_image(rendered[0], settings, _page_seed(settings, 0)))
    b = np.asarray(process_image(rendered[0], settings, _page_seed(settings, 1)))
    assert not np.array_equal(a, b)


def test_output_carries_no_metadata_from_the_source(sample_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    scanify_pdf(str(sample_pdf), str(out), build("office", seed=1, dpi=72), pages=[0])
    doc = pymupdf.open(out)
    try:
        assert not (doc.metadata or {}).get("title")
    finally:
        doc.close()


def test_empty_document_is_reported(tmp_path):
    # PyMuPDF refuses to *save* a page-less document, so hand-write one.
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n"
    )
    with pytest.raises(ValueError, match="no pages"):
        scanify_pdf(str(empty), str(tmp_path / "o.pdf"), build("office"))


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_runs_end_to_end(sample_pdf, tmp_path, preset):
    out = tmp_path / f"{preset}.pdf"
    result = scanify_pdf(str(sample_pdf), str(out), build(preset, seed=3, dpi=72), pages=[0])
    assert result.out_bytes > 0 and open_pages(out)[0] == 1


def test_bw_pages_are_stored_losslessly(tmp_path):
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    assert encode(img, build("fax")).startswith(b"\x89PNG")


def test_gray_and_colour_pages_are_stored_as_jpeg(tmp_path):
    img = Image.new("RGB", (32, 32), (200, 180, 160))
    assert encode(img, build("office")).startswith(b"\xff\xd8")
    assert encode(img, build("office", mode="color")).startswith(b"\xff\xd8")


def test_processing_keeps_the_page_size(sample_pdf):
    doc = pymupdf.open(sample_pdf)
    try:
        rendered = render_page(doc.load_page(0), 100)
    finally:
        doc.close()
    out = process_image(rendered, build("worn", seed=1), np.random.default_rng(0))
    assert out.size == rendered.size


def test_stages_can_be_skipped(sample_pdf):
    doc = pymupdf.open(sample_pdf)
    try:
        rendered = render_page(doc.load_page(0), 72)
    finally:
        doc.close()
    settings = build("worn", seed=1)
    full = np.asarray(process_image(rendered, settings, np.random.default_rng(0)))
    no_dust = np.asarray(process_image(rendered, settings, np.random.default_rng(0),
                                       skip=["specks"]))
    assert not np.array_equal(full, no_dust)


def test_render_page_scales_with_dpi(sample_pdf):
    doc = pymupdf.open(sample_pdf)
    try:
        low = render_page(doc.load_page(0), 72)
        high = render_page(doc.load_page(0), 144)
    finally:
        doc.close()
    assert high.width == pytest.approx(low.width * 2, abs=2)

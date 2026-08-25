import pytest

pytest.importorskip("flask")

from scanify.web import create_app
from scanify.web.store import DocumentStore


@pytest.fixture
def store(tmp_path):
    keeper = DocumentStore(root=str(tmp_path / "docs"))
    yield keeper
    keeper.close()


@pytest.fixture
def client(store):
    app = create_app(store)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def uploaded(client, sample_pdf):
    response = client.post("/api/documents",
                           data={"file": (sample_pdf.open("rb"), "report.pdf")},
                           content_type="multipart/form-data")
    return response.get_json()


def settings(**overrides):
    return {"dpi": 72, "seed": 1, **overrides}


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"scanify" in response.data


def test_presets_endpoint_lists_every_preset(client):
    body = client.get("/api/presets").get_json()
    assert set(body["presets"]) == {"book", "clean", "fax", "office", "photocopy", "worn"}
    assert body["default"] == "office"
    assert body["presets"]["fax"]["mode"] == "bw"


def test_upload_reports_the_page_count(uploaded):
    assert uploaded["pages"] == 3
    assert uploaded["name"] == "report.pdf"
    assert len(uploaded["id"]) == 32


def test_upload_keeps_only_the_file_name(client, sample_pdf):
    response = client.post(
        "/api/documents",
        data={"file": (sample_pdf.open("rb"), r"C:\Users\me\secret\report.pdf")},
        content_type="multipart/form-data")
    assert response.get_json()["name"] == "report.pdf"


def test_upload_rejects_a_non_pdf(client):
    import io
    response = client.post("/api/documents",
                           data={"file": (io.BytesIO(b"not a pdf at all"), "x.pdf")},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "not a PDF" in response.get_json()["error"]


def test_upload_without_a_file_is_rejected(client):
    assert client.post("/api/documents", data={}).status_code == 400


def test_preview_returns_an_image(client, uploaded):
    response = client.post("/api/preview",
                           json={"id": uploaded["id"], "page": 0, "settings": settings()})
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert len(response.data) > 500


def test_preview_of_a_later_page_works(client, uploaded):
    response = client.post("/api/preview",
                           json={"id": uploaded["id"], "page": 2, "settings": settings()})
    assert response.status_code == 200


def test_preview_clamps_a_page_beyond_the_end(client, uploaded):
    response = client.post("/api/preview",
                           json={"id": uploaded["id"], "page": 99, "settings": settings()})
    assert response.status_code == 200


def test_preview_accepts_a_paper_tint_list(client, uploaded):
    response = client.post("/api/preview", json={
        "id": uploaded["id"], "page": 0,
        "settings": settings(paper_tint=[1.0, 0.95, 0.88])})
    assert response.status_code == 200


def test_preview_ignores_unknown_keys(client, uploaded):
    response = client.post("/api/preview", json={
        "id": uploaded["id"], "page": 0, "settings": settings(sharpness=4)})
    assert response.status_code == 200


@pytest.mark.parametrize("bad", [{"quality": 400}, {"dpi": 5}, {"mode": "sepia"}])
def test_preview_rejects_impossible_settings(client, uploaded, bad):
    response = client.post("/api/preview",
                           json={"id": uploaded["id"], "page": 0, "settings": settings(**bad)})
    assert response.status_code == 400


@pytest.mark.parametrize("doc_id", ["", "nope", "../../etc/passwd", "0" * 32])
def test_preview_of_an_unknown_document_is_404(client, doc_id):
    response = client.post("/api/preview",
                           json={"id": doc_id, "page": 0, "settings": settings()})
    assert response.status_code == 404


def test_convert_returns_a_pdf_attachment(client, uploaded):
    response = client.post("/api/convert",
                           json={"id": uploaded["id"], "name": "report.pdf",
                                 "settings": settings()})
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert "report-scan.pdf" in response.headers["Content-Disposition"]


def test_convert_of_an_unknown_document_is_404(client):
    response = client.post("/api/convert", json={"id": "a" * 32, "settings": settings()})
    assert response.status_code == 404


def test_convert_leaves_no_scratch_file_behind(client, uploaded, store):
    client.post("/api/convert", json={"id": uploaded["id"], "settings": settings()})
    assert sorted(p.name for p in store.root.glob("*.pdf")) == [f"{uploaded['id']}.pdf"]


@pytest.mark.parametrize("doc_id", ["../escape", "abc", "", "A" * 32, "../" * 5 + "x"])
def test_store_rejects_ids_that_are_not_plain_hex(store, doc_id):
    with pytest.raises(KeyError):
        store.path(doc_id)


def test_store_round_trip(store, sample_pdf):
    doc_id, pages = store.add(sample_pdf.read_bytes())
    assert pages == 3
    assert store.path(doc_id).read_bytes() == sample_pdf.read_bytes()


def test_store_sweeps_stale_documents(tmp_path, sample_pdf):
    import os
    import time

    keeper = DocumentStore(ttl=0.0, root=str(tmp_path / "docs"))
    try:
        doc_id, _ = keeper.add(sample_pdf.read_bytes())
        path = keeper.root / f"{doc_id}.pdf"
        os.utime(path, (time.time() - 10, time.time() - 10))
        assert keeper.sweep() == 1
        assert not path.exists()
    finally:
        keeper.close()


def test_store_keeps_fresh_documents(store, sample_pdf):
    doc_id, _ = store.add(sample_pdf.read_bytes())
    assert store.sweep() == 0
    assert store.path(doc_id).exists()


def test_close_removes_only_its_own_directory(tmp_path, sample_pdf):
    root = tmp_path / "docs"
    keeper = DocumentStore(root=str(root))
    keeper.add(sample_pdf.read_bytes())
    keeper.close()
    assert root.exists()  # a caller-supplied directory is left alone


# --- photo mode -----------------------------------------------------------

def photo_settings(**overrides):
    return {"width": 320, "seed": 1, **overrides}


def test_presets_endpoint_also_lists_the_photo_presets(client):
    body = client.get("/api/presets").get_json()
    assert set(body["photo_presets"]) == {
        "desk", "table", "wood", "handheld", "overhead", "evening"}
    assert body["photo_default"] == "desk"
    assert body["photo_preview_width"] > 0


def test_photo_preview_returns_an_image(client, uploaded):
    response = client.post("/api/photo/preview",
                           json={"id": uploaded["id"], "page": 0,
                                 "settings": photo_settings()})
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


def test_photo_preview_clamps_the_page(client, uploaded):
    response = client.post("/api/photo/preview",
                           json={"id": uploaded["id"], "page": 42,
                                 "settings": photo_settings()})
    assert response.status_code == 200


def test_photo_preview_accepts_a_surface_list(client, uploaded):
    response = client.post("/api/photo/preview", json={
        "id": uploaded["id"], "page": 0,
        "settings": photo_settings(surface=[0.8, 0.75, 0.7])})
    assert response.status_code == 200


@pytest.mark.parametrize("bad", [{"quality": 300}, {"width": 20}, {"margin": 0.9}])
def test_photo_preview_rejects_impossible_settings(client, uploaded, bad):
    response = client.post("/api/photo/preview",
                           json={"id": uploaded["id"], "page": 0,
                                 "settings": photo_settings(**bad)})
    assert response.status_code == 400


def test_photo_preview_of_an_unknown_document_is_404(client):
    response = client.post("/api/photo/preview",
                           json={"id": "b" * 32, "page": 0, "settings": photo_settings()})
    assert response.status_code == 404


def test_photo_convert_of_one_page_gives_a_jpeg(client, uploaded):
    response = client.post("/api/photo/convert",
                           json={"id": uploaded["id"], "name": "report.pdf", "page": 0,
                                 "scope": "page", "settings": photo_settings()})
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert "report-photo.jpg" in response.headers["Content-Disposition"]


def test_photo_convert_of_every_page_gives_a_pdf(client, uploaded):
    response = client.post("/api/photo/convert",
                           json={"id": uploaded["id"], "name": "report.pdf",
                                 "scope": "all", "settings": photo_settings()})
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert "report-photo.pdf" in response.headers["Content-Disposition"]


def test_photo_convert_defaults_to_the_single_page(client, uploaded):
    response = client.post("/api/photo/convert",
                           json={"id": uploaded["id"], "settings": photo_settings()})
    assert response.mimetype == "image/jpeg"


def test_photo_convert_of_an_unknown_document_is_404(client):
    response = client.post("/api/photo/convert",
                           json={"id": "c" * 32, "settings": photo_settings()})
    assert response.status_code == 404


def test_scan_and_photo_settings_do_not_leak_into_each_other(client, uploaded):
    """A scan-only key must not be accepted as a photo setting, and vice versa."""
    ok = client.post("/api/photo/preview",
                     json={"id": uploaded["id"], "settings": photo_settings(dpi=999)})
    assert ok.status_code == 200  # unknown keys are ignored, not fatal
    ok = client.post("/api/preview",
                     json={"id": uploaded["id"], "settings": settings(tilt=99)})
    assert ok.status_code == 200

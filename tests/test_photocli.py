from pathlib import Path

import pytest
from PIL import Image

from scanify.photocli import main, output_paths


@pytest.mark.parametrize("target, count, expected", [
    ("photo.jpg", 1, ["photo.jpg"]),
    ("photo.jpg", 3, ["photo-01.jpg", "photo-02.jpg", "photo-03.jpg"]),
    ("shot.png", 2, ["shot-01.png", "shot-02.png"]),
    ("out/", 2, ["out/page-01.jpg", "out/page-02.jpg"]),
    ("a/b/photo.webp", 1, ["a/b/photo.webp"]),
])
def test_output_paths(target, count, expected):
    assert [str(p) for p in output_paths(target, count)] == [str(Path(e)) for e in expected]


def test_output_paths_uses_a_real_directory(tmp_path):
    (tmp_path / "shots").mkdir()
    paths = output_paths(str(tmp_path / "shots"), 2)
    assert paths[0].name == "page-01.jpg" and paths[0].parent.name == "shots"


def test_output_paths_rejects_an_unknown_extension():
    with pytest.raises(ValueError, match="expected a .jpg"):
        output_paths("photo.txt", 1)


def test_single_photo(sample_pdf, tmp_path):
    out = tmp_path / "photo.jpg"
    assert main([str(sample_pdf), "-o", str(out), "--pages", "1",
                 "--width", "320", "--seed", "1"]) == 0
    assert Image.open(out).width == 320


def test_numbered_photos(sample_pdf, tmp_path):
    assert main([str(sample_pdf), "-o", str(tmp_path / "p.jpg"),
                 "--width", "300", "--seed", "1"]) == 0
    assert sorted(f.name for f in tmp_path.glob("*.jpg")) == ["p-01.jpg", "p-02.jpg", "p-03.jpg"]


def test_photos_into_a_directory(sample_pdf, tmp_path):
    target = tmp_path / "shots"
    assert main([str(sample_pdf), "-o", f"{target}/", "--pages", "1-2",
                 "--width", "300", "--seed", "1"]) == 0
    assert sorted(f.name for f in target.glob("*.jpg")) == ["page-01.jpg", "page-02.jpg"]


def test_photos_collected_into_a_pdf(sample_pdf, tmp_path):
    import pymupdf

    out = tmp_path / "album.pdf"
    assert main([str(sample_pdf), "-o", str(out), "--width", "300", "--seed", "1"]) == 0
    doc = pymupdf.open(out)
    try:
        assert doc.page_count == 3
    finally:
        doc.close()


def test_preset_and_set_reach_the_render(sample_pdf, tmp_path, capsys):
    assert main([str(sample_pdf), "-o", str(tmp_path / "p.jpg"), "--preset", "wood",
                 "--set", "tilt=9.5", "--list-settings"]) == 0
    printed = capsys.readouterr().out
    assert "tilt                   9.5" in printed
    assert "surface" in printed


def test_tuple_settings_can_be_overridden(sample_pdf, tmp_path, capsys):
    main([str(sample_pdf), "-o", str(tmp_path / "p.jpg"),
          "--set", "surface=0.5,0.4,0.3", "--list-settings"])
    assert "(0.5, 0.4, 0.3)" in capsys.readouterr().out


@pytest.mark.parametrize("argv, message", [
    (["--set", "bokeh=1"], "unknown setting"),
    (["--set", "tilt"], "NAME=VALUE"),
    (["--set", "surface=0.5,0.4"], "comma separated"),
    (["--quality", "300"], "quality"),
    (["--width", "50"], "width"),
    (["--set", "margin=0.9"], "margin"),
    (["--pages", "0"], "start at 1"),
])
def test_bad_arguments_exit_with_two(sample_pdf, tmp_path, capsys, argv, message):
    code = main([str(sample_pdf), "-o", str(tmp_path / "p.jpg"), "--width", "300"] + argv)
    assert code == 2
    assert message in capsys.readouterr().err


def test_pages_outside_the_document_exit_with_two(sample_pdf, tmp_path, capsys):
    code = main([str(sample_pdf), "-o", str(tmp_path / "p.jpg"),
                 "--width", "300", "--pages", "9"])
    assert code == 2
    assert "outside the document" in capsys.readouterr().err


def test_missing_input_exits_with_one(tmp_path, capsys):
    code = main([str(tmp_path / "gone.pdf"), "-o", str(tmp_path / "p.jpg"), "--width", "300"])
    assert code == 1
    assert "scanify-photo:" in capsys.readouterr().err

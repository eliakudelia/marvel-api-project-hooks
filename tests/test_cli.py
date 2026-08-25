import pytest

from scanify.cli import coerce, main, parse_pages


@pytest.mark.parametrize("spec, expected", [
    ("1", [0]),
    ("3", [2]),
    ("1-3", [0, 1, 2]),
    ("1-2,5", [0, 1, 4]),
    ("5,1-2", [0, 1, 4]),          # sorted
    ("1,1,2", [0, 1]),             # de-duplicated
    (" 2 , 4 ", [1, 3]),           # whitespace tolerant
    ("2-2", [1]),
])
def test_parse_pages(spec, expected):
    assert parse_pages(spec) == expected


@pytest.mark.parametrize("spec, message", [
    ("0", "start at 1"),
    ("3-1", "ends before it starts"),
    ("2-", "needs an end"),
    (",", "no pages selected"),
])
def test_parse_pages_rejects_bad_input(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_pages(spec)


def test_coerce_casts_by_declared_type():
    assert coerce("dpi", "300") == 300 and isinstance(coerce("dpi", "300"), int)
    assert coerce("noise", "0.05") == pytest.approx(0.05)
    assert coerce("mode", "bw") == "bw"
    assert coerce("dither", "false") is False
    assert coerce("dither", "yes") is True
    assert coerce("seed", "7") == 7
    assert coerce("paper_tint", "1.0,0.9,0.8") == (1.0, 0.9, 0.8)


def test_coerce_rejects_unknown_and_malformed():
    with pytest.raises(ValueError, match="unknown setting"):
        coerce("sharpness", "3")
    with pytest.raises(ValueError, match="three comma separated"):
        coerce("paper_tint", "1.0,0.9")
    with pytest.raises(ValueError, match="expected a boolean"):
        coerce("dither", "maybe")


def test_end_to_end_run(sample_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    assert main([str(sample_pdf), "-o", str(out), "--seed", "1", "--dpi", "72"]) == 0
    assert out.stat().st_size > 0


def test_set_flag_reaches_the_pipeline(sample_pdf, tmp_path, capsys):
    assert main([str(sample_pdf), "-o", str(tmp_path / "o.pdf"),
                 "--set", "noise=0.04", "--list-settings"]) == 0
    assert "noise                0.04" in capsys.readouterr().out


def test_list_settings_prints_every_field(capsys):
    from dataclasses import fields
    from scanify.settings import Settings
    main(["x.pdf", "-o", "y.pdf", "--list-settings"])
    printed = capsys.readouterr().out
    assert all(f.name in printed for f in fields(Settings))


@pytest.mark.parametrize("argv, message", [
    (["--set", "nope=1"], "unknown setting"),
    (["--set", "noise"], "NAME=VALUE"),
    (["--quality", "200"], "quality"),
    (["--dpi", "10"], "dpi"),
    (["--pages", "0"], "start at 1"),
])
def test_bad_arguments_exit_with_two(sample_pdf, tmp_path, capsys, argv, message):
    code = main([str(sample_pdf), "-o", str(tmp_path / "o.pdf"), "--dpi", "72"] + argv)
    assert code == 2
    assert message in capsys.readouterr().err


def test_missing_input_exits_with_one(tmp_path, capsys):
    assert main([str(tmp_path / "absent.pdf"), "-o", str(tmp_path / "o.pdf"), "--dpi", "72"]) == 1
    assert "scanify:" in capsys.readouterr().err


def test_page_out_of_range_exits_with_one(sample_pdf, tmp_path, capsys):
    code = main([str(sample_pdf), "-o", str(tmp_path / "o.pdf"), "--pages", "99", "--dpi", "72"])
    assert code == 1
    assert "outside the document" in capsys.readouterr().err


def test_preview_is_written(sample_pdf, tmp_path):
    preview = tmp_path / "preview.png"
    main([str(sample_pdf), "-o", str(tmp_path / "o.pdf"), "--preview", str(preview), "--dpi", "72"])
    assert preview.exists()

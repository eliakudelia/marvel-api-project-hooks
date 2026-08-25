import pytest

from scanify.settings import PRESETS, Settings, build


def test_office_preset_matches_the_defaults():
    assert build("office") == Settings()


def test_preset_values_are_applied():
    assert build("fax").mode == "bw"
    assert build("clean").dpi == 300


def test_overrides_win_over_the_preset():
    assert build("fax", mode="gray", dpi=400).mode == "gray"
    assert build("fax", dpi=400).dpi == 400


def test_none_overrides_are_ignored():
    """The CLI passes None for flags the user did not give."""
    assert build("clean", dpi=None).dpi == build("clean").dpi


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        build("nope")


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        build("office", sharpness=3)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_buildable_and_sane(name):
    s = build(name)
    assert s.mode in {"color", "gray", "bw"}
    assert 1 <= s.quality <= 95
    assert s.dpi >= 36
    assert s.black_point < s.white_point


def test_scale_tracks_dpi():
    assert build("office", dpi=200).scale() == 1.0
    assert build("office", dpi=400).scale() == 2.0

import numpy as np
import pytest

from scanify import effects
from scanify.settings import Settings, build


def page(value=1.0, size=64):
    return np.full((size, size, 3), value, dtype=np.float32)


def test_paper_keeps_blank_sheets_bright():
    """Regression: mottling used to only darken, washing the page out to grey."""
    s = build("worn")
    out = effects.paper(page(), s, np.random.default_rng(0))
    assert out.mean() > 0.93


def test_paper_tint_is_warm():
    out = effects.paper(page(), build("worn"), np.random.default_rng(0))
    assert out[..., 0].mean() > out[..., 2].mean()


def test_paper_adds_variation():
    flat = page()
    out = effects.paper(flat, build("worn"), np.random.default_rng(0))
    assert out.std() > 0.002


def test_vignette_darkens_corners_far_more_than_the_centre():
    s = build("office", vignette=0.4, brightness=0.0)
    out = effects.lighting(page(size=128), s, np.random.default_rng(0))
    centre = out[60:68, 60:68].mean()
    corner = out[:8, :8].mean()
    assert corner < centre * 0.9
    assert centre > 0.98  # the middle of the page must stay clean


def test_fold_shadow_darkens_one_side_only():
    s = build("office", vignette=0.0, brightness=0.0, fold_shadow=0.5)
    out = effects.lighting(page(size=64), s, np.random.default_rng(0))
    left, right = out[:, :6].mean(), out[:, -6:].mean()
    assert min(left, right) < 0.75 and max(left, right) > 0.95


def test_geometry_is_a_noop_when_every_knob_is_zero():
    s = build("office", rotate=0.0, shift=0.0, bow=0.0, wave=0.0)
    arr = page()
    assert effects.geometry(arr, s, np.random.default_rng(0)) is arr


def test_rotation_moves_content():
    arr = page(1.0, 128)
    arr[40:88, 40:88] = 0.0
    s = build("office", rotate=3.0, shift=0.0)
    out = effects.geometry(arr, s, np.random.default_rng(0))
    assert not np.allclose(out, arr, atol=0.02)


def test_geometry_fills_new_corners_with_paper_not_black():
    s = build("office", rotate=4.0, shift=0.0)
    out = effects.geometry(page(0.5), s, np.random.default_rng(0))
    assert out[0, 0].mean() > 0.9


def test_ink_bleed_darkens_glyph_edges():
    arr = page(1.0, 32)
    arr[14:18, 14:18] = 0.0
    out = effects.ink_bleed(arr, build("office", bleed=0.8), np.random.default_rng(0))
    assert out[13, 15].mean() < arr[13, 15].mean()


def test_tone_lifts_black_and_holds_white():
    s = build("office")
    out = effects.tone(np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
                       s, np.random.default_rng(0))
    assert out[0, 0].mean() == pytest.approx(s.black_point, abs=1e-5)
    assert out[0, 1].mean() > 0.95


def test_noise_can_be_switched_off():
    arr = page()
    assert effects.sensor_noise(arr, build("office", noise=0.0), np.random.default_rng(0)) is arr


def test_noise_stays_in_range():
    out = effects.sensor_noise(page(0.5), build("office", noise=0.3), np.random.default_rng(0))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_specks_only_darken_the_page():
    s = build("office", dust=0.01, scratches=0)
    out = effects.specks(page(), s, np.random.default_rng(0))
    assert out.mean() < 1.0


def test_specks_is_a_noop_without_dust_or_scratches():
    arr = page()
    s = build("office", dust=0.0, scratches=0)
    assert effects.specks(arr, s, np.random.default_rng(0)) is arr


def test_gray_mode_removes_colour():
    arr = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    out = effects.colorize(arr, build("office", mode="gray"), np.random.default_rng(0))
    assert np.allclose(out[..., 0], out[..., 2])


def test_color_mode_is_untouched():
    arr = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert effects.colorize(arr, build("office", mode="color"), np.random.default_rng(0)) is arr


def test_bw_mode_produces_only_two_levels():
    arr = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    out = effects.colorize(arr, build("fax"), np.random.default_rng(0))
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_bw_white_cut_keeps_light_paper_free_of_dither():
    """Regression: sensor noise on blank paper used to dither into a moire."""
    noisy = page(0.94, 64) + np.random.default_rng(0).normal(0, 0.03, (64, 64, 3)).astype(np.float32)
    out = effects.colorize(np.clip(noisy, 0, 1), build("fax"), np.random.default_rng(0))
    assert out.mean() == 1.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        effects.colorize(page(), build("office", mode="sepia"), np.random.default_rng(0))


def test_stage_order_starts_with_ink_and_ends_with_colour():
    names = [name for name, _ in effects.STAGES]
    assert names[0] == "bleed" and names[-1] == "colorize"
    assert names.index("paper") < names.index("geometry") < names.index("optics")

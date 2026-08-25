import numpy as np
import pytest
from PIL import Image

from scanify import photo
from scanify.photo import (PHOTO_PRESETS, PhotoSettings, build_photo, cast_shadow,
                           chromatic_aberration, homography, light_field,
                           photograph_pdf, quad_mask, scale_quad, sheet_quad, shoot)

SQUARE = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]])


def blank_page(width=300, height=420):
    return Image.new("RGB", (width, height), (255, 255, 255))


# --- settings -------------------------------------------------------------

def test_desk_preset_matches_the_defaults():
    assert build_photo("desk") == PhotoSettings()


@pytest.mark.parametrize("name", sorted(PHOTO_PRESETS))
def test_every_preset_is_sane(name):
    s = build_photo(name)
    assert 0.0 <= s.margin < 0.5
    assert 1 <= s.quality <= 95
    assert len(s.surface) == 3 and len(s.paper_tint) == 3
    assert all(0.0 <= c <= 1.0 for c in s.surface)


def test_unknown_preset_and_setting_are_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        build_photo("mars")
    with pytest.raises(ValueError, match="unknown setting"):
        build_photo("desk", bokeh=2)


def test_overrides_and_none_handling():
    assert build_photo("wood", tilt=0.5).tilt == 0.5
    assert build_photo("wood", tilt=None).tilt == build_photo("wood").tilt


def test_page_settings_carry_the_paper_across():
    s = build_photo("wood")
    page = s.as_page_settings(180)
    assert page.dpi == 180
    assert page.paper_tint == s.paper_tint and page.bleed == s.bleed


# --- geometry -------------------------------------------------------------

def test_homography_maps_all_four_corners_exactly():
    src = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], float)
    dst = np.array([[10, 5], [110, 15], [105, 205], [5, 195]], float)
    matrix = homography(src, dst)
    for source, target in zip(src, dst):
        mapped = matrix @ np.array([source[0], source[1], 1.0])
        assert np.allclose(mapped[:2] / mapped[2], target, atol=1e-6)


def test_homography_of_a_pure_translation():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    matrix = homography(src, src + 5.0)
    mapped = matrix @ np.array([5.0, 5.0, 1.0])
    assert np.allclose(mapped[:2] / mapped[2], [10.0, 10.0])


def test_quad_mask_is_one_inside_and_zero_outside():
    mask = quad_mask(100, 100, SQUARE)
    assert mask[50, 50] == pytest.approx(1.0)
    assert mask[2, 2] == pytest.approx(0.0)


def test_quad_mask_area_matches_the_quad():
    assert quad_mask(100, 100, SQUARE).sum() == pytest.approx(80 * 80, rel=0.02)


def test_quad_mask_edges_are_soft():
    mask = quad_mask(100, 100, SQUARE)
    assert 0.0 < mask[50, 10] < 1.0


def test_quad_mask_survives_reversed_winding():
    forward = quad_mask(100, 100, SQUARE)
    backward = quad_mask(100, 100, SQUARE[::-1])
    assert np.allclose(forward, backward, atol=0.02)


def test_scale_quad_keeps_the_centre():
    assert np.allclose(scale_quad(SQUARE, 0.5).mean(axis=0), SQUARE.mean(axis=0))


def test_sheet_quad_is_axis_aligned_when_nothing_is_asked_for():
    s = build_photo("desk", tilt=0.0, perspective=0.0, offset=0.0)
    quad = sheet_quad(400, 600, 300, 450, s, np.random.default_rng(0))
    assert quad[0][1] == pytest.approx(quad[1][1])  # top edge level
    assert quad[0][0] == pytest.approx(quad[3][0])  # left edge plumb
    assert np.allclose(quad.mean(axis=0), [200, 300])


def test_sheet_quad_actually_tilts():
    s = build_photo("desk", tilt=8.0, perspective=0.0, offset=0.0)
    quad = sheet_quad(400, 600, 300, 450, s, np.random.default_rng(0))
    assert abs(quad[0][1] - quad[1][1]) > 5


def test_perspective_makes_opposite_edges_differ():
    s = build_photo("desk", tilt=0.0, perspective=0.15, offset=0.0)
    quad = sheet_quad(400, 600, 300, 450, s, np.random.default_rng(0))
    top = np.hypot(*(quad[1] - quad[0]))
    bottom = np.hypot(*(quad[2] - quad[3]))
    assert abs(top - bottom) > 10


def test_a_requested_angle_is_never_lost_to_chance():
    """Regression: uniform(-tilt, tilt) kept producing dead-on shots."""
    s = build_photo("desk", tilt=6.0, perspective=0.0, offset=0.0)
    for seed in range(12):
        quad = sheet_quad(400, 600, 300, 450, s, np.random.default_rng(seed))
        assert abs(quad[0][1] - quad[1][1]) > 3


# --- light and shadow -----------------------------------------------------

def test_light_never_pushes_past_white():
    """Regression: the lamp used to brighten, blowing the paper out to 255."""
    for name in PHOTO_PRESETS:
        field = light_field(120, 90, build_photo(name))
        assert field.max() <= 1.0 + 1e-6
        assert field.min() > 0.5


def test_light_is_brightest_towards_the_lamp():
    field = light_field(120, 120, build_photo("desk", light_angle=225.0, light_falloff=0.3))
    assert field[5, 5] > field[-5, -5]  # 225 degrees puts the lamp upper left


def test_light_is_flat_without_falloff():
    assert np.allclose(light_field(40, 40, build_photo("desk", light_falloff=0.0)), 1.0)


def test_shadow_falls_away_from_the_light():
    s = build_photo("desk", light_angle=225.0, shadow=0.5, contact=0.3)
    shadow = cast_shadow(200, 200, SQUARE * 2.0, s)
    upper_left = shadow[15, 15]
    lower_right = shadow[-15, -15]
    assert lower_right > upper_left


def test_shadow_is_strongest_under_the_sheet():
    s = build_photo("desk")
    shadow = cast_shadow(200, 200, SQUARE * 2.0, s)
    assert shadow[100, 100] > shadow[5, 5]


# --- camera ---------------------------------------------------------------

def test_chromatic_aberration_is_a_noop_at_zero():
    arr = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    assert chromatic_aberration(arr, 0.0) is arr


def test_chromatic_aberration_stays_centred():
    """Regression: resize-then-crop lost half a pixel and fringed every edge."""
    arr = np.zeros((64, 64, 3), dtype=np.float32)
    arr[24:40, 24:40] = 1.0  # symmetric about the centre
    out = chromatic_aberration(arr, 0.02)
    assert np.allclose(out[..., 0], out[..., 0][:, ::-1], atol=0.02)
    assert np.allclose(out[..., 0], out[..., 0][::-1, :], atol=0.02)


def test_chromatic_aberration_separates_the_channels():
    arr = np.zeros((64, 64, 3), dtype=np.float32)
    arr[24:40, 24:40] = 1.0
    out = chromatic_aberration(arr, 0.05)
    assert not np.allclose(out[..., 0], out[..., 2], atol=0.01)


def test_camera_keeps_values_in_range():
    arr = np.full((32, 32, 3), 0.9, dtype=np.float32)
    out = photo.camera(arr, build_photo("handheld"), np.random.default_rng(0))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_depth_of_field_softens_the_corners_only():
    arr = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    out = photo.depth_of_field(arr, build_photo("desk", defocus=0.02))
    centre_change = np.abs(out[60:68, 60:68] - arr[60:68, 60:68]).mean()
    corner_change = np.abs(out[:8, :8] - arr[:8, :8]).mean()
    assert corner_change > centre_change * 3


# --- the whole shot -------------------------------------------------------

def test_shoot_frames_the_sheet_with_a_margin():
    s = build_photo("desk", width=500, seed=1)
    image = shoot(blank_page(), s, np.random.default_rng(0))
    assert image.width == 500
    assert image.height > image.width  # portrait page, portrait frame


def test_shoot_leaves_no_blown_highlights():
    for name in PHOTO_PRESETS:
        s = build_photo(name, width=400, seed=1)
        arr = np.asarray(shoot(blank_page(), s, np.random.default_rng(0)).convert("L"))
        assert (arr >= 255).mean() < 0.01, name


def test_shoot_puts_paper_over_a_darker_surface():
    s = build_photo("wood", width=400, seed=1)
    arr = np.asarray(shoot(blank_page(), s, np.random.default_rng(0)).convert("L"), float)
    centre = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    corner = arr[3, 3]
    assert centre > corner + 20


def test_zero_margin_fills_the_frame():
    s = build_photo("desk", width=400, margin=0.0, tilt=0.0, perspective=0.0, offset=0.0)
    image = shoot(blank_page(), s, np.random.default_rng(0))
    assert image.width == 400


# --- documents ------------------------------------------------------------

def test_photograph_every_page(sample_pdf):
    shots = photograph_pdf(str(sample_pdf), build_photo("desk", width=320, seed=1))
    assert len(shots) == 3
    assert all(shot.width == 320 for shot in shots)


def test_photograph_selected_pages(sample_pdf):
    shots = photograph_pdf(str(sample_pdf), build_photo("desk", width=320, seed=1),
                           pages=[0, 2])
    assert len(shots) == 2


def test_photograph_reports_pages_outside_the_document(sample_pdf):
    with pytest.raises(ValueError, match="outside the document"):
        photograph_pdf(str(sample_pdf), build_photo("desk", width=320), pages=[9])


def test_photograph_progress_hook(sample_pdf):
    seen = []
    photograph_pdf(str(sample_pdf), build_photo("desk", width=320, seed=1),
                   progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_seed_makes_the_shot_reproducible(sample_pdf):
    settings = build_photo("handheld", width=320, seed=11)
    first = photograph_pdf(str(sample_pdf), settings, pages=[0])[0]
    second = photograph_pdf(str(sample_pdf), settings, pages=[0])[0]
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_different_seeds_move_the_sheet(sample_pdf):
    a = photograph_pdf(str(sample_pdf), build_photo("handheld", width=320, seed=1), pages=[0])[0]
    b = photograph_pdf(str(sample_pdf), build_photo("handheld", width=320, seed=2), pages=[0])[0]
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_each_page_is_framed_differently(sample_pdf):
    shots = photograph_pdf(str(sample_pdf), build_photo("handheld", width=320, seed=4))
    assert not np.array_equal(np.asarray(shots[0]), np.asarray(shots[1]))

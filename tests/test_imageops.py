import numpy as np

from scanify.imageops import blur, luminance, remap, to_array, to_image, value_noise


def test_roundtrip_through_pillow_is_stable():
    arr = (np.arange(48, dtype=np.float32) / 47.0).reshape(4, 4, 3)
    assert np.allclose(to_array(to_image(arr)), arr, atol=1.0 / 255.0)


def test_remap_with_identity_coordinates_returns_the_source():
    arr = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
    out = remap(arr, xx, yy, (0.0, 0.0, 0.0))
    # The last row and column have no neighbour to interpolate against.
    assert np.allclose(out[:15, :15], arr[:15, :15], atol=1e-5)


def test_remap_fills_outside_samples():
    arr = np.ones((8, 8, 3), dtype=np.float32)
    coords = np.full((8, 8), -50.0, dtype=np.float32)
    out = remap(arr, coords, coords, (0.25, 0.5, 0.75))
    assert np.allclose(out[..., 0], 0.25) and np.allclose(out[..., 2], 0.75)


def test_remap_handles_pages_taller_than_one_chunk():
    arr = np.random.default_rng(1).random((600, 12, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:600, 0:12].astype(np.float32)
    out = remap(arr, xx, yy, (0.0, 0.0, 0.0))
    assert np.allclose(out[:599, :11], arr[:599, :11], atol=1e-5)


def test_blur_is_a_noop_at_zero_radius():
    arr = np.random.default_rng(2).random((8, 8, 3)).astype(np.float32)
    assert blur(arr, 0.0) is arr


def test_blur_reduces_variance():
    arr = np.random.default_rng(3).random((64, 64, 3)).astype(np.float32)
    assert blur(arr, 2.0).var() < arr.var()


def test_value_noise_shape_and_range():
    noise = value_noise(70, 50, 12, np.random.default_rng(4))
    assert noise.shape == (70, 50)
    assert 0.0 <= noise.min() and noise.max() <= 1.0


def test_value_noise_is_smoother_than_white_noise():
    gen = np.random.default_rng(5)
    smooth = value_noise(128, 128, 24, gen)
    white = gen.random((128, 128)).astype(np.float32)
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(white, axis=1)).mean()


def test_luminance_weights_green_most():
    red = luminance(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    green = luminance(np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32))
    assert green > red

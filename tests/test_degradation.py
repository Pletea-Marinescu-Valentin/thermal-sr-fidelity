import numpy as np
import pytest

from tsrf.data.degradation import (
    TAU2_NETD_K, degrade, imresize_bicubic, synthesize_lr,
)


def test_constant_image_survives_downsampling():
    img = np.full((64, 64), 7500.0)
    out = imresize_bicubic(img, 0.25)
    assert out.shape == (16, 16)
    np.testing.assert_allclose(out, 7500.0, rtol=0, atol=1e-9)


def test_output_shape_matches_scale():
    for size, scale, expect in [(64, 0.25, 16), (128, 0.5, 64), (48, 1 / 3, 16)]:
        out = imresize_bicubic(np.zeros((size, size)), scale)
        assert out.shape == (expect, expect)


def test_linear_ramp_stays_linear_in_the_interior():
    ramp = np.tile(np.linspace(6000, 8000, 64), (64, 1))
    out = imresize_bicubic(ramp, 0.25)
    diffs = np.diff(out[8])[2:-2]
    assert np.allclose(diffs, diffs[0], rtol=1e-6)


def test_antialiasing_suppresses_checkerboard():
    board = np.indices((64, 64)).sum(axis=0) % 2
    board = 7000.0 + 500.0 * board

    aa = imresize_bicubic(board, 0.25)[2:-2, 2:-2]   # interior: borders replicate
    assert aa.std() < 1e-6                      # collapses exactly to the mean
    assert abs(aa.mean() - 7250.0) < 1e-6

    # Control: plain decimation samples one checkerboard phase at every step, so the
    # pattern vanishes and the frame reports the wrong DC level -- 7000 instead of
    # 7250, a 250-count bias that is 10 K of pure resampling error.
    naive = board[::4, ::4].astype(np.float64)
    assert naive.std() == 0.0
    assert abs(naive.mean() - 7250.0) == pytest.approx(250.0)


def test_border_uses_replicate_padding():
    ramp = np.tile(np.linspace(6000, 8000, 64), (64, 1))
    out = imresize_bicubic(ramp, 0.25)
    steps = np.diff(out[8])
    interior_step = steps[2:-2][0]
    deviation = abs(steps[-1] - interior_step)
    # The edge step differs, but only mildly: ~0.4% of the step size, because taps
    # are clamped rather than extrapolated.
    assert deviation > 0.1
    assert deviation < 0.02 * interior_step
    assert out.min() >= ramp.min() - 1e-6                    # and stays in range
    assert out.max() <= ramp.max() + 1e-6


def test_roundtrip_introduces_no_spatial_shift():
    rng = np.random.default_rng(0)
    # Smooth, asymmetric content so a shift is unambiguous.
    yy, xx = np.mgrid[0:256, 0:256]
    img = (7000 + 400 * np.sin(xx / 17.0) + 300 * np.cos(yy / 23.0)
           + rng.normal(0, 5, (256, 256)))

    back = imresize_bicubic(imresize_bicubic(img, 0.25), 4)
    interior = (slice(30, -30), slice(30, -30))

    base = np.abs(back[interior] - img[interior]).mean()
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 2), (-2, -2)]:
        shifted = np.roll(np.roll(back, dy, axis=0), dx, axis=1)
        assert np.abs(shifted[interior] - img[interior]).mean() > base, (
            f"shifting by ({dy},{dx}) improved the match -- resampler is misaligned")


def test_mean_is_approximately_preserved():
    rng = np.random.default_rng(0)
    img = rng.normal(7500, 300, (128, 128))
    assert abs(imresize_bicubic(img, 0.25).mean() - img.mean()) < 2.0


# ------------------------------------------------------------------ synthesize

def test_synthesize_lr_shape_and_dtype():
    hr = np.full((512, 640), 7500.0)
    lr = synthesize_lr(hr, scale=4)
    assert lr.shape == (128, 160)
    assert lr.dtype == np.float64


def test_non_divisible_size_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        synthesize_lr(np.zeros((510, 640)), scale=4)


def test_noise_is_added_at_the_specified_netd():
    hr = np.full((256, 256), 7500.0)
    rng = np.random.default_rng(42)
    lr = synthesize_lr(hr, scale=4, netd_k=TAU2_NETD_K, rng=rng)
    # 0.05 K at 0.04 K/count = 1.25 counts of noise.
    assert 1.0 < lr.std() < 1.5
    assert abs(lr.mean() - 7500.0) < 0.2


def test_noise_is_reproducible_under_a_fixed_seed():
    hr = np.full((64, 64), 7500.0)
    a = synthesize_lr(hr, scale=4, netd_k=0.05, rng=np.random.default_rng(7))
    b = synthesize_lr(hr, scale=4, netd_k=0.05, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_classic_preset_is_noise_free_and_deterministic():
    rng = np.random.default_rng(0)
    hr = rng.normal(7500, 300, (256, 256))
    np.testing.assert_array_equal(degrade(hr, preset="classic"),
                                  degrade(hr, preset="classic"))


def test_realistic_preset_blurs_more_than_classic():
    rng = np.random.default_rng(0)
    hr = rng.normal(7500, 300, (256, 256))
    classic = degrade(hr, preset="classic")
    realistic = degrade(hr, preset="realistic", rng=np.random.default_rng(1))
    # Optical blur removes variance; noise adds only ~1.25 counts back.
    assert realistic.std() < classic.std()


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        degrade(np.zeros((64, 64)), preset="nope")


def test_degradation_preserves_hotspot_peak_approximately():
    hr = np.full((256, 256), 7500.0)
    hr[100:160, 100:160] = 8300.0          # 60x60 px, 32 K above ambient
    lr = synthesize_lr(hr, scale=4)
    assert lr.max() > 8290.0

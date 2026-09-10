import numpy as np
import pytest

from tsrf.metrics.fractal import (
    DEFAULT_SCALES, flat_cold_mask, local_std, scaling_exponent, texture_scaling,
)


def fbm_surface(n, hurst, rng):
    """Fractional Brownian surface by spectral synthesis.

    A 2D fBm of Hurst exponent H has power spectrum S(k) ~ k**-(2H+2), so
    filtering white noise by k**-(H+1) in amplitude produces a surface whose
    local standard deviation scales as w**H.
    """
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0                     # leave the DC term unamplified

    spectrum = np.fft.fft2(rng.standard_normal((n, n))) * k ** (-(hurst + 1.0))
    spectrum[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spectrum))
    return field / field.std()


# ------------------------------------------------------------------ the limits

def test_white_noise_has_hurst_zero():
    # Uncorrelated detector noise: the local standard deviation does not depend
    # on window size, so H = 0 and the surface fills the volume, D = 3.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((256, 256))
    out = scaling_exponent(noise)
    assert abs(out["hurst"]) < 0.05
    assert abs(out["box_dimension"] - 3.0) < 0.05
    # A flat ladder is the case that makes R^2 useless and the RMS residual
    # informative: the fit is excellent even though it explains no variance.
    assert out["fit_rms_resid"] < 0.02


def test_linear_ramp_has_hurst_one():
    # A bandlimited surface: within a window the spread grows in proportion to
    # the window, so H = 1 and D = 2.
    ramp = np.tile(np.arange(256, dtype=np.float64), (256, 1))
    out = scaling_exponent(ramp)
    assert abs(out["hurst"] - 1.0) < 0.05
    assert abs(out["box_dimension"] - 2.0) < 0.05
    assert out["fit_rms_resid"] < 0.02


def test_responds_affinely_to_the_known_hurst_exponent():
    """The estimator is a monotone affine proxy for H, not an unbiased one.

    The discrete variance-scaling estimator compresses the range: over windows
    of 3-17 px it reports about 0.69 H + 0.23. What the metric needs is not
    unbiasedness but a faithful ordering and a constant sensitivity, so that a
    difference between two reconstructions of the same reference is attenuated
    by a known factor rather than distorted. This test pins both.
    """
    rng = np.random.default_rng(7)
    truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    est = np.array([
        np.mean([scaling_exponent(fbm_surface(256, h, rng))["hurst"]
                 for _ in range(3)])
        for h in truth])

    assert np.all(np.diff(est) > 0)                      # strictly monotone
    slope, offset = np.polyfit(truth, est, 1)
    assert np.corrcoef(truth, est)[0, 1] > 0.995         # affine, not merely monotone
    assert 0.6 < slope < 0.8
    assert np.max(np.abs(est - (slope * truth + offset))) < 0.03


# --------------------------------------------------- orthogonality to M4

def test_exponent_is_invariant_to_amplitude():
    # The property that makes M6 independent of M4: rescaling temperature
    # multiplies every rung of the ladder equally and moves only the intercept.
    rng = np.random.default_rng(3)
    field = fbm_surface(256, 0.6, rng)

    base = scaling_exponent(field)
    scaled = scaling_exponent(field * 7.5)

    assert abs(base["hurst"] - scaled["hurst"]) < 1e-9
    assert scaled["log_intercept"] - base["log_intercept"] == pytest.approx(
        np.log(7.5), abs=1e-9)


def test_exponent_is_invariant_to_offset():
    rng = np.random.default_rng(4)
    field = fbm_surface(256, 0.6, rng)
    assert scaling_exponent(field)["hurst"] == pytest.approx(
        scaling_exponent(field + 300.0)["hurst"], abs=1e-6)


# ------------------------------------------------------------------ behaviour

def test_smoothing_raises_the_exponent_and_lowers_the_dimension():
    # Over-smoothing removes small-scale content, which steepens the ladder.
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(5)
    field = fbm_surface(256, 0.4, rng)

    sharp = scaling_exponent(field)
    smooth = scaling_exponent(gaussian_filter(field, 2.0))
    assert smooth["hurst"] > sharp["hurst"] + 0.1
    assert smooth["box_dimension"] < sharp["box_dimension"]


def test_added_fine_grain_lowers_the_exponent():
    # Fabricated small-scale structure does the opposite.
    rng = np.random.default_rng(6)
    field = fbm_surface(256, 0.7, rng)
    grainy = field + 0.3 * rng.standard_normal(field.shape)
    assert scaling_exponent(grainy)["hurst"] < scaling_exponent(field)["hurst"] - 0.1


# ------------------------------------------------------------------ the metric

def test_texture_scaling_reports_zero_delta_for_a_perfect_reconstruction():
    rng = np.random.default_rng(8)
    gt = 290.0 + fbm_surface(128, 0.5, rng)
    out = texture_scaling(gt.copy(), gt)
    assert out["flat_delta_hurst"] == pytest.approx(0.0, abs=1e-9)
    assert out["flat_fit_resid_gt"] == pytest.approx(out["flat_fit_resid_sr"])
    assert out["full_delta_dim"] == pytest.approx(0.0, abs=1e-9)
    assert out["n_flat_px"] > 0


def test_texture_scaling_uses_the_m4_flat_set():
    from tsrf.metrics.thermal import cold_region_smoothness
    rng = np.random.default_rng(9)
    gt = 290.0 + fbm_surface(128, 0.5, rng)
    pred = gt + 0.05 * rng.standard_normal(gt.shape)

    mask = flat_cold_mask(gt)
    m4 = cold_region_smoothness(pred, gt)
    assert texture_scaling(pred, gt)["n_flat_px"] == int(mask.sum()) == m4["n_flat_px"]


def test_texture_scaling_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        texture_scaling(np.zeros((8, 8)), np.zeros((8, 9)))


def test_local_std_matches_numpy_on_a_patch():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((32, 32))
    got = local_std(x, 5)[10, 10]
    assert got == pytest.approx(x[8:13, 8:13].std(), rel=1e-9)


def test_ladder_is_returned_at_every_scale():
    rng = np.random.default_rng(11)
    gt = 290.0 + fbm_surface(128, 0.5, rng)
    out = texture_scaling(gt + 0.01, gt)
    assert len(out["flat_ladder_gt_k"]) == len(DEFAULT_SCALES)


# ------------------------------------------------------- M7: correspondence

def test_correspondence_is_perfect_for_an_exact_reconstruction():
    from tsrf.metrics.fractal import texture_correspondence
    rng = np.random.default_rng(20)
    gt = 290.0 + fbm_surface(128, 0.5, rng)
    out = texture_correspondence(gt.copy(), gt)
    assert out["flat_hf_corr"] == pytest.approx(1.0, abs=1e-9)
    assert out["flat_hf_amp_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_correspondence_separates_fabrication_from_reconstruction():
    """The case M4 and M6 cannot distinguish, and M7 must.

    The scene is built the way a flat cold region actually looks: a smooth
    low-frequency field carrying fine texture on top. Two predictions keep that
    smooth part exactly and differ only in the texture -- one attenuates the
    measured texture, the other replaces it with an independent draw of the
    same amplitude and the same statistics. The amplitude ratio ranks the
    fabricated one as near perfect; only the correlation exposes it.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter
    from tsrf.metrics.fractal import texture_correspondence
    rng = np.random.default_rng(21)

    def fine_texture(seed):
        f = fbm_surface(256, 0.5, np.random.default_rng(seed))
        return f - uniform_filter(f, size=7, mode="reflect")

    # Smooth at the 7 px scale, so the high-pass sees only the texture term.
    low = 290.0 + 3.0 * gaussian_filter(rng.standard_normal((256, 256)), 12.0)
    measured = fine_texture(1)

    gt = low + measured
    # Match the fabricated draw to the measured one *on the flat set*, so the
    # two differ in nothing an image statistic can see.
    other = fine_texture(2)
    on_flat = flat_cold_mask(low + measured)
    fabricated_texture = other * (measured[on_flat].std() / other[on_flat].std())

    smoothed = low + 0.3 * measured                   # keeps the measurement
    fabricated = low + fabricated_texture             # same statistics, independent

    s_out = texture_correspondence(smoothed, gt)
    f_out = texture_correspondence(fabricated, gt)

    assert s_out["flat_hf_amp_ratio"] < 0.4           # visibly over-smoothed
    assert s_out["flat_hf_corr"] > 0.95               # but it is the real texture

    assert f_out["flat_hf_amp_ratio"] == pytest.approx(1.0, abs=0.1)
    assert abs(f_out["flat_hf_corr"]) < 0.1           # and yet uninformative

    # The ranking the two metrics give is opposite: this is the whole point.
    assert abs(f_out["flat_hf_amp_ratio"] - 1) < abs(s_out["flat_hf_amp_ratio"] - 1)
    assert f_out["flat_hf_corr"] < s_out["flat_hf_corr"]


def test_correspondence_uses_the_same_flat_set_as_m4_and_m6():
    from tsrf.metrics.fractal import texture_correspondence
    rng = np.random.default_rng(22)
    gt = 290.0 + fbm_surface(128, 0.5, rng)
    pred = gt + 0.05 * rng.standard_normal(gt.shape)
    assert np.isfinite(texture_correspondence(pred, gt)["flat_hf_corr"])
    assert texture_scaling(pred, gt)["n_flat_px"] == int(flat_cold_mask(gt).sum())


def test_correspondence_rejects_shape_mismatch():
    from tsrf.metrics.fractal import texture_correspondence
    with pytest.raises(ValueError):
        texture_correspondence(np.zeros((8, 8)), np.zeros((8, 9)))

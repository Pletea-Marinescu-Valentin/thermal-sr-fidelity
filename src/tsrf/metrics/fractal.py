"""M6: multiscale texture scaling (fractal) fidelity.

M4 asks *how much* local variation a reconstruction emits in regions the sensor
recorded as uniform. It cannot say whether that variation has the same
statistical character as the measured one: a ratio of 1.0 is reported
identically whether the model reproduced sensor-level thermal texture or
replaced it with fabricated structure of the same amplitude.

This module measures the character. For a self-affine surface the local
standard deviation scales with window size as a power law,

    sigma_w  ~  w ** H,

with H the Hurst exponent; the box dimension of the surface is D = 3 - H.
Fitting the exponent over a ladder of window sizes therefore separates
*amplitude* (the intercept, which M4 already measures) from *scaling* (the
slope, which it does not). Two limits fix the interpretation:

    white detector noise   sigma_w constant in w  ->  H = 0, D = 3
    smooth (bandlimited)   sigma_w proportional w ->  H = 1, D = 2

so H falls as small-scale content is added and rises as it is removed.

The discrete estimator is an affine, not an identity, map of the true exponent:
over synthetic fractional Brownian surfaces of known H it responds as
H_est ~ 0.69 H + 0.23 with r = 0.999 (`tests/test_fractal.py`). We therefore
report it as a variance-scaling exponent and compare reconstruction against
reference, where the common bias cancels, rather than quoting absolute values
against the literature.

The slope is invariant to a global rescaling of temperature -- multiplying T by
c multiplies every sigma_w by c and shifts only the intercept -- which is what
makes M6 orthogonal to M4 by construction rather than by observation.
"""

import numpy as np
from scipy.ndimage import uniform_filter

from .hotspot import ambient_temperature

#: Window ladder for the log-log fit, in pixels. Odd sizes keep the box filter
#: centred. The span 3..17 covers 2.5 octaves: wide enough to fit a slope, and
#: short enough that a box centred on a flat pixel still reads mostly flat
#: neighbours, which matters because the exponent is fitted over a mask.
DEFAULT_SCALES = (3, 5, 9, 17)

#: Window used to define the flat set, matching M4 so that both metrics speak
#: about exactly the same pixels.
FLAT_WINDOW = 7


def local_std(x, window):
    """Local standard deviation over a `window` x `window` box."""
    mean = uniform_filter(x, size=window, mode="reflect")
    sq = uniform_filter(x * x, size=window, mode="reflect")
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def _fit_loglog(scales, values):
    """Least-squares slope of log(values) on log(scales), with a residual.

    Returns (slope, intercept, rms_residual). The residual is reported in
    natural-log units rather than as R^2, because the exponent is allowed to be
    zero here -- white detector noise is one of the two reference limits -- and
    R^2 degenerates exactly there: with a flat ladder the total sum of squares
    vanishes and the ratio becomes meaningless. The RMS residual stays
    interpretable at every slope: 0.05 means the ladder departs from a straight
    line by about 5 percent, and a large value is a warning that the surface is
    not self-affine over this range of scales.
    """
    v = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v <= 0):
        return float("nan"), float("nan"), float("nan")

    x = np.log(np.asarray(scales, dtype=np.float64))
    y = np.log(v)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(resid ** 2)))


def scaling_exponent(t_kelvin, mask=None, scales=DEFAULT_SCALES):
    """Hurst exponent and box dimension of one image, over `mask` if given."""
    t = np.asarray(t_kelvin, dtype=np.float64)
    ladder = []
    for w in scales:
        s = local_std(t, w)
        ladder.append(float(s[mask].mean() if mask is not None else s.mean()))

    h, intercept, resid = _fit_loglog(scales, ladder)
    return {
        "hurst": h,
        "box_dimension": 3.0 - h if np.isfinite(h) else float("nan"),
        "log_intercept": intercept,
        "fit_rms_resid": resid,
        "ladder_k": ladder,
    }


def texture_correspondence(t_pred_k, t_gt_k, window=FLAT_WINDOW,
                           flat_percentile=25.0, require_cold=True):
    """M7: does the emitted fine texture correspond to the measured one?

    M4 and M6 are both statistics of a single image: how much fine variation it
    carries, and how that variation scales. A model that samples convincing
    thermal texture from a learned prior can satisfy both while bearing no
    relation to what the sensor actually recorded, because neither metric ever
    compares the two fields pixel by pixel at that scale -- the pixel metric
    that would, M0, is dominated by the low-frequency component and cannot see
    it either.

    We therefore high-pass both images with the same box filter that defines the
    flat set, and correlate the residuals over it. The pair (amplitude, r)
    separates the two failure modes without ambiguity: over-smoothing gives a
    small amplitude, fabrication gives amplitude near 1 with r near 0, and
    genuine reconstruction is the only way to obtain both.
    """
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")

    hp_p = p - uniform_filter(p, size=window, mode="reflect")
    hp_g = g - uniform_filter(g, size=window, mode="reflect")
    flat = flat_cold_mask(g, window, flat_percentile, require_cold)

    def stats(a, b):
        if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan"), float("nan")
        return float(np.corrcoef(a, b)[0, 1]), float(b.std() / a.std())

    r_flat, amp_flat = stats(hp_g[flat], hp_p[flat]) if flat.any() else (
        float("nan"), float("nan"))
    r_full, amp_full = stats(hp_g.ravel(), hp_p.ravel())
    return {
        "flat_hf_corr": r_flat,
        "flat_hf_amp_ratio": amp_flat,
        "full_hf_corr": r_full,
        "full_hf_amp_ratio": amp_full,
    }


def flat_cold_mask(t_gt_k, window=FLAT_WINDOW, flat_percentile=25.0,
                   require_cold=True):
    """The M4 flat cold set: quietest quartile of the reference, below ambient."""
    g = np.asarray(t_gt_k, dtype=np.float64)
    gstd = local_std(g, window)
    flat = gstd <= np.percentile(gstd, flat_percentile)
    if require_cold:
        flat &= g <= ambient_temperature(g)
    return flat


def texture_scaling(t_pred_k, t_gt_k, scales=DEFAULT_SCALES,
                    window=FLAT_WINDOW, flat_percentile=25.0,
                    require_cold=True):
    """M6: scaling-exponent agreement between a reconstruction and its reference.

    The exponent is fitted twice: over the flat cold set of M4, where texture
    fabrication lives, and over the whole frame, where genuine thermal
    structure dominates and the exponent is therefore harder to move.
    """
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")

    flat = flat_cold_mask(g, window, flat_percentile, require_cold)
    has_flat = bool(flat.any())
    out = {"n_flat_px": int(flat.sum())}

    # One pass over the ladder. The two regions and the two images share the
    # same filtered fields, so each local_std is computed once rather than four
    # times; on a 640x512 frame that is the difference between 32 box filters
    # per frame and 8.
    ladders = {("flat", "gt"): [], ("flat", "sr"): [],
               ("full", "gt"): [], ("full", "sr"): []}
    for w in scales:
        sg, sp = local_std(g, w), local_std(p, w)
        ladders[("full", "gt")].append(float(sg.mean()))
        ladders[("full", "sr")].append(float(sp.mean()))
        if has_flat:
            ladders[("flat", "gt")].append(float(sg[flat].mean()))
            ladders[("flat", "sr")].append(float(sp[flat].mean()))

    for tag in ("flat", "full"):
        if tag == "flat" and not has_flat:
            out.update({f"flat_{k}": float("nan") for k in
                        ("hurst_gt", "hurst_sr", "delta_hurst",
                         "dim_gt", "dim_sr", "delta_dim",
                         "fit_resid_gt", "fit_resid_sr")})
            continue

        hg, _, rg = _fit_loglog(scales, ladders[(tag, "gt")])
        hp, _, rp = _fit_loglog(scales, ladders[(tag, "sr")])
        dim = lambda h: 3.0 - h if np.isfinite(h) else float("nan")  # noqa: E731
        out.update({
            f"{tag}_hurst_gt": hg,
            f"{tag}_hurst_sr": hp,
            f"{tag}_delta_hurst": hp - hg,
            f"{tag}_dim_gt": dim(hg),
            f"{tag}_dim_sr": dim(hp),
            f"{tag}_delta_dim": dim(hp) - dim(hg),
            f"{tag}_fit_resid_gt": rg,
            f"{tag}_fit_resid_sr": rp,
        })
        if tag == "flat":
            out["flat_ladder_gt_k"] = ladders[("flat", "gt")]
            out["flat_ladder_sr_k"] = ladders[("flat", "sr")]
    return out

import numpy as np
import pytest

from tsrf.metrics import (
    cold_region_smoothness,
    detect_hotspots,
    gradient_fidelity,
    hallucination_metrics,
    hotspot_preservation,
    mask_iou,
    radiometric_error,
    thermal_ordering,
)

AMBIENT_K = 293.15  # 20 C
H = W = 128
RNG = np.random.default_rng(1337)


def scene(blobs=(), ambient=AMBIENT_K, noise_k=0.0, size=(H, W)):
    t = np.full(size, ambient, dtype=np.float64)
    for cx, cy, r, dk in blobs:
        t[cy - r:cy + r, cx - r:cx + r] = ambient + dk
    if noise_k:
        t = t + RNG.normal(0, noise_k, size)
    return t


def downup(t, factor=4):
    import cv2
    h, w = t.shape
    lr = cv2.resize(t, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ detection

def test_detect_finds_expected_blobs():
    t = scene([(30, 30, 6, 20.0), (90, 90, 6, 20.0)])
    mask, spots = detect_hotspots(t, AMBIENT_K + 10.0, min_area=20)
    assert len(spots) == 2
    assert mask.sum() == 2 * (12 * 12)
    assert all(s.peak_k == pytest.approx(AMBIENT_K + 20.0) for s in spots)


def test_area_filter_rejects_specks():
    # A 2x2 blob is 4 px, below min_area=20, so it must not register.
    t = scene([(64, 64, 1, 20.0)])
    _, spots = detect_hotspots(t, AMBIENT_K + 10.0, min_area=20)
    assert spots == []


def test_ambient_only_scene_has_no_hotspots():
    _, spots = detect_hotspots(scene(), AMBIENT_K + 10.0, min_area=20)
    assert spots == []


# ------------------------------------------------------------------------- M0

def test_radiometric_error_is_zero_for_identity():
    t = scene([(40, 40, 8, 25.0)])
    e = radiometric_error(t, t)
    assert e["rmse_k"] == 0.0 and e["mae_k"] == 0.0 and e["bias_k"] == 0.0


def test_radiometric_error_reports_kelvin_bias():
    t = scene([(40, 40, 8, 25.0)])
    e = radiometric_error(t + 2.0, t)
    assert e["bias_k"] == pytest.approx(2.0)
    assert e["rmse_k"] == pytest.approx(2.0)


# ------------------------------------------------------------------------- M1

def test_perfect_prediction_scores_perfectly():
    t = scene([(30, 30, 8, 20.0), (90, 90, 8, 25.0)])
    m = hotspot_preservation(t, t, delta_k=10.0)
    assert m["mask_iou"] == 1.0
    assert m["count_error"] == 0
    assert m["peak_position_error_px"] == 0.0
    assert m["peak_temperature_error_k"] == 0.0


def test_missing_hotspot_is_counted():
    gt = scene([(30, 30, 8, 20.0), (90, 90, 8, 20.0)])
    pred = scene([(30, 30, 8, 20.0)])  # second blob erased
    m = hotspot_preservation(pred, gt, delta_k=10.0)
    assert m["count_gt"] == 2 and m["count_pred"] == 1
    assert m["count_error"] == -1
    assert m["n_unmatched_gt"] == 1
    assert m["mask_iou"] < 1.0


def test_small_uniform_bias_is_caught_by_temperature_not_geometry():
    gt = scene([(64, 64, 8, 20.0)])
    m = hotspot_preservation(gt + 8.0, gt, delta_k=10.0)
    assert m["mask_iou"] == 1.0
    assert m["peak_temperature_error_k"] == pytest.approx(8.0)
    assert m["peak_error_on_gt_support_k"] == pytest.approx(8.0)


def test_large_uniform_bias_is_not_hidden_by_adaptive_thresholding():
    gt = scene([(64, 64, 8, 20.0)])
    m = hotspot_preservation(gt + 12.0, gt, delta_k=10.0)
    # The whole frame now reads as one hotspot, so IoU collapses to
    # |GT blob| / |image| = 16*16 / 128*128.
    assert m["mask_iou"] == pytest.approx((16 * 16) / (H * W))
    assert m["count_pred"] == 1
    # The single predicted component overlaps GT at IoU 0.0156, below iou_tau, so
    # nothing matches and the matched-pair statistics are undefined by design.
    assert m["n_matched"] == 0
    assert np.isnan(m["peak_temperature_error_k"])
    # The matching-free variant still reports the bias -- which is why it exists.
    assert m["peak_error_on_gt_support_k"] == pytest.approx(12.0)


# ------------------------------------------------------------------------- M2

def test_invented_hotspot_is_flagged():
    gt = scene([(30, 30, 8, 20.0)])
    pred = scene([(30, 30, 8, 20.0), (95, 95, 8, 20.0)])  # blob 2 invented
    lr_up = downup(gt)
    m = hallucination_metrics(pred, gt, lr_up, delta_k=10.0)
    assert m["n_hallucinated"] == 1
    assert m["n_missed"] == 0


def test_hotspot_visible_in_lr_is_not_hallucination():
    gt = scene([(30, 30, 8, 20.0)])
    lr_scene = scene([(30, 30, 8, 20.0), (95, 95, 10, 25.0)])
    pred = scene([(30, 30, 8, 20.0), (95, 95, 8, 20.0)])
    m = hallucination_metrics(pred, gt, downup(lr_scene), delta_k=10.0)
    assert m["n_unmatched_pred"] == 1
    assert m["n_hallucinated"] == 0


def test_faithful_prediction_hallucinates_nothing():
    gt = scene([(30, 30, 8, 20.0), (90, 90, 8, 20.0)])
    m = hallucination_metrics(gt, gt, downup(gt), delta_k=10.0)
    assert m["n_hallucinated"] == 0
    assert m["n_missed"] == 0
    assert m["hallucination_rate"] == 0.0


def test_erased_hotspot_counts_as_miss_not_hallucination():
    gt = scene([(30, 30, 8, 20.0), (90, 90, 8, 20.0)])
    pred = scene([(30, 30, 8, 20.0)])
    m = hallucination_metrics(pred, gt, downup(gt), delta_k=10.0)
    assert m["n_missed"] == 1
    assert m["n_hallucinated"] == 0


# ------------------------------------------------------------------------- M3

def test_ordering_preserved_when_ranking_intact():
    gt = scene([(30, 30, 8, 30.0), (90, 90, 8, 20.0)])
    pred = scene([(30, 30, 8, 28.0), (90, 90, 8, 19.0)])  # cooler, same order
    m = thermal_ordering(pred, gt, delta_k=5.0)
    assert m["n_regions"] == 2
    assert m["spearman_rho"] == pytest.approx(1.0)
    assert m["top1_preserved"] == 1.0


def test_ordering_broken_when_hottest_region_swaps():
    gt = scene([(30, 30, 8, 30.0), (90, 90, 8, 20.0)])
    pred = scene([(30, 30, 8, 20.0), (90, 90, 8, 30.0)])  # ranking inverted
    m = thermal_ordering(pred, gt, delta_k=5.0)
    assert m["spearman_rho"] == pytest.approx(-1.0)
    assert m["top1_preserved"] == 0.0


def test_ordering_undefined_below_two_regions():
    gt = scene([(64, 64, 8, 20.0)])
    assert np.isnan(thermal_ordering(gt, gt, delta_k=5.0)["spearman_rho"])


# ------------------------------------------------------------------------- M4

def test_injected_texture_in_flat_area_is_detected():
    # Realistic GT: a real sensor floor is smooth but never exactly constant.
    gt = scene([(30, 30, 8, 20.0)], noise_k=0.05)
    pred = gt + RNG.normal(0, 0.5, gt.shape)            # GAN-style fake detail
    m = cold_region_smoothness(pred, gt)
    assert m["texture_ratio"] > 2.0
    assert m["excess_texture_k"] > 0.0


def test_perfectly_flat_gt_yields_undefined_ratio_but_usable_excess():
    gt = scene([(30, 30, 8, 20.0)])                     # exactly constant background
    m = cold_region_smoothness(gt + RNG.normal(0, 0.5, gt.shape), gt)
    assert np.isnan(m["texture_ratio"])
    assert m["excess_texture_k"] > 0.0


def test_faithful_flat_area_has_unit_texture_ratio():
    gt = scene([(30, 30, 8, 20.0)], noise_k=0.3)
    m = cold_region_smoothness(gt, gt)
    assert m["texture_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["residual_std_k"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------- M5

def test_gradient_identity_is_perfect():
    gt = scene([(64, 64, 12, 20.0)])
    m = gradient_fidelity(gt, gt)
    assert m["gradient_ratio"] == pytest.approx(1.0)
    assert m["orientation_cosine"] == pytest.approx(1.0)
    assert m["gradient_mae_k_per_px"] == pytest.approx(0.0)


def test_oversmoothed_edges_lose_gradient():
    import cv2
    gt = scene([(64, 64, 12, 20.0)])
    pred = cv2.GaussianBlur(gt, (9, 9), 3.0)
    m = gradient_fidelity(pred, gt)
    assert m["gradient_ratio"] < 0.8  # L1-regression failure mode


# ---------------------------------------------------------------------- misc

def test_mask_iou_edge_cases():
    a = np.zeros((8, 8), dtype=bool)
    assert mask_iou(a, a) == 1.0          # both empty -> defined as perfect
    b = a.copy()
    b[:4] = True
    assert mask_iou(b, b) == 1.0
    assert mask_iou(a, b) == 0.0


def test_shape_mismatch_raises():
    a = scene(size=(64, 64))
    b = scene(size=(32, 32))
    for fn in (radiometric_error, hotspot_preservation, thermal_ordering,
               cold_region_smoothness, gradient_fidelity):
        with pytest.raises(ValueError):
            fn(a, b)

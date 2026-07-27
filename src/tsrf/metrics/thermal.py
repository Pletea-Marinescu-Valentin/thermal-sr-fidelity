import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr

from .hotspot import _component_masks, ambient_temperature


# --------------------------------------------------------------------------- M0

def radiometric_error(t_pred_k, t_gt_k):
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    d = p - g
    return {
        "rmse_k": float(np.sqrt(np.mean(d ** 2))),
        "mae_k": float(np.mean(np.abs(d))),
        "max_abs_error_k": float(np.max(np.abs(d))),
        "bias_k": float(np.mean(d)),
        "p99_abs_error_k": float(np.percentile(np.abs(d), 99)),
    }


# --------------------------------------------------------------------------- M3

def thermal_ordering(t_pred_k, t_gt_k, delta_k=5.0, min_area=20, top_k=10):
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")

    threshold = ambient_temperature(g) + delta_k
    comps = _component_masks(g, threshold, min_area)
    if len(comps) < 2:
        return {"n_regions": len(comps), "spearman_rho": float("nan"),
                "spearman_p": float("nan"), "top1_preserved": float("nan"),
                "top1_temp_gap_k": float("nan")}

    gt_means = np.array([g[m].mean() for m in comps])
    order = np.argsort(-gt_means)[:top_k]
    comps = [comps[i] for i in order]
    gt_means = gt_means[order]
    pred_means = np.array([p[m].mean() for m in comps])

    rho, pval = spearmanr(gt_means, pred_means)
    gap = float(gt_means[0] - gt_means[1]) if len(gt_means) > 1 else float("nan")
    return {
        "n_regions": len(comps),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "top1_preserved": float(int(np.argmax(pred_means) == 0)),
        "top1_temp_gap_k": gap,
    }


# --------------------------------------------------------------------------- M4

def cold_region_smoothness(t_pred_k, t_gt_k, window=7, flat_percentile=25.0,
                           require_cold=True):
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")

    def local_std(x):
        mean = uniform_filter(x, size=window, mode="reflect")
        sq = uniform_filter(x * x, size=window, mode="reflect")
        return np.sqrt(np.maximum(sq - mean * mean, 0.0))

    gstd = local_std(g)
    pstd = local_std(p)

    flat = gstd <= np.percentile(gstd, flat_percentile)
    if require_cold:
        flat &= g <= ambient_temperature(g)
    if not flat.any():
        return {"n_flat_px": 0, "residual_std_k": float("nan"),
                "gt_local_std_k": float("nan"), "pred_local_std_k": float("nan"),
                "texture_ratio": float("nan"), "excess_texture_k": float("nan")}

    gt_rough = float(gstd[flat].mean())
    pred_rough = float(pstd[flat].mean())
    return {
        "n_flat_px": int(flat.sum()),
        "residual_std_k": float(np.std(p[flat] - g[flat])),
        "gt_local_std_k": gt_rough,
        "pred_local_std_k": pred_rough,
        "texture_ratio": pred_rough / gt_rough if gt_rough > 1e-9 else float("nan"),
        "excess_texture_k": pred_rough - gt_rough,
    }


# --------------------------------------------------------------------------- M5

def gradient_fidelity(t_pred_k, t_gt_k, edge_percentile=90.0):
    p = np.asarray(t_pred_k, dtype=np.float64)
    g = np.asarray(t_gt_k, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")

    gx_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    gx_p = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)

    mag_g = np.hypot(gx_g, gy_g)
    mag_p = np.hypot(gx_p, gy_p)

    # Strictly-positive guard: in scenes dominated by flat regions the percentile
    # itself can be 0 (e.g. 98.8% zero-gradient pixels -> percentile(90) == 0), and
    # `mag_g >= 0` would select the whole image, silently averaging real edges
    # together with flat background and pulling every ratio toward 1.
    edges = (mag_g >= np.percentile(mag_g, edge_percentile)) & (mag_g > 0)
    if not edges.any():
        return {"n_edge_px": 0, "gradient_mae_k_per_px": float("nan"),
                "gradient_ratio": float("nan"),
                "orientation_cosine": float("nan")}

    dot = gx_p[edges] * gx_g[edges] + gy_p[edges] * gy_g[edges]
    denom = mag_p[edges] * mag_g[edges]
    valid = denom > 1e-9
    cos = float(np.mean(dot[valid] / denom[valid])) if valid.any() else float("nan")

    return {
        "n_edge_px": int(edges.sum()),
        "gradient_mae_k_per_px": float(np.mean(np.abs(mag_p[edges] - mag_g[edges]))),
        "gradient_ratio": float(mag_p[edges].mean() / mag_g[edges].mean()),
        "orientation_cosine": cos,
    }

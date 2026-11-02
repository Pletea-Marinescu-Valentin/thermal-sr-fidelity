from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

#: Temperature offsets above ambient, in Kelvin, swept in the paper.
DEFAULT_DELTAS_K = (5.0, 10.0, 20.0)

#: Minimum connected-component area in pixels at HR, to reject sensor noise.
DEFAULT_MIN_AREA = 20


@dataclass(frozen=True)
class Hotspot:

    area: int
    centroid_xy: tuple
    peak_xy: tuple
    peak_k: float
    mean_k: float
    bbox_xywh: tuple

    def as_dict(self):
        return asdict(self)


def ambient_temperature(t_kelvin):
    return float(np.median(t_kelvin))


def detect_hotspots(t_kelvin, threshold_k, min_area=DEFAULT_MIN_AREA):
    t = np.asarray(t_kelvin, dtype=np.float64)
    raw = (t >= threshold_k).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)

    mask = np.zeros_like(raw)
    spots = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == i
        mask[comp] = 1
        vals = t[comp]
        ys, xs = np.nonzero(comp)
        k = int(np.argmax(vals))
        spots.append(Hotspot(
            area=area,
            centroid_xy=(float(centroids[i][0]), float(centroids[i][1])),
            peak_xy=(int(xs[k]), int(ys[k])),
            peak_k=float(vals.max()),
            mean_k=float(vals.mean()),
            bbox_xywh=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                       int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
        ))
    return mask, spots


def _component_masks(t_kelvin, threshold_k, min_area):
    t = np.asarray(t_kelvin, dtype=np.float64)
    raw = (t >= threshold_k).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    return [labels == i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def mask_iou(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _iou_matrix(pred_masks, gt_masks):
    n_p, n_g = len(pred_masks), len(gt_masks)
    shape = pred_masks[0].shape
    pred_lab = np.zeros(shape, dtype=np.int32)
    gt_lab = np.zeros(shape, dtype=np.int32)
    pred_area = np.empty(n_p, dtype=np.int64)
    gt_area = np.empty(n_g, dtype=np.int64)
    for i, m in enumerate(pred_masks):
        pred_lab[m] = i + 1
        pred_area[i] = m.sum()
    for j, m in enumerate(gt_masks):
        gt_lab[m] = j + 1
        gt_area[j] = m.sum()

    both = (pred_lab > 0) & (gt_lab > 0)
    idx = (pred_lab[both] - 1) * n_g + (gt_lab[both] - 1)
    inter = np.bincount(idx, minlength=n_p * n_g).reshape(n_p, n_g).astype(np.float64)
    union = pred_area[:, None] + gt_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_components(pred_masks, gt_masks, iou_tau=0.1):
    if not pred_masks or not gt_masks:
        return [], list(range(len(pred_masks))), list(range(len(gt_masks)))

    iou = _iou_matrix(pred_masks, gt_masks)
    rows, cols = linear_sum_assignment(-iou)
    matches, used_p, used_g = [], set(), set()
    for i, j in zip(rows, cols):
        if iou[i, j] >= iou_tau:
            matches.append((int(i), int(j), float(iou[i, j])))
            used_p.add(int(i))
            used_g.add(int(j))
    return (matches,
            [i for i in range(len(pred_masks)) if i not in used_p],
            [j for j in range(len(gt_masks)) if j not in used_g])


def hotspot_preservation(t_pred_k, t_gt_k, delta_k=10.0,
                         min_area=DEFAULT_MIN_AREA, iou_tau=0.1):
    t_pred_k = np.asarray(t_pred_k, dtype=np.float64)
    t_gt_k = np.asarray(t_gt_k, dtype=np.float64)
    if t_pred_k.shape != t_gt_k.shape:
        raise ValueError(f"shape mismatch: {t_pred_k.shape} vs {t_gt_k.shape}")

    threshold = ambient_temperature(t_gt_k) + delta_k
    pred_mask, pred_spots = detect_hotspots(t_pred_k, threshold, min_area)
    gt_mask, gt_spots = detect_hotspots(t_gt_k, threshold, min_area)

    pred_comps = _component_masks(t_pred_k, threshold, min_area)
    gt_comps = _component_masks(t_gt_k, threshold, min_area)
    matches, unmatched_pred, unmatched_gt = match_components(
        pred_comps, gt_comps, iou_tau)

    pos_err, peak_err = [], []
    for pi, gj, _ in matches:
        p, g = pred_spots[pi], gt_spots[gj]
        pos_err.append(float(np.hypot(p.peak_xy[0] - g.peak_xy[0],
                                      p.peak_xy[1] - g.peak_xy[1])))
        peak_err.append(abs(p.peak_k - g.peak_k))

    # Assumption-free variant: peak error measured over the GT support, so it needs
    # no component matching and stays defined when the model finds no hotspot at all.
    if gt_mask.any():
        support = gt_mask.astype(bool)
        peak_on_support = abs(float(t_pred_k[support].max() - t_gt_k[support].max()))
    else:
        peak_on_support = float("nan")

    return {
        "delta_k": delta_k,
        "threshold_k": threshold,
        "mask_iou": mask_iou(pred_mask, gt_mask),
        "count_pred": len(pred_spots),
        "count_gt": len(gt_spots),
        "count_error": len(pred_spots) - len(gt_spots),
        "abs_count_error": abs(len(pred_spots) - len(gt_spots)),
        "n_matched": len(matches),
        "peak_position_error_px": float(np.mean(pos_err)) if pos_err else float("nan"),
        "peak_temperature_error_k": float(np.mean(peak_err)) if peak_err else float("nan"),
        "peak_error_on_gt_support_k": peak_on_support,
        "n_unmatched_pred": len(unmatched_pred),
        "n_unmatched_gt": len(unmatched_gt),
    }


def hallucination_metrics(t_pred_k, t_gt_k, t_lr_upsampled_k, delta_k=10.0,
                          min_area=DEFAULT_MIN_AREA, iou_tau=0.1,
                          lr_relax=0.5, lr_overlap_tau=0.05):
    t_pred_k = np.asarray(t_pred_k, dtype=np.float64)
    t_gt_k = np.asarray(t_gt_k, dtype=np.float64)
    t_lr_k = np.asarray(t_lr_upsampled_k, dtype=np.float64)
    if not (t_pred_k.shape == t_gt_k.shape == t_lr_k.shape):
        raise ValueError("pred, gt and upsampled LR must share a shape, got "
                         f"{t_pred_k.shape}, {t_gt_k.shape}, {t_lr_k.shape}")

    ambient = ambient_temperature(t_gt_k)
    threshold = ambient + delta_k
    lr_threshold = ambient + delta_k * lr_relax

    pred_comps = _component_masks(t_pred_k, threshold, min_area)
    gt_comps = _component_masks(t_gt_k, threshold, min_area)
    lr_mask, _ = detect_hotspots(t_lr_k, lr_threshold, min_area)
    lr_bool = lr_mask.astype(bool)

    matches, unmatched_pred, unmatched_gt = match_components(
        pred_comps, gt_comps, iou_tau)

    hallucinated = []
    for i in unmatched_pred:
        comp = pred_comps[i]
        # Fraction of the invented region that the LR input could justify.
        support = np.logical_and(comp, lr_bool).sum() / max(comp.sum(), 1)
        if support < lr_overlap_tau:
            hallucinated.append(i)

    area_px = t_gt_k.size
    return {
        "delta_k": delta_k,
        "n_hallucinated": len(hallucinated),
        "n_unmatched_pred": len(unmatched_pred),
        "n_missed": len(unmatched_gt),
        "n_pred": len(pred_comps),
        "n_gt": len(gt_comps),
        "hallucination_rate": len(hallucinated) / max(len(pred_comps), 1),
        "miss_rate": len(unmatched_gt) / max(len(gt_comps), 1),
        "hallucinated_per_megapixel": len(hallucinated) * 1e6 / area_px,
        "hallucinated_area_px": int(sum(pred_comps[i].sum() for i in hallucinated)),
    }

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tsrf.data.degradation import degrade, imresize_bicubic  # noqa: E402
from tsrf.data.radiometry import raw_to_celsius, raw_to_kelvin  # noqa: E402
from tsrf.eval import bicubic_upsampler, load_checkpoint, load_scaler, torch_upsampler  # noqa: E402
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics.hotspot import (  # noqa: E402
    _component_masks, ambient_temperature, detect_hotspots, match_components,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import DISPLAY_ORDER, MODEL_SPECS, available_models  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val")
FIG_DIR = Path("results/figures")
DELTA_K = 10.0
MIN_AREA = 20


def load_models(scaler):
    models = {"bicubic": bicubic_upsampler(4)}
    for name in available_models():
        spec = MODEL_SPECS[name]
        net = build_model(spec["builder"], scale=4, **spec["kwargs"])
        net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
        models[name] = torch_upsampler(net, scaler)
    return models


def classify_regions(t_sr_k, t_gt_k, t_lr_up_k, threshold, lr_threshold):
    pred = _component_masks(t_sr_k, threshold, MIN_AREA)
    gt = _component_masks(t_gt_k, threshold, MIN_AREA)
    lr_mask, _ = detect_hotspots(t_lr_up_k, lr_threshold, MIN_AREA)
    lr_bool = lr_mask.astype(bool)

    _, unmatched_pred, unmatched_gt = match_components(pred, gt, iou_tau=0.1)

    halluc = np.zeros(t_gt_k.shape, bool)
    for i in unmatched_pred:
        comp = pred[i]
        if np.logical_and(comp, lr_bool).sum() / max(comp.sum(), 1) < 0.05:
            halluc |= comp
    missed = np.zeros(t_gt_k.shape, bool)
    for j in unmatched_gt:
        missed |= gt[j]
    return halluc, missed


def pick_frames(frames, n, pool):
    scored = []
    for name in frames[:pool]:
        raw = np.array(Image.open(SPLIT_DIR / "analyticsData" / name))
        t = raw_to_kelvin(raw)
        mask, spots = detect_hotspots(t, ambient_temperature(t) + DELTA_K, MIN_AREA)
        scored.append((mask.sum(), name))
    scored.sort(reverse=True)
    return [n_ for _, n_ in scored[:n]]


def render(frame_name, models, scaler, out_path):
    raw = np.array(Image.open(SPLIT_DIR / "analyticsData" / frame_name),
                   dtype=np.float64)
    lr = degrade(raw, scale=4, preset="classic")
    t_gt = raw_to_kelvin(raw)
    t_lr_up = raw_to_kelvin(imresize_bicubic(lr, 4))
    ambient = ambient_temperature(t_gt)
    thr, lr_thr = ambient + DELTA_K, ambient + DELTA_K * 0.5
    gt_mask, _ = detect_hotspots(t_gt, thr, MIN_AREA)

    order = ["GT"] + [m for m in DISPLAY_ORDER if m in models]
    vmin, vmax = raw_to_celsius(raw).min(), raw_to_celsius(raw).max()

    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4.2))
    for ax, key in zip(np.atleast_1d(axes), order):
        if key == "GT":
            img_c, title, halluc, missed = raw_to_celsius(raw), "Ground truth", None, None
        else:
            sr = np.asarray(models[key](lr), dtype=np.float64)
            img_c = raw_to_celsius(sr)
            halluc, missed = classify_regions(
                raw_to_kelvin(sr), t_gt, t_lr_up, thr, lr_thr)
            title = MODEL_SPECS.get(key, {}).get("label", key)

        ax.imshow(img_c, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.contour(gt_mask, levels=[0.5], colors="lime", linewidths=0.8)
        if halluc is not None and halluc.any():
            ax.contourf(halluc, levels=[0.5, 1], colors="red", alpha=0.45)
        if missed is not None and missed.any():
            ax.contourf(missed, levels=[0.5, 1], colors="deepskyblue", alpha=0.35)
        if key == "GT":
            ax.set_title(title, fontsize=10)
        else:
            import cv2
            n_h = cv2.connectedComponents(halluc.astype(np.uint8), 8)[0] - 1
            ax.set_title(f"{title}\nhalluc regions: {n_h}", fontsize=10)
        ax.axis("off")

    handles = [Line2D([0], [0], color="lime", lw=2, label="GT hotspot"),
               Line2D([0], [0], color="red", lw=6, alpha=0.45, label="hallucinated"),
               Line2D([0], [0], color="deepskyblue", lw=6, alpha=0.35, label="missed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{frame_name}   (ambient {ambient - 273.15:.1f} C, "
                 f"threshold +{DELTA_K:.0f} K)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--limit-pool", type=int, default=200)
    args = ap.parse_args()

    scaler = load_scaler()
    models = load_models(scaler)
    print(f"models: {list(models)}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_splits()["test"]
    chosen = pick_frames(frames, args.frames, args.limit_pool)
    print(f"selected {len(chosen)} high-hotspot frames")
    for i, name in enumerate(chosen):
        render(name, models, scaler, FIG_DIR / f"qual_{i:02d}_{Path(name).stem}.png")
    print(f"\nwrote {len(chosen)} figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()

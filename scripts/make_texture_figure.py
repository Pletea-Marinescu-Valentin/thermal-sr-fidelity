import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.radiometry import raw_to_celsius, raw_to_kelvin  # noqa: E402
from tsrf.eval import load_checkpoint, load_scaler, torch_upsampler  # noqa: E402
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics.hotspot import ambient_temperature  # noqa: E402
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
OUT = Path("paper/fig_texture.pdf")
WIN = 7
CROP = 128


def local_std(x, w=WIN):
    m = uniform_filter(x, size=w, mode="reflect")
    sq = uniform_filter(x * x, size=w, mode="reflect")
    return np.sqrt(np.maximum(sq - m * m, 0.0))


def load_model(name, scaler):
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def flat_mask(t_gt):
    s = local_std(t_gt)
    return (s <= np.percentile(s, 25)) & (t_gt <= ambient_temperature(t_gt))


def pick_crop(t_gt, crop=CROP):
    F = flat_mask(t_gt)
    h, w = t_gt.shape
    best, best_cov = None, -1.0
    for y in range(0, h - crop, 32):
        for x in range(0, w - crop, 32):
            cov = F[y:y + crop, x:x + crop].mean()
            if cov > best_cov:
                best_cov, best = cov, (y, x)
    return best, best_cov


def main():
    scaler = load_scaler()
    models = {k: load_model(k, scaler) for k in ("edsr_lite", "esrgan")}

    # Pick a frame whose ESRGAN texture ratio is closest to the dataset-level mean,
    # so the panel is representative rather than a favourable outlier.
    from tsrf.metrics import cold_region_smoothness
    target = 1.113                      # aggregate R_tex for ESRGAN, Table I
    frames = load_splits()["test"]
    best = None
    for name in frames[:40]:
        raw = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
        lr = degrade(raw, scale=4, preset="classic")
        sr = np.asarray(models["esrgan"](lr), dtype=np.float64)
        r = cold_region_smoothness(raw_to_kelvin(sr),
                                   raw_to_kelvin(raw))["texture_ratio"]
        if np.isnan(r):
            continue
        if best is None or abs(r - target) < best[0]:
            best = (abs(r - target), name, r)
    _, name, r_frame = best
    print(f"frame {name}  frame-level ESRGAN R_tex = {r_frame:.3f} "
          f"(dataset mean {target})")

    raw = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
    (y, x), cov = pick_crop(raw_to_kelvin(raw))
    print(f"crop ({y},{x}), {cov*100:.0f}% of it belongs to the flat-cold set F")
    lr = degrade(raw, scale=4, preset="classic")
    panels = [("Ground truth", raw)]
    for key, label in (("edsr_lite", "EDSR (L1)"), ("esrgan", "ESRGAN")):
        panels.append((label, np.asarray(models[key](lr), dtype=np.float64)))

    sl = (slice(y, y + CROP), slice(x, x + CROP))
    crops_c = [raw_to_celsius(p)[sl] for _, p in panels]
    stds = [local_std(raw_to_kelvin(p))[sl] for _, p in panels]

    # Ratios are averaged over F only, exactly as (12) defines them.
    Fc = flat_mask(raw_to_kelvin(raw))[sl]
    ref_std = stds[0][Fc].mean()
    ratios = [s[Fc].mean() / ref_std for s in stds]

    vmin, vmax = crops_c[0].min(), crops_c[0].max()
    smax = max(s.max() for s in stds)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
    for j, ((label, _), c, s, ratio) in enumerate(
            zip(panels, crops_c, stds, ratios)):
        axes[0, j].imshow(c, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(label, fontsize=11)
        im = axes[1, j].imshow(s, cmap="viridis", vmin=0, vmax=smax)
        axes[1, j].set_xlabel(
            "reference" if j == 0 else rf"$R_{{\rm tex}} = {ratio:.2f}$", fontsize=11)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("apparent temp.", fontsize=10)
    axes[1, 0].set_ylabel(r"local std $\sigma_w$", fontsize=10)

    cb = fig.colorbar(im, ax=axes[1, :], fraction=0.025, pad=0.01)
    cb.set_label("K", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    for (label, _), s, ratio in zip(panels, stds, ratios):
        print(f"  {label:14s} sigma_w over F = {s[Fc].mean():.4f} K  "
              f"R_tex = {ratio:.3f}")


if __name__ == "__main__":
    main()

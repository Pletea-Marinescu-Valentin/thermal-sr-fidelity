"""Figure for M6/M7: the scale ladder, and the amplitude-correspondence plane.

Panel (a) plots the mean local standard deviation over the flat cold set against
window size, in log-log, for the reference and three reconstructions. The slope
is the variance-scaling exponent of M6, the vertical offset is what M4 measures.

Panel (b) places every model in the plane of the two quantities that actually
separate the failure modes: how much fine texture is emitted, and how much of it
corresponds to the measurement. Faithful reconstruction is the top-right corner;
over-smoothing falls to the left; fabrication falls to the bottom.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.radiometry import raw_to_kelvin  # noqa: E402
from tsrf.eval import (  # noqa: E402
    bicubic_upsampler, load_checkpoint, load_scaler, torch_upsampler,
)
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics.fractal import (  # noqa: E402
    DEFAULT_SCALES, flat_cold_mask, local_std, texture_correspondence,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
OUT = Path("paper/fig_scaling.pdf")
N_FRAMES = 150

PANEL = [
    ("bicubic",   "Bicubic",     "tab:gray",   "s"),
    ("edsr_lite", "EDSR (L1)",   "tab:blue",   "o"),
    ("rrdb",      "RRDB (L1)",   "tab:cyan",   "v"),
    ("esrgan",    "ESRGAN",      "tab:red",    "D"),
]


def load_model(name, scaler):
    if name == "bicubic":
        return bicubic_upsampler(4)
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def main():
    scaler = load_scaler()
    frames = load_splits()["test"][:N_FRAMES]
    print(f"{len(frames)} frames")

    hrs = [np.array(Image.open(SPLIT_DIR / n), dtype=np.float64) for n in frames]
    lrs = [degrade(h, scale=4, preset="classic") for h in hrs]
    gts = [raw_to_kelvin(h) for h in hrs]
    masks = [flat_cold_mask(g) for g in gts]

    ref_ladder = np.mean(
        [[local_std(g, w)[m].mean() for w in DEFAULT_SCALES]
         for g, m in zip(gts, masks)], axis=0)

    ladders, amps, corrs = {}, {}, {}
    for key, label, _, _ in PANEL:
        model = load_model(key, scaler)
        srs = [raw_to_kelvin(np.asarray(model(l), dtype=np.float64)) for l in lrs]
        ladders[key] = np.mean(
            [[local_std(s, w)[m].mean() for w in DEFAULT_SCALES]
             for s, m in zip(srs, masks)], axis=0)
        stats = [texture_correspondence(s, g) for s, g in zip(srs, gts)]
        amps[key] = float(np.nanmean([t["flat_hf_amp_ratio"] for t in stats]))
        corrs[key] = float(np.nanmean([t["flat_hf_corr"] for t in stats]))
        print(f"  {label:11s} amp={amps[key]:.3f}  corr={corrs[key]:.3f}  "
              f"ladder={np.round(ladders[key], 4)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    w = np.array(DEFAULT_SCALES, dtype=float)

    ax1.loglog(w, ref_ladder, "k-", marker="*", ms=9, lw=2.2, label="Reference",
               zorder=3)
    for key, label, colour, marker in PANEL:
        # ESRGAN is drawn dashed and on top: it lies on the reference curve, and
        # a solid line would simply hide it.
        dashed = key == "esrgan"
        ax1.loglog(w, ladders[key], color=colour, marker=marker, ms=4.5,
                   lw=1.4, label=label, ls="--" if dashed else "-",
                   zorder=4 if dashed else 2)
    ax1.set_xlabel(r"window size $w$ [px]")
    ax1.set_ylabel(r"$\overline{\sigma_w}$ over $F$ [K]")
    ax1.set_xticks(w)
    ax1.set_xticklabels([str(int(v)) for v in w])
    ax1.set_xticks([], minor=True)          # the log minor ticks only add clutter
    ax1.tick_params(axis="both", labelsize=8)
    ax1.grid(alpha=0.25, which="major", lw=0.4)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax1.set_title("(a) texture amplitude across scales", fontsize=9)

    ax2.axhline(0, color="0.8", lw=0.7)
    ax2.axvline(1, color="0.8", lw=0.7, ls="--")
    for key, label, colour, marker in PANEL:
        ax2.scatter(amps[key], corrs[key], color=colour, marker=marker, s=55,
                    zorder=4, label=label)
    ax2.scatter([1.0], [1.0], color="k", marker="*", s=120, zorder=5)
    ax2.annotate("faithful", (1.0, 1.0), textcoords="offset points",
                 xytext=(-6, -13), fontsize=7.5, ha="right")
    ax2.annotate("over-smoothed", (0.29, 0.20), textcoords="offset points",
                 xytext=(6, 9), fontsize=7.5)
    ax2.annotate("fabricated", (1.141, 0.039), textcoords="offset points",
                 xytext=(-8, 8), fontsize=7.5, ha="right")
    ax2.set_xlabel(r"texture amplitude ratio  (M7)")
    ax2.set_ylabel(r"texture correspondence $r$  (M7)")
    ax2.set_xlim(0.05, 1.35)
    ax2.set_ylim(-0.08, 1.12)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.set_title("(b) amplitude is not correspondence", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

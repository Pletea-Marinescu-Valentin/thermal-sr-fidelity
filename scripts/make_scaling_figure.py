"""Figure for M6/M7: the scale ladder, and the amplitude-correspondence plane.

Panel (a) plots the mean local standard deviation over the flat cold set against
window size, in log-log, for the reference and three reconstructions. The slope
is the variance-scaling exponent of M6, the vertical offset is what M4 measures.

Panel (b) places every model in the plane of the two quantities that actually
separate the failure modes: how much fine texture is emitted, and how much of it
corresponds to the measurement. Faithful reconstruction is the top-right corner;
over-smoothing falls to the left; fabrication falls to the bottom.
"""

import json
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
    DEFAULT_SCALES, flat_cold_mask, local_std,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
OUT = Path("paper/fig_scaling.pdf")
RESULTS = Path("results")
#: Ladders cost a forward pass per model per frame, so they are cached and a
#: layout change does not mean re-running the models.
CACHE = RESULTS / "adas_figure_ladders.json"
N_FRAMES = 1144

PANEL = [
    ("bicubic",   "Bicubic",     "tab:gray",   "s"),
    ("edsr_lite", "EDSR (L1)",   "tab:blue",   "o"),
    ("rrdb",      "RRDB (L1)",   "tab:cyan",   "v"),
    ("esrgan",    "ESRGAN",      "tab:red",    "D"),
]


def geometric_mean_ladder(per_frame):
    """Mean ladder whose fitted slope equals the mean of the per-frame slopes.

    The exponent is a least-squares slope, hence a linear functional of the log
    values, so averaging in log space commutes with fitting and averaging in
    linear space does not. Plotting an arithmetic mean therefore draws a curve
    whose slope is not the quantity reported anywhere in the paper: on the
    FLAME 3 reference it reads 0.34 against the 0.618 the tables quote. The
    geometric mean is the curve a reader can measure and land on the number.
    """
    import numpy as _np
    return _np.exp(_np.mean(_np.log(_np.asarray(per_frame, dtype=float)), axis=0))


def load_model(name, scaler):
    if name == "bicubic":
        return bicubic_upsampler(4)
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def cached_ladders(force=False):
    if CACHE.exists() and not force:
        c = json.loads(CACHE.read_text())
        if c["n_frames"] == N_FRAMES:
            print(f"reusing cached ladders over {N_FRAMES} frames")
            return (np.array(c["reference"]),
                    {k: np.array(v) for k, v in c["models"].items()})
    return None


def compute_ladders(scaler, frames):
    """Stream the test split once, accumulating log-ladders per model.

    One frame is held at a time. Materialising the whole split instead costs
    about 3 GB for the references alone, before the three derived fields and
    the four reconstructions, which on this machine pages rather than runs.
    """
    models = {key: load_model(key, scaler) for key, _, _, _ in PANEL}
    n_scales = len(DEFAULT_SCALES)
    log_sums = {k: np.zeros(n_scales) for k in ["__ref__", *models]}

    for i, name in enumerate(frames, 1):
        hr = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
        gt = raw_to_kelvin(hr)
        mask = flat_cold_mask(gt)
        lr = degrade(hr, scale=4, preset="classic")

        log_sums["__ref__"] += np.log([local_std(gt, w)[mask].mean()
                                       for w in DEFAULT_SCALES])
        for key, model in models.items():
            sr = raw_to_kelvin(np.asarray(model(lr), dtype=np.float64))
            log_sums[key] += np.log([local_std(sr, w)[mask].mean()
                                     for w in DEFAULT_SCALES])
        if i % 100 == 0:
            print(f"    {i}/{len(frames)}")

    n = len(frames)
    ref = np.exp(log_sums.pop("__ref__") / n)
    return ref, {k: np.exp(v / n) for k, v in log_sums.items()}


def main():
    scaler = load_scaler()
    frames = load_splits()["test"][:N_FRAMES]
    print(f"{len(frames)} frames")

    cached = cached_ladders()
    if cached:
        ref_ladder, ladders = cached
    else:
        ref_ladder, ladders = compute_ladders(scaler, frames)
        RESULTS.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(
            {"n_frames": len(frames), "scales": list(DEFAULT_SCALES),
             "reference": ref_ladder.tolist(),
             "models": {k: v.tolist() for k, v in ladders.items()}}, indent=1))

    # The plane is read from the aggregates rather than recomputed here, so
    # every point a reader compares against the tables is the number the tables
    # print, over all test frames rather than a subset of them.
    amps, corrs = {}, {}
    for key, label, _, _ in PANEL:
        agg = json.loads((RESULTS / f"{key}_classic.json").read_text())["aggregate"]
        amps[key] = agg["m7_flat_hf_amp_ratio"]["mean"]
        corrs[key] = agg["m7_flat_hf_corr"]["mean"]
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
    ax2.scatter([1.0], [1.0], color="0.6", marker="*", s=120, zorder=5)
    ax2.annotate("exact copy\n(not attainable at $\\times 4$)", (1.0, 1.0),
                 textcoords="offset points", xytext=(-8, -8), fontsize=7,
                 ha="right", va="top", color="0.45")
    # The point a reconstruction can actually reach: most of this band is not
    # in the input, so the target is the ceiling, not the reference itself.
    ceiling = json.loads((RESULTS / "m7_ceiling.json").read_text())
    cx, cy = ceiling["linear_ceiling_amp"], ceiling["linear_ceiling_corr"]
    ax2.scatter([cx], [cy], color="k", marker="*", s=130, zorder=6)
    ax2.annotate("attainable\n(best linear)", (cx, cy),
                 textcoords="offset points", xytext=(62, -22), fontsize=7,
                 ha="center", arrowprops=dict(arrowstyle="-", color="0.5",
                                              lw=0.7, shrinkB=6))
    ax2.annotate("over-smoothed", (0.30, 0.17), textcoords="offset points",
                 xytext=(26, 26), fontsize=7.5, ha="center",
                 arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7,
                                 shrinkB=6))
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

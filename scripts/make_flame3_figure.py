"""Figure for the FLAME 3 replication.

Panel (a) is the M6 ladder on FLAME 3: mean local standard deviation over the
flat cold set against window size, for the reference and four reconstructions.
It is the same construction as the ADAS panel, on a different instrument, a
different scene class and a reference whose own exponent differs, so the
question it answers is whether the pattern is a property of the sensor or of
the objective.

Panel (b) puts both datasets in the amplitude-correspondence plane at once and
draws the displacement the adversarial objective produces on each, from the
RRDB it was fine-tuned from to ESRGAN. The two arrows point the same way while
starting and ending in different places, which is the whole claim of the
replication: the direction transfers, the magnitudes do not. The constructed
controls of the positive-control experiment are drawn as open symbols, so the
model positions can be read against references whose correct answer is fixed.

Run: .venv/Scripts/python.exe scripts/make_flame3_figure.py [--frames 150]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.flame3 import Flame3Scaler, frames as flame3_frames  # noqa: E402
from tsrf.data.flame3 import load_celsius  # noqa: E402
from tsrf.data.radiometry import ZERO_CELSIUS_K  # noqa: E402
from tsrf.eval import (  # noqa: E402
    bicubic_upsampler, load_checkpoint, torch_upsampler,
)
from tsrf.metrics.fractal import (  # noqa: E402
    DEFAULT_SCALES, flat_cold_mask, local_std,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import flame3_specs  # noqa: E402

RESULTS = Path("results")
OUT = Path("paper/fig_flame3.pdf")
#: Ladders need a forward pass per model per frame, so they are cached. Layout
#: changes then cost nothing, which is the difference between tuning a figure
#: and re-running the models to move a label.
CACHE = RESULTS / "f3_figure_ladders.json"

PANEL = [
    ("bicubic",   "Bicubic",   "tab:gray", "s"),
    ("edsr_lite", "EDSR (L1)", "tab:blue", "o"),
    ("rrdb",      "RRDB (L1)", "tab:cyan", "v"),
    ("esrgan",    "ESRGAN",    "tab:red",  "D"),
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
    spec = flame3_specs()[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def ladders(n_frames, force=False):
    """Reference and per-model scale ladders over the FLAME 3 test frames."""
    if CACHE.exists() and not force:
        c = json.loads(CACHE.read_text())
        if c["n_frames"] == n_frames:
            print(f"reusing cached ladders over {n_frames} frames")
            return np.array(c["reference"]), {k: np.array(v)
                                              for k, v in c["models"].items()}
    scaler = Flame3Scaler()
    paths = flame3_frames("cv")[:n_frames]
    if not paths:
        raise FileNotFoundError("no FLAME 3 CV frames; run extract_flame3.py")
    print(f"{len(paths)} frames")

    celsius = [load_celsius(p) for p in paths]
    gts = [c + ZERO_CELSIUS_K for c in celsius]
    masks = [flat_cold_mask(g) for g in gts]
    # The rasters are Celsius, so the NETD applies as a temperature directly;
    # the ADAS counts-per-Kelvin default would inject 25 times the sensor noise.
    lrs = [degrade(c, scale=4, preset="classic", k_per_count=1.0)
           for c in celsius]

    ref = geometric_mean_ladder(
        [[local_std(g, w)[m].mean() for w in DEFAULT_SCALES]
         for g, m in zip(gts, masks)])
    print(f"  {'Reference':11s} ladder={np.round(ref, 4)}")

    out = {}
    for key, label, _, _ in PANEL:
        model = load_model(key, scaler)
        srs = [np.asarray(model(lr), dtype=np.float64) + ZERO_CELSIUS_K
               for lr in lrs]
        out[key] = geometric_mean_ladder(
            [[local_std(s, w)[m].mean() for w in DEFAULT_SCALES]
             for s, m in zip(srs, masks)])
        print(f"  {label:11s} ladder={np.round(out[key], 4)}")
    CACHE.write_text(json.dumps(
        {"n_frames": len(paths), "scales": list(DEFAULT_SCALES),
         "reference": ref.tolist(),
         "models": {k: v.tolist() for k, v in out.items()}}, indent=1))
    return ref, out


def plane_points():
    """(amplitude, correspondence) for every arm and control, both datasets."""
    def agg(prefix, model, key):
        path = RESULTS / f"{prefix}{model}_classic.json"
        return json.loads(path.read_text())["aggregate"][key]["mean"]

    def control(prefix, arm, key):
        path = RESULTS / f"{prefix}positive_control.json"
        v = json.loads(path.read_text())["aggregate"][arm][key]
        # The control file stores bootstrap summaries, the model files store
        # plain means for the same keys; accept either rather than assume.
        return v["mean"] if isinstance(v, dict) else v

    models, controls = {}, {}
    for tag, prefix in (("adas", ""), ("flame3", "f3_")):
        models[tag] = {k: (agg(prefix, k, "m7_flat_hf_amp_ratio"),
                           agg(prefix, k, "m7_flat_hf_corr"))
                       for k, _, _, _ in PANEL}
        controls[tag] = {a: (control(prefix, a, "m7_flat_hf_amp_ratio"),
                             control(prefix, a, "m7_flat_hf_corr"))
                         for a in ("attenuated", "fabricated_local")}
    return models, controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=738)
    ap.add_argument("--force", action="store_true",
                    help="recompute the ladders instead of using the cache")
    args = ap.parse_args()

    ref, lad = ladders(args.frames, args.force)
    models, controls = plane_points()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    w = np.array(DEFAULT_SCALES, dtype=float)

    ax1.loglog(w, ref, "k-", marker="*", ms=9, lw=2.2, label="Reference",
               zorder=3)
    for key, label, colour, marker in PANEL:
        dashed = key == "esrgan"
        ax1.loglog(w, lad[key], color=colour, marker=marker, ms=4.5, lw=1.4,
                   label=label, ls="--" if dashed else "-",
                   zorder=4 if dashed else 2)
    ax1.set_xlabel(r"window size $w$ [px]")
    ax1.set_ylabel(r"$\overline{\sigma_w}$ over $F$ [K]")
    ax1.set_xticks(w)
    ax1.set_xticklabels([str(int(v)) for v in w])
    ax1.set_xticks([], minor=True)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.grid(alpha=0.25, which="major", lw=0.4)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax1.set_title("(a) FLAME 3: texture amplitude across scales", fontsize=9)

    ax2.axhline(0, color="0.8", lw=0.7)
    ax2.axvline(1, color="0.8", lw=0.7, ls="--")
    for tag, filled, dname in (("adas", False, "ADAS"),
                               ("flame3", True, "FLAME 3")):
        for key, label, colour, marker in PANEL:
            x, y = models[tag][key]
            ax2.scatter(x, y, marker=marker, s=48, zorder=4,
                        facecolor=colour if filled else "none",
                        edgecolor=colour, linewidths=1.2)
        # The displacement the objective produces, on each dataset.
        x0, y0 = models[tag]["rrdb"]
        x1, y1 = models[tag]["esrgan"]
        ax2.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                     arrowprops=dict(arrowstyle="-|>", color="0.35",
                                     lw=1.3, shrinkA=6, shrinkB=6,
                                     linestyle="-" if filled else "--"))
        # Labelled at the arrow head: the tails sit almost on top of each
        # other, the heads are far apart, which is itself the finding.
        ax2.annotate(dname, xy=(x1, y1), textcoords="offset points",
                     xytext=(-2, -14), fontsize=7.5, color="0.3", ha="center",
                     fontweight="bold" if filled else "normal")
        for arm, mark in (("attenuated", "P"), ("fabricated_local", "X")):
            cx, cy = controls[tag][arm]
            ax2.scatter(cx, cy, marker=mark, s=46, zorder=5, color="0.45",
                        alpha=0.9 if filled else 0.45)

    ax2.scatter([1.0], [1.0], color="k", marker="*", s=110, zorder=6)
    ax2.annotate("faithful", (1.0, 1.0), textcoords="offset points",
                 xytext=(-8, -4), fontsize=7.5, ha="right", va="center")
    ax2.annotate("attenuation controls", (0.36, 0.88),
                 textcoords="offset points", xytext=(11, 2), fontsize=7,
                 color="0.35", va="center")
    ax2.annotate("fabrication controls", (1.03, 0.011),
                 textcoords="offset points", xytext=(14, 12), fontsize=7,
                 color="0.35", ha="center")
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="none", mec="0.3",
               label="ADAS"),
        Line2D([], [], marker="o", ls="none", mfc="0.3", mec="0.3",
               label="FLAME 3"),
    ], fontsize=7, frameon=False, loc="upper left", handletextpad=0.3)
    ax2.set_xlabel(r"texture amplitude ratio  (M7)")
    ax2.set_ylabel(r"texture correspondence $r$  (M7)")
    ax2.set_xlim(0.05, 1.42)
    ax2.set_ylim(-0.12, 1.12)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.set_title("(b) the same displacement on both datasets", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

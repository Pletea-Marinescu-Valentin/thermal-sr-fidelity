"""Figure 1: the protocol on one line, from measurement to decision.

A reader meeting this work for the first time has to hold five things at once:
that the reference is a radiometric measurement rather than a picture, that the
low-resolution input is synthesised from it in the float domain, that the
metrics are computed in Kelvin, that they split into perceptual and fidelity
groups which disagree, and that the comparison is paired at video level. The
figure carries that structure so the text does not have to repeat it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("paper/fig_protocol.pdf")

INK = "#1a1a1a"
MEASURE = "#d8e6f2"
MODEL = "#e8e8e8"
FIDELITY = "#f6dcd6"
PERCEPT = "#e6e6e6"


def box(ax, x, y, w, h, label, sub=None, face=MODEL, fontsize=7.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8, edgecolor=INK, facecolor=face))
    cy = y + h / 2 + (0.035 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=fontsize, color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.05, sub, ha="center", va="center",
                fontsize=6.2, color="#555555", style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color=INK,
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.03, label, ha="center",
                va="bottom", fontsize=6.2, color="#555555")


def main():
    fig, ax = plt.subplots(figsize=(7.1, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    h, y = 0.20, 0.60
    box(ax, 0.005, y, 0.175, h, "16-bit T-linear HR",
        "measurement, Kelvin", face=MEASURE)
    box(ax, 0.225, y, 0.155, h, "Degradation", r"$\times 4$, float domain")
    box(ax, 0.425, y, 0.145, h, "Reconstruction", "5 methods")
    box(ax, 0.615, y, 0.175, h, "Metrics in Kelvin", "on HR grid")
    box(ax, 0.835, y, 0.160, h, "Paired tests", "video-level")

    mid = y + h / 2
    for x0, x1 in ((0.180, 0.225), (0.380, 0.425), (0.570, 0.615), (0.790, 0.835)):
        arrow(ax, x0, mid, x1, mid)

    # The reference re-enters at the metric stage: every metric is a comparison
    # against the measurement, which is the point of the whole protocol.
    ax.add_patch(FancyArrowPatch(
        (0.0925, y), (0.6975, y - 0.012), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color=INK, linestyle=(0, (3, 2)),
        connectionstyle="arc3,rad=0.22"))
    ax.text(0.395, 0.445, "reference", ha="center", va="center",
            fontsize=6.2, color="#555555", style="italic")

    # The two metric groups hang off the metric stage and disagree with each
    # other; that disagreement is the paper, so it is drawn rather than stated.
    gy, gh = 0.135, 0.165
    box(ax, 0.455, gy, 0.215, gh, "PSNR / SSIM / LPIPS", "perceptual",
        face=PERCEPT, fontsize=6.8)
    box(ax, 0.740, gy, 0.215, gh, "M0–M7", "radiometric fidelity",
        face=FIDELITY, fontsize=6.8)

    for xt in (0.5625, 0.8475):
        ax.add_patch(FancyArrowPatch(
            (0.7025, y - 0.005), (xt, gy + gh + 0.005), arrowstyle="-|>",
            mutation_scale=8, linewidth=0.8, color=INK,
            connectionstyle="arc3,rad=0.0"))

    ax.annotate("", xy=(0.735, gy + gh / 2), xytext=(0.675, gy + gh / 2),
                arrowprops=dict(arrowstyle="<|-|>", linewidth=0.8,
                                color="#8c3b28", mutation_scale=7))
    ax.text(0.705, gy - 0.025, "disagree", ha="center", va="top",
            fontsize=6.4, color="#8c3b28")

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

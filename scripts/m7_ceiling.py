"""How much of the reference fine texture is recoverable from the input at all?

M7 is reported as a relative score because decimation removes most of the band
it measures, so the attainable correlation is far below unity and comparing a
model against 1 would be meaningless. That is honest but incomplete: it leaves
open whether r = 0.17 is near the ceiling or far from it, and therefore whether
the regressors are doing well or badly in absolute terms.

This computes a ceiling that is well defined rather than assumed. Over the flat
cold set we fit the least-squares linear predictor of the reference's high-pass
residual from a neighbourhood of the bicubically upsampled low-resolution
input, and report the correlation it achieves. Three properties make the number
usable:

  * it is an upper bound for any reconstruction that is a linear function of
    the same input, since it is the optimal such function by construction;
  * it is fitted on the very frames it is evaluated on, with no held-out split,
    so it is optimistic on purpose: a ceiling should not be beatable by luck;
  * it is *not* an upper bound for a nonlinear model, which can in principle
    exceed it, and the regressors here do. That is itself the useful reading:
    exceeding the linear ceiling is evidence of genuine nonlinear recovery,
    and falling far below it is not explained by the information gap.

Run: .venv/Scripts/python.exe scripts/m7_ceiling.py [--frames 200] [--dataset adas]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsrf.data.degradation import degrade, imresize_bicubic  # noqa: E402
from tsrf.data.radiometry import raw_to_kelvin  # noqa: E402
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics.fractal import FLAT_WINDOW, flat_cold_mask  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
RESULTS = Path("results")

#: Half-width of the neighbourhood the predictor may use. At x4 the input
#: determines the output over a support of a few pixels, so 2 (a 5x5 patch, 25
#: coefficients) covers more than one low-resolution sample in every direction.
RADIUS = 2


def neighbourhood(field, radius=RADIUS):
    """Stack of shifted copies: one column per offset in the patch."""
    shifts = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifts.append(np.roll(np.roll(field, dy, axis=0), dx, axis=1))
    return np.stack(shifts, axis=-1)


def high_pass(x, window=FLAT_WINDOW):
    return x - uniform_filter(x, size=window, mode="reflect")


def frame_source(dataset, n):
    if dataset == "flame3":
        from tsrf.data.flame3 import ZERO_CELSIUS_K, frames as f3_frames
        from tsrf.data.flame3 import load_celsius

        def load(p):
            celsius = load_celsius(p)
            lr = degrade(celsius, scale=4, preset="classic", k_per_count=1.0)
            return celsius + ZERO_CELSIUS_K, lr + ZERO_CELSIUS_K

        return [(p.name, (lambda p=p: load(p))) for p in f3_frames("cv")[:n]]

    def load(name):
        hr = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
        return raw_to_kelvin(hr), raw_to_kelvin(degrade(hr, scale=4,
                                                        preset="classic"))

    return [(nm, (lambda nm=nm: load(nm))) for nm in load_splits()["test"][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adas", choices=("adas", "flame3"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--radius", type=int, default=RADIUS,
                    help="half-width of the predictor's support, in pixels")
    args = ap.parse_args()
    radius = args.radius

    source = frame_source(args.dataset, args.frames)
    if not source:
        sys.exit(f"no frames for {args.dataset}")
    print(f"{len(source)} frames, {args.dataset}")

    # Pass one: accumulate the normal equations over the flat cold set.
    p = (2 * radius + 1) ** 2
    xtx, xty = np.zeros((p, p)), np.zeros(p)
    for i, (_, load) in enumerate(source, 1):
        gt, lr = load()
        up = imresize_bicubic(lr, 4.0) if lr.shape != gt.shape else lr
        mask = flat_cold_mask(gt)
        if not mask.any():
            continue
        x = neighbourhood(high_pass(up), radius)[mask]
        y = high_pass(gt)[mask]
        xtx += x.T @ x
        xty += x.T @ y
        if i % 50 == 0:
            print(f"    fit {i}/{len(source)}")

    coef = np.linalg.solve(xtx + 1e-6 * np.eye(p), xty)

    # Pass two: the correlation that predictor achieves, aggregated as M7 is.
    corrs, amps = [], []
    for i, (_, load) in enumerate(source, 1):
        gt, lr = load()
        up = imresize_bicubic(lr, 4.0) if lr.shape != gt.shape else lr
        mask = flat_cold_mask(gt)
        if not mask.any():
            continue
        pred = neighbourhood(high_pass(up), radius)[mask] @ coef
        truth = high_pass(gt)[mask]
        if pred.std() < 1e-9 or truth.std() < 1e-9:
            continue
        corrs.append(float(np.corrcoef(truth, pred)[0, 1]))
        amps.append(float(pred.std() / truth.std()))
        if i % 50 == 0:
            print(f"    score {i}/{len(source)}")

    out = {
        "dataset": args.dataset,
        "n_frames": len(corrs),
        "radius": radius,
        "window": FLAT_WINDOW,
        "linear_ceiling_corr": float(np.mean(corrs)),
        "linear_ceiling_corr_sd": float(np.std(corrs)),
        "linear_ceiling_amp": float(np.mean(amps)),
    }
    prefix = "f3_" if args.dataset == "flame3" else ""
    suffix = "" if radius == RADIUS else f"_r{radius}"
    path = RESULTS / f"{prefix}m7_ceiling{suffix}.json"
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\noptimal linear predictor from the same input:")
    print(f"  correspondence r = {out['linear_ceiling_corr']:.3f} "
          f"(sd {out['linear_ceiling_corr_sd']:.3f} across frames)")
    print(f"  amplitude ratio  = {out['linear_ceiling_amp']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Does the negative result on discrete hallucination survive its thresholds?

M2 reports that invented hot regions are rare. A negative result is only worth
as much as its most permissive assumption, and M2 carries four beyond the
detection threshold delta, which the paper already sweeps:

  min_area        smallest region that counts at all
  iou_tau         overlap at which a predicted region is matched to a real one
  lr_relax        how far below delta the low-resolution input is thresholded
  lr_overlap_tau  how much low-resolution support excuses an unmatched region

Three of these are set generously toward the models. A region the input barely
hints at is excused rather than counted, which is the right default for a claim
about *invention*, but it means the reported rate is a lower bound. The
question this answers is whether the negative survives at the strict end of
each: if tightening lr_overlap_tau or lr_relax makes hallucinations appear,
the result is a property of the thresholds and not of the models.

One pass over the frames, every setting scored on the same reconstructions, so
the comparison is not confounded by sampling.

Run: .venv/Scripts/python.exe scripts/m2_sensitivity.py [--frames 200]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.radiometry import raw_to_kelvin  # noqa: E402
from tsrf.eval import (  # noqa: E402
    bicubic_upsampler, load_checkpoint, load_scaler, torch_upsampler,
)
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics import hallucination_metrics  # noqa: E402
from tsrf.metrics.hotspot import DEFAULT_MIN_AREA  # noqa: E402
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
RESULTS = Path("results")
OUT = RESULTS / "m2_sensitivity.json"

MODELS = [("bicubic", "Bicubic"), ("edsr_lite", "EDSR"), ("esrgan", "ESRGAN")]

DEFAULTS = dict(delta_k=10.0, min_area=DEFAULT_MIN_AREA, iou_tau=0.1,
                lr_relax=0.5, lr_overlap_tau=0.05)

#: One axis at a time around the defaults. The last value of each list is the
#: strict end, where the criterion is least forgiving of the models.
SWEEPS = {
    "min_area": [10, DEFAULT_MIN_AREA, 50, 100],
    "iou_tau": [0.05, 0.1, 0.3, 0.5],
    "lr_relax": [0.0, 0.25, 0.5, 0.75, 1.0],
    "lr_overlap_tau": [0.01, 0.05, 0.2, 0.5],
}


def load_model(name, scaler):
    if name == "bicubic":
        return bicubic_upsampler(4)
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def settings():
    """(axis, value, kwargs) for every point of every sweep."""
    for axis, values in SWEEPS.items():
        for value in values:
            yield axis, value, {**DEFAULTS, axis: value}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    scaler = load_scaler()
    names = load_splits()["test"][:args.frames]
    models = {key: load_model(key, scaler) for key, _ in MODELS}
    upsampler = bicubic_upsampler(4)

    grid = list(settings())
    halluc = {(k, a, v): [] for k, _ in MODELS for a, v, _ in grid}
    missed = {(k, a, v): [] for k, _ in MODELS for a, v, _ in grid}

    for i, name in enumerate(names, 1):
        hr = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
        gt = raw_to_kelvin(hr)
        lr = degrade(hr, scale=4, preset="classic")
        lr_up = raw_to_kelvin(np.asarray(upsampler(lr), dtype=np.float64))
        for key, _ in MODELS:
            sr = raw_to_kelvin(np.asarray(models[key](lr), dtype=np.float64))
            for axis, value, kwargs in grid:
                m = hallucination_metrics(sr, gt, lr_up, **kwargs)
                halluc[(key, axis, value)].append(m["n_hallucinated"])
                missed[(key, axis, value)].append(m["n_missed"])
        if i % 25 == 0:
            print(f"    {i}/{len(names)}")

    out = {"n_frames": len(names), "defaults": DEFAULTS, "sweeps": SWEEPS,
           "hallucinated_per_frame": {}, "missed_per_frame": {}}
    for (key, axis, value), vals in halluc.items():
        out["hallucinated_per_frame"][f"{key}|{axis}|{value}"] = float(np.mean(vals))
    for (key, axis, value), vals in missed.items():
        out["missed_per_frame"][f"{key}|{axis}|{value}"] = float(np.mean(vals))

    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))

    print(f"\nhallucinated regions per frame, {len(names)} frames, delta = 10 K")
    for axis, values in SWEEPS.items():
        print(f"\n  {axis}  (default {DEFAULTS[axis]})")
        head = "".join(f"{v:>10}" for v in values)
        print(f"    {'model':<10}{head}")
        for key, label in MODELS:
            row = "".join(f"{out['hallucinated_per_frame'][f'{key}|{axis}|{v}']:10.3f}"
                          for v in values)
            print(f"    {label:<10}{row}")

    worst = max(out["hallucinated_per_frame"].values())
    print(f"\nlargest rate anywhere in the sweep: {worst:.3f} per frame")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

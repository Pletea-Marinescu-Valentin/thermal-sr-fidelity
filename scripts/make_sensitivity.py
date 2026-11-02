r"""Are the M6/M7 conclusions an artefact of the metrics' design choices?

M6 fixes a window ladder and M4, M6 and M7 all inherit M4's flat set, the
quietest quartile of the reference below ambient. Both are judgement calls, and
a protocol paper has to show that its conclusions do not depend on them. This
sweeps each choice and reports whether the ordering of the models survives.

It also measures correspondence scale by scale. M7 uses a single high-pass at
the M4 window; splitting it across the ladder shows where in the spectrum the
low-resolution input still carries the measurement and where a model is free to
invent, which is the empirical version of the bound discussed in Section V-D.

Frames are processed one at a time and only scalars are kept. Holding the
reference and every model's output for the whole subset instead costs several
gigabytes and pages the machine to a standstill, which is easy to do by
accident here: a 640x512 float64 frame is 2.6 MB and the sweeps multiply it by
models, window sizes and mask definitions.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.radiometry import raw_to_kelvin  # noqa: E402
from tsrf.eval import (  # noqa: E402
    bicubic_upsampler, load_checkpoint, load_scaler, torch_upsampler,
)
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics.fractal import (  # noqa: E402
    FLAT_WINDOW, _fit_loglog, flat_cold_mask, local_std,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
OUT = Path("results/sensitivity.json")
N_FRAMES = 200
MODELS = ["bicubic", "edsr_lite", "esrgan"]

LADDERS = {
    "3-9": (3, 5, 9),
    "3-17 (default)": (3, 5, 9, 17),
    "3-33": (3, 5, 9, 17, 33),
    "5-33": (5, 9, 17, 33),
    "3-17 dense": (3, 5, 7, 9, 13, 17),
}
PERCENTILES = (10.0, 25.0, 40.0, 50.0)
CORR_WINDOWS = (3, 5, 9, 17, 33)
DEFAULT_LADDER = LADDERS["3-17 (default)"]


def load_model(name, scaler):
    if name == "bicubic":
        return bicubic_upsampler(4)
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    return torch_upsampler(net, scaler)


def hurst_on(cache, field_key, field, mask, scales):
    """Fit the exponent, reusing local_std across ladders that share a window."""
    ladder = []
    for w in scales:
        key = (field_key, w)
        if key not in cache:
            cache[key] = local_std(field, w)
        ladder.append(float(cache[key][mask].mean()))
    return _fit_loglog(scales, ladder)[0]


def corr_at(hp_pred, hp_gt, mask):
    a, b = hp_gt[mask], hp_pred[mask]
    if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    scaler = load_scaler()
    frames = load_splits()["test"][:N_FRAMES]
    models = {m: load_model(m, scaler) for m in MODELS}
    print(f"{len(frames)} frames, models {MODELS}", flush=True)

    acc = defaultdict(list)
    for i, fn in enumerate(frames, 1):
        raw = np.array(Image.open(SPLIT_DIR / fn), dtype=np.float64)
        gt = raw_to_kelvin(raw)
        lr = degrade(raw, scale=4, preset="classic")
        masks = {p: flat_cold_mask(gt, FLAT_WINDOW, p) for p in PERCENTILES}

        for name, model in models.items():
            sr = raw_to_kelvin(np.asarray(model(lr), dtype=np.float64))
            cache = {}

            for label, scales in LADDERS.items():
                d = (hurst_on(cache, "sr", sr, masks[25.0], scales)
                     - hurst_on(cache, "gt", gt, masks[25.0], scales))
                acc[f"ladder|{label}|{name}"].append(d)

            for p in PERCENTILES:
                d = (hurst_on(cache, "sr", sr, masks[p], DEFAULT_LADDER)
                     - hurst_on(cache, "gt", gt, masks[p], DEFAULT_LADDER))
                acc[f"pct_dh|{p}|{name}"].append(d)

            for w in CORR_WINDOWS:
                hp_s = sr - uniform_filter(sr, size=w, mode="reflect")
                hp_g = gt - uniform_filter(gt, size=w, mode="reflect")
                acc[f"corr_w|{w}|{name}"].append(corr_at(hp_s, hp_g, masks[25.0]))
                if w == FLAT_WINDOW:
                    for p in PERCENTILES:
                        acc[f"pct_r|{p}|{name}"].append(
                            corr_at(hp_s, hp_g, masks[p]))
            # M7's own window is 7, which is not on the ladder; add it.
            hp_s = sr - uniform_filter(sr, size=FLAT_WINDOW, mode="reflect")
            hp_g = gt - uniform_filter(gt, size=FLAT_WINDOW, mode="reflect")
            for p in PERCENTILES:
                acc[f"pct_r|{p}|{name}"].append(corr_at(hp_s, hp_g, masks[p]))

        if i % 50 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    mean = {k: float(np.nanmean(v)) for k, v in acc.items()}

    def table(title, prefix, keys):
        print(f"\n{title}:")
        print(f"  {'':<18}" + "".join(f"{m:>14}" for m in MODELS))
        rows = {}
        for k in keys:
            vals = [mean.get(f"{prefix}|{k}|{m}", float('nan')) for m in MODELS]
            rows[str(k)] = dict(zip(MODELS, vals))
            print(f"  {str(k):<18}" + "".join(f"{v:>14.3f}" for v in vals))
        return rows

    out = {"n_frames": len(frames), "models": MODELS}
    out["ladders"] = table("M6 delta Hurst by window ladder (flat pct 25)",
                           "ladder", list(LADDERS))
    out["dh_by_percentile"] = table("M6 delta Hurst by flat-set percentile",
                                    "pct_dh", list(PERCENTILES))
    out["r_by_percentile"] = table("M7 correspondence by flat-set percentile",
                                   "pct_r", list(PERCENTILES))
    out["corr_by_window"] = table("M7 correspondence by high-pass window",
                                  "corr_w", list(CORR_WINDOWS))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

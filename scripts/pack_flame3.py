"""Pack FLAME 3 into memory-mapped arrays, in Celsius.

Mirrors `pack_dataset.py` for the ADAS track, with one difference that matters:
FLAME 3 stores temperature, not sensor counts, so the cache holds float32
Celsius rather than uint16 counts, and the two cameras are converted to that
common scale on the way in.

The split is by flight plot and by instrument, so nothing leaks:

    train  NADIR plot 2   Autel XT709, Hanna Hammock, Florida
    val    NADIR plot 1   Autel XT709, same burn, a different plot
    test   CV subset      DJI M30T, Sycan Marsh, Oregon

Test is therefore a different camera, a different burn, a different state and a
different season from anything the models see while training.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.degradation import degrade  # noqa: E402
from tsrf.data.flame3 import frames as flame3_frames  # noqa: E402
from tsrf.data.flame3 import load_celsius  # noqa: E402

CACHE = Path("data/cache")

SPLITS = {
    "f3train": ("nadir", "plot2"),
    "f3val": ("nadir", "plot1"),
    "f3test": ("cv", None),
}


def pack(split, scale=4, preset="classic"):
    subset, group = SPLITS[split]
    paths = flame3_frames(subset, group)
    if not paths:
        sys.exit(f"{split}: no frames found; run scripts/extract_flame3.py first")

    probe = load_celsius(paths[0])
    h, w = probe.shape
    lh, lw = h // scale, w // scale
    print(f"{split}: {len(paths)} frames from {subset}/{group or 'all'}, "
          f"HR {h}x{w} -> LR {lh}x{lw}")

    CACHE.mkdir(parents=True, exist_ok=True)
    hr_path = CACHE / f"{split}_hr_c_f32.npy"
    lr_path = CACHE / f"{split}_lr_{preset}_f32.npy"
    hr = np.lib.format.open_memmap(hr_path, mode="w+", dtype=np.float32,
                                   shape=(len(paths), h, w))
    lr = np.lib.format.open_memmap(lr_path, mode="w+", dtype=np.float32,
                                   shape=(len(paths), lh, lw))

    t0 = time.time()
    for i, p in enumerate(paths):
        c = load_celsius(p)
        if c.shape != (h, w):
            sys.exit(f"{p}: shape {c.shape}, expected {(h, w)}")
        hr[i] = c.astype(np.float32)
        # Noise is added at training time, so pack the noise-free degradation.
        lr[i] = degrade(c, scale=scale, preset=preset, netd_k=0.0).astype(np.float32)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}  {time.time() - t0:.0f}s")

    hr.flush()
    lr.flush()
    (CACHE / f"{split}_index.json").write_text(json.dumps(
        {"frames": [p.as_posix() for p in paths], "scale": scale,
         "preset": preset, "hr_shape": [h, w], "lr_shape": [lh, lw],
         "subset": subset, "group": group, "units": "celsius"}, indent=1))
    print(f"  wrote {hr_path.name} ({hr.nbytes / 1e9:.2f} GB) and "
          f"{lr_path.name} ({lr.nbytes / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--preset", default="classic")
    args = ap.parse_args()
    for s in args.splits:
        if s not in SPLITS:
            sys.exit(f"unknown split {s!r}, expected one of {list(SPLITS)}")
        pack(s, preset=args.preset)


if __name__ == "__main__":
    main()

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.splits import (  # noqa: E402
    assert_no_video_leakage, group_by_video, parse_frame_name, _stable_unit,
)
from tsrf.data.radiometry import (  # noqa: E402
    RadiometricScaler, TLINEAR_K_PER_COUNT, raw_to_celsius,
)

ADAS = Path("data/adas")
OUT = Path("configs")
SEED = 1337
VAL_FRACTION = 0.12          # of the official-train videos
UINT16_BINS = 1 << 16


def list_tiffs(split_dir):
    return sorted(p.name for p in (ADAS / split_dir / "analyticsData").glob("*.tiff"))


def build_splits():
    train_pool = list_tiffs("images_thermal_train")
    test = list_tiffs("images_thermal_val")

    videos = sorted(group_by_video(train_pool))
    val_videos = {v for v in videos if _stable_unit(v, SEED) < VAL_FRACTION}

    splits = {"train": [], "val": [], "test": test}
    for name in train_pool:
        video, _ = parse_frame_name(name)
        splits["val" if video in val_videos else "train"].append(name)
    for k in ("train", "val"):
        splits[k].sort()

    assert_no_video_leakage({k: v for k, v in splits.items() if k != "test"})
    train_v = {parse_frame_name(n)[0] for n in splits["train"]}
    test_v = {parse_frame_name(n)[0] for n in splits["test"]}
    assert not (train_v & test_v), "train/test video leakage"

    for k, v in splits.items():
        nv = len({parse_frame_name(n)[0] for n in v})
        print(f"  {k:5s}: {len(v):6d} frames / {nv:3d} videos")
    return splits


def pooled_histogram(frames, split_dir="images_thermal_train",
                     cache=Path("configs/train_histogram.npy")):
    if cache.exists():
        print(f"  (reusing cached histogram {cache})")
        return np.load(cache)
    hist = np.zeros(UINT16_BINS, dtype=np.int64)
    t0 = time.time()
    for i, name in enumerate(frames, 1):
        raw = np.array(Image.open(ADAS / split_dir / "analyticsData" / name))
        hist += np.bincount(raw.ravel(), minlength=UINT16_BINS)
        if i % 2500 == 0:
            print(f"    {i}/{len(frames)} frames  {time.time() - t0:.0f}s")
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, hist)
    return hist


def fit_scaler(hist, low_pct=0.01, hot_ceiling_c=100.0):
    total = hist.sum()
    cdf = np.cumsum(hist) / total
    lo = int(np.searchsorted(cdf, low_pct / 100.0))
    hi = int(round((hot_ceiling_c + 273.15) / TLINEAR_K_PER_COUNT))
    nz = np.nonzero(hist)[0]

    below = int(hist[:lo].sum())
    above = int(hist[hi + 1:].sum())
    n_frames = total / (640 * 512)

    print(f"  pooled pixels      : {total:,}")
    print(f"  absolute range     : {nz[0]} .. {nz[-1]} counts "
          f"({raw_to_celsius(nz[0]):.1f} .. {raw_to_celsius(nz[-1]):.1f} C)")
    print(f"  cold floor  p{low_pct} : {lo} counts ({raw_to_celsius(lo):.1f} C), "
          f"clips {below / total * 100:.4f}% ({below / n_frames:.1f} px/frame)")
    print(f"  hot ceiling {hot_ceiling_c:.0f} C : {hi} counts, "
          f"clips {above / total * 100:.6f}% ({above / n_frames:.1f} px/frame)")
    return RadiometricScaler(raw_min=float(lo), raw_max=float(hi)), below, above


def main():
    OUT.mkdir(exist_ok=True)
    print("building video-disjoint splits...")
    splits = build_splits()

    print("\nfitting global scaler on the train split only...")
    hist = pooled_histogram(splits["train"])
    scaler, below, above = fit_scaler(hist)

    print(f"  span               : {scaler.span_counts:.0f} counts "
          f"= {scaler.span_kelvin:.1f} K")

    (OUT / "splits.json").write_text(json.dumps(splits, indent=1))
    (OUT / "scaler.json").write_text(json.dumps({
        "raw_min": scaler.raw_min,
        "raw_max": scaler.raw_max,
        "span_kelvin": scaler.span_kelvin,
        "celsius_range": [raw_to_celsius(scaler.raw_min), raw_to_celsius(scaler.raw_max)],
        "fitted_on": "train split only",
        "cold_floor_percentile": 0.01,
        "hot_ceiling_celsius": 100.0,
        "clipped_below_fraction": below / int(hist.sum()),
        "clipped_above_fraction": above / int(hist.sum()),
    }, indent=1))
    print(f"\nwrote {OUT/'splits.json'} and {OUT/'scaler.json'}")


if __name__ == "__main__":
    main()

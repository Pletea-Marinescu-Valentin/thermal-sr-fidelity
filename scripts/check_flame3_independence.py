"""Is the FLAME 3 CV subset a video, or a set of independent frames?

The answer decides the bootstrap cluster, and with it every p-value on that
split. Clustering 738 frames into the two groups `fire` and `nofire` leaves a
paired test with two units, whose smallest attainable p-value is 0.5, so every
comparison returns p = 1.00 after correction no matter how large the effect.
That is only the right thing to do if the frames really are dependent.

Three measurements, none of which assumes the answer:

  1. pixelwise correlation between numerically adjacent frames. Adjacent video
     frames from a UAV overlap almost completely and correlate near 1.
  2. spread of the consecutive differences of a per-frame summary against the
     spread of the summary itself. Independence predicts a ratio of sqrt(2);
     positive dependence pushes it below.
  3. integrated autocorrelation time of the per-frame metrics, and of the
     paired differences between two models, over the evaluation records.

Run: .venv/Scripts/python.exe scripts/check_flame3_independence.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.flame3 import frames, load_celsius  # noqa: E402

RESULTS = Path("results")
KEYS = ["std_lpips", "m0_rmse_k", "m4_texture_ratio", "m6_flat_delta_hurst",
        "m7_flat_hf_amp_ratio", "m7_flat_hf_corr"]


def acf(x, nlags=60):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return np.full(nlags + 1, np.nan)
    return np.array([1.0 if k == 0 else float((x[:-k] * x[k:]).mean() / var)
                     for k in range(nlags + 1)])


def integrated_time(r):
    """1 + 2 * sum of the autocorrelations, truncated at the first zero."""
    cut = int(np.argmax(r <= 0)) if (r <= 0).any() else len(r)
    return float(1.0 + 2.0 * r[1:cut].sum())


def adjacent_frame_correlation(paths, n_pairs=25):
    out = []
    for i in range(min(n_pairs, len(paths) - 1)):
        a = load_celsius(paths[i]).ravel()
        b = load_celsius(paths[i + 1]).ravel()
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def main():
    paths = frames("cv", "fire")
    if not paths:
        raise FileNotFoundError("no CV frames; run scripts/extract_flame3.py")

    r_adj = adjacent_frame_correlation(paths)
    print("1. pixelwise correlation of numerically adjacent frames")
    print(f"   mean {r_adj.mean():+.3f}, max {r_adj.max():+.3f}, "
          f"over {len(r_adj)} pairs")
    print("   a contiguous video would sit near +1")

    means = np.array([load_celsius(p).mean() for p in paths[:200]])
    ratio = float(np.diff(means).std() / means.std())
    rho1 = 1.0 - ratio ** 2 / 2.0
    print("\n2. spread of consecutive differences in per-frame mean temperature")
    print(f"   sd(frame mean) {means.std():.2f} K, "
          f"sd(consecutive difference) {np.diff(means).std():.2f} K")
    print(f"   ratio {ratio:.3f} against sqrt(2) = {np.sqrt(2):.3f} "
          f"-> implied lag-1 autocorrelation {rho1:+.3f}")

    a_path = RESULTS / "f3_esrgan_classic_per_frame.json"
    b_path = RESULTS / "f3_rrdb_classic_per_frame.json"
    if not (a_path.exists() and b_path.exists()):
        print("\n3. skipped: per-frame results not present")
        return
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    keep = [i for i, n in enumerate(a["names"]) if "/fire/" in n]
    print("\n3. integrated autocorrelation time over the evaluation records")
    print(f"   {'metric':<24}{'raw':>8}{'paired difference':>20}")
    for k in KEYS:
        raw = np.array([a["records"][i].get(k, np.nan) for i in keep], float)
        dif = np.array([a["records"][i].get(k, np.nan)
                        - b["records"][i].get(k, np.nan) for i in keep], float)
        print(f"   {k:<24}{integrated_time(acf(raw)):8.1f}"
              f"{integrated_time(acf(dif)):20.1f}")
    print("\n   one frame means the records carry no usable memory, so the "
          "frame is the sampling unit")


if __name__ == "__main__":
    main()

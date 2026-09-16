"""Recover and validate the XT709 counts-to-temperature scale.

The FLAME 3 NADIR plots ship their thermal frames twice: as raw integer
rasters, and as an orthorectified product already in Celsius. The raw rasters
hold counts around 3000, and the obvious reading of such counts is decikelvin,
T[C] = counts/10 - 273.15. That reading is wrong, but only by 1.85 K, so no
plausibility check on scene temperature can reject it. At the precision this
protocol reports, around 0.5 K of radiometric error, a 1.85 K systematic bias
would be several times the quantity being measured.

The two products of the same frame let the scale be recovered instead of
assumed. Orthorectification warps the image, so pixels cannot be compared
directly, but it is close to value-preserving, so frame statistics can. This
script matches raw and georeferenced frames by index and reports the implied
offset

    counts / 10  -  Celsius

per frame, from the median of each product, and adopts its median. Nodata
margins of the georeferenced product, written as exact zeros, are excluded.

Both products are quantised to 0.1, so the estimate is too. Over the 296
matched pairs the median is 275.0000 and the modal value is 275.0, carried by
40% of frames with the neighbouring bins 274.9 and 275.1 almost symmetric
about it, which is what a quantisation-limited estimator centred on 275.0
should look like.

One consequence is worth stating: this offset cancels in every metric the
protocol reports, because all of them compare a reconstruction against its own
reference and both are produced by the same conversion. Getting it right
matters for the absolute temperatures quoted in the text, not for the
comparisons the conclusions rest on.

Reads the archives directly; nothing needs to be extracted first.
"""

import argparse
import io
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.flame3 import XT709_OFFSET_C  # noqa: E402
from tsrf.data.radiometry import ZERO_CELSIUS_K  # noqa: E402

ARCHIVE = "NADIRPlots.zip"


def index_frames(z):
    raw, geo = {}, {}
    for name in z.namelist():
        m = re.search(r"/(plot \d+)/duringburn/raw_thermal_tiff/(IRX_\d+)\.TIFF$",
                      name, re.I)
        if m:
            raw[m.group(1, 2)] = name
        m = re.search(
            r"/(plot \d+)/duringburn/geo_thermal_tiff_celsius/(IRX_\d+)_ref_geo\.TIFF$",
            name, re.I)
        if m:
            geo[m.group(1, 2)] = name
    return raw, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth matched pair")
    args = ap.parse_args()

    if not Path(args.archive).exists():
        sys.exit(f"{args.archive} not found")

    with zipfile.ZipFile(args.archive) as z:
        raw, geo = index_frames(z)
        pairs = sorted(set(raw) & set(geo))
        print(f"{len(raw)} raw frames, {len(geo)} georeferenced, "
              f"{len(pairs)} matched")

        def load(name):
            with z.open(name) as fh:
                return np.array(Image.open(io.BytesIO(fh.read())), dtype=np.float64)

        # One estimate per frame, from the median of each product. The frame
        # median is the robust choice: upper quantiles are displaced by the
        # orthorectification warp, which resamples a moving fire front, and
        # using them inflates the spread without moving the centre.
        offsets, nodata = [], []
        used = pairs[::args.stride]
        for key in used:
            a, b = load(raw[key]), load(geo[key])
            nodata.append((b == 0).mean())
            b = b[b > 0]
            offsets.append(np.median(a) / 10.0 - np.median(b))

    offsets = np.asarray(offsets)
    med = float(np.median(offsets))
    print(f"frames used: {len(used)}")
    print(f"georeferenced nodata: {np.mean(nodata) * 100:.1f}% of pixels\n")

    # Both products are quantised to 0.1, so the estimator is too; the mode is
    # as informative as the median here and easier to read.
    counts = Counter(np.round(offsets, 1))
    print("implied offset (counts/10 - Celsius)")
    print(f"  median {med:.4f}   mean {offsets.mean():.4f}   std {offsets.std():.4f}")
    print(f"  IQR    [{np.percentile(offsets, 25):.3f}, "
          f"{np.percentile(offsets, 75):.3f}]")
    top = counts.most_common(5)
    print("  mode   " + ", ".join(f"{v:.1f} ({n})" for v, n in top) + "\n")

    print(f"adopted offset : {XT709_OFFSET_C:.2f}   ->  T[C] = counts/10 - "
          f"{XT709_OFFSET_C:.2f}")
    print(f"decikelvin     : {ZERO_CELSIUS_K:.2f}   ->  rejected, differs by "
          f"{XT709_OFFSET_C - ZERO_CELSIUS_K:.2f} K")

    # The estimator is quantised to 0.1 and the warp adds a little spread,
    # so agreement is judged at that resolution rather than exactly.
    ok = abs(med - XT709_OFFSET_C) <= 0.05
    print(f"\n{'VALIDATED' if ok else 'MISMATCH'}: recovered median is "
          f"{med:.4f}, adopted constant is {XT709_OFFSET_C:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

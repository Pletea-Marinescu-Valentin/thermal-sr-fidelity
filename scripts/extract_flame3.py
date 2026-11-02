"""Extract only the radiometric thermal frames from the FLAME 3 archives.

The two archives total 39 GB, of which 37 GB is visual-spectrum imagery this
study never reads: the pipeline is single-channel and works on temperature. The
thermal rasters are 1.8 GB, so we unpack those and leave the rest compressed.

Two decisions here are not cosmetic.

We take `raw_thermal_tiff` from the NADIR plots and ignore
`geo_thermal_tiff_celsius`. The georeferenced product has been orthorectified,
which resamples the image onto a map grid and fills the margins with zeros; 6%
of its pixels are exactly zero. Resampling is itself a smoothing operation, so
a georeferenced raster would report the interpolator's texture rather than the
sensor's, and the texture statistics of Section V are the whole point.

We also keep the two subsets apart, because they are different instruments.
The CV subset was flown with a DJI M30T over Sycan Marsh, Oregon; the NADIR
plots with an Autel Robotics XT709 over Hanna Hammock, Florida. They do not
even store temperature the same way, which `tsrf.data.flame3` handles.
"""

import argparse
import re
import sys
import time
import zipfile
from pathlib import Path

OUT = Path("data/flame3")

# (archive, regex over the archive member, destination subdirectory)
SELECTORS = [
    ("CVSubset.zip",
     re.compile(r"/(Fire|No Fire)/Thermal/Celsius TIFF/(\d+)\.TIFF$", re.I),
     lambda m: f"cv/{m.group(1).lower().replace(' ', '')}"),
    ("NADIRPlots.zip",
     re.compile(r"/plot (\d+)/duringburn/raw_thermal_tiff/(IRX_\d+)\.TIFF$", re.I),
     lambda m: f"nadir/plot{m.group(1)}"),
]


def extract(archive, pattern, dest_for, force=False):
    src = Path(archive)
    if not src.exists():
        print(f"  {archive}: not found, skipping")
        return 0

    written = skipped = 0
    t0 = time.time()
    with zipfile.ZipFile(src) as z:
        members = [(n, m) for n in z.namelist()
                   for m in [pattern.search(n)] if m]
        print(f"  {archive}: {len(members)} thermal frames")
        for name, m in members:
            out = OUT / dest_for(m) / Path(name).name
            if out.exists() and not force:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as fh:
                out.write_bytes(fh.read())
            written += 1
            if written % 200 == 0:
                print(f"    {written} extracted")
    print(f"  {archive}: {written} written, {skipped} already present, "
          f"{time.time() - t0:.0f}s")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-extract frames that are already on disk")
    args = ap.parse_args()

    print("extracting FLAME 3 thermal rasters")
    total = sum(extract(a, p, d, args.force) for a, p, d in SELECTORS)

    print("\non disk:")
    for sub in sorted(p for p in OUT.rglob("*") if p.is_dir()):
        n = len(list(sub.glob("*.TIFF")))
        if n:
            size = sum(f.stat().st_size for f in sub.glob("*.TIFF")) / 2**20
            print(f"  {sub.as_posix():<28} {n:5d} frames  {size:7.1f} MB")
    return 0 if total or any(OUT.rglob("*.TIFF")) else 1


if __name__ == "__main__":
    sys.exit(main())

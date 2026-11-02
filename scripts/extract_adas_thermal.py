import sys
import time
import zipfile
from pathlib import Path

WANTED_PREFIXES = (
    "images_thermal_train/",
    "images_thermal_val/",
    "video_thermal_test/",
)


def main(archive, dest):
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        members = [i for i in zf.infolist()
                   if i.filename.startswith(WANTED_PREFIXES)]
        total = sum(i.file_size for i in members)
        print(f"extracting {len(members)} members, {total / 1e9:.2f} GB uncompressed")

        done = 0
        t0 = time.time()
        for n, info in enumerate(members, 1):
            target = dest / info.filename
            # Skip files already extracted at the right size, so the script is
            # safe to re-run after an interruption.
            if target.exists() and target.stat().st_size == info.file_size:
                done += info.file_size
                continue
            zf.extract(info, dest)
            done += info.file_size
            if n % 2000 == 0 or n == len(members):
                el = time.time() - t0
                print(f"  {n}/{len(members)}  {done / 1e9:.2f} GB  {el:.0f}s")

    for p in WANTED_PREFIXES:
        d = dest / p
        if d.exists():
            tiffs = len(list((d / "analyticsData").glob("*.tiff"))) \
                if (d / "analyticsData").exists() else 0
            jpgs = len(list((d / "data").glob("*.jpg"))) \
                if (d / "data").exists() else 0
            print(f"  {p}: {tiffs} tiff, {jpgs} jpg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/FLIR_ADAS_v2.zip",
         sys.argv[2] if len(sys.argv) > 2 else "data/adas")

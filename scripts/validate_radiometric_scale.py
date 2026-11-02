import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.radiometry import (  # noqa: E402
    TLINEAR_K_PER_COUNT, TLINEAR_K_PER_COUNT_LOWRES, raw_to_celsius,
)

PLAUSIBLE_PERSON_C = (15.0, 40.0)


def thermal_path_for(split_dir, file_name):
    stem = Path(file_name).stem
    return Path(split_dir) / "analyticsData" / f"{stem}.tiff"


def main(split_dir="data/adas/images_thermal_train", n_images=300):
    split_dir = Path(split_dir)
    coco = json.loads((split_dir / "coco.json").read_text())

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    person_ids = {i for i, n in cats.items() if n == "person"}
    if not person_ids:
        raise SystemExit(f"no 'person' category found; categories = {sorted(cats.values())}")

    by_image = {}
    for a in coco["annotations"]:
        if a["category_id"] in person_ids:
            by_image.setdefault(a["image_id"], []).append(a["bbox"])
    images = {im["id"]: im["file_name"] for im in coco["images"]}
    print(f"{len(by_image)} images carry person boxes "
          f"({sum(len(v) for v in by_image.values())} boxes total)")

    person_c, ambient_c, delta_over_ambient = [], [], []
    coldest_c = []
    used = skipped = 0

    for image_id, boxes in sorted(by_image.items()):
        if used >= n_images:
            break
        tif = thermal_path_for(split_dir, images[image_id])
        if not tif.exists():
            skipped += 1
            continue
        raw = np.array(Image.open(tif))
        celsius = raw_to_celsius(raw)
        amb = float(np.median(celsius))
        ambient_c.append(amb)
        coldest_c.append(float(np.percentile(celsius, 0.1)))

        for x, y, w, h in boxes:
            x0, y0 = max(int(x), 0), max(int(y), 0)
            x1, y1 = min(int(x + w), raw.shape[1]), min(int(y + h), raw.shape[0])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = celsius[y0:y1, x0:x1]
            # Upper decile of the box: a box contains background too, and the
            # warm tail is the person.
            t = float(np.percentile(crop, 90))
            person_c.append(t)
            delta_over_ambient.append(t - amb)
        used += 1

    if not person_c:
        raise SystemExit("no usable person crops found -- is the split extracted?")

    person_c = np.array(person_c)
    delta = np.array(delta_over_ambient)
    inside = ((person_c >= PLAUSIBLE_PERSON_C[0]) &
              (person_c <= PLAUSIBLE_PERSON_C[1])).mean()

    print(f"\nimages used: {used}  (skipped, no TIFF: {skipped})")
    print(f"person crops: {len(person_c)}")
    print(f"\n--- hypothesis A: {TLINEAR_K_PER_COUNT} K/count ---")
    print(f"  person p90 temperature : median {np.median(person_c):6.1f} C   "
          f"IQR {np.percentile(person_c, 25):.1f}..{np.percentile(person_c, 75):.1f} C")
    print(f"  frame ambient (median) : median {np.median(ambient_c):6.1f} C")
    print(f"  coldest 0.1% of frame  : median {np.median(coldest_c):6.1f} C")
    print(f"  person minus ambient   : median {np.median(delta):+6.1f} K   "
          f"warmer in {(delta > 0).mean() * 100:.1f}% of crops")
    print(f"  within {PLAUSIBLE_PERSON_C[0]:.0f}-{PLAUSIBLE_PERSON_C[1]:.0f} C: "
          f"{inside * 100:.1f}% of crops")

    ratio = TLINEAR_K_PER_COUNT_LOWRES / TLINEAR_K_PER_COUNT
    alt = (person_c + 273.15) * ratio - 273.15
    print(f"\n--- hypothesis B: {TLINEAR_K_PER_COUNT_LOWRES} K/count ---")
    print(f"  person p90 temperature : median {np.median(alt):6.1f} C  "
          f"<- rejected if implausible")

    verdict = "SUPPORTED" if (inside > 0.9 and (delta > 0).mean() > 0.9) else "QUESTIONABLE"
    print(f"\nVERDICT for {TLINEAR_K_PER_COUNT} K/count: {verdict}")
    return 0 if verdict == "SUPPORTED" else 1


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "data/adas/images_thermal_train"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    sys.exit(main(split, count))

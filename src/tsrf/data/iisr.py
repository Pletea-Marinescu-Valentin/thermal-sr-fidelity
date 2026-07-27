import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/iisr/FLIR-IISR")

#: Intensity levels above the frame median. NOT Kelvin -- see module docstring.
DEFAULT_DELTAS_LEVELS = (40.0, 60.0, 80.0)


def frame_ids(root=ROOT):
    return sorted(int(p.stem) for p in (Path(root) / "HR").glob("*.bmp"))


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64)


def load_pair(idx, root=ROOT, kind="LR_4x"):
    root = Path(root)
    return (load_gray(root / "HR" / f"{idx:06d}.bmp"),
            load_gray(root / kind / f"{idx:06d}.bmp"))


def load_metadata(root=ROOT):
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    degr = json.loads((root / "degradation.json").read_text())
    return {
        "scene_labels": scene["labels"],
        "scene": {int(Path(k).stem): v["scene_label"] for k, v in scene["data"].items()},
        "degradation_labels": degr["labels"],
        "degradation": {int(Path(k).stem): v["degradation_label"]
                        for k, v in degr["data"].items()},
    }


def crop_to_multiple(img, factor=4):
    h, w = img.shape[:2]
    return img[:h - h % factor, :w - w % factor]


def saturation_fraction(img, level=255):
    return float((np.asarray(img) >= level).mean())

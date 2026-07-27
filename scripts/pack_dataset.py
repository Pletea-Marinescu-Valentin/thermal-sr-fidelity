import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.degradation import degrade  # noqa: E402

CACHE = Path("data/cache")
SPLIT_DIRS = {
    "train": "data/adas/images_thermal_train",
    "val": "data/adas/images_thermal_train",
    "test": "data/adas/images_thermal_val",
}


def pack(split, scale=4, preset="classic"):
    frames = json.loads(Path("configs/splits.json").read_text())[split]
    src = Path(SPLIT_DIRS[split]) / "analyticsData"
    CACHE.mkdir(parents=True, exist_ok=True)

    probe = np.array(Image.open(src / frames[0]))
    h, w = probe.shape
    lh, lw = h // scale, w // scale
    print(f"{split}: {len(frames)} frames, HR {h}x{w} -> LR {lh}x{lw}")

    hr_path = CACHE / f"{split}_hr_u16.npy"
    lr_path = CACHE / f"{split}_lr_{preset}_f32.npy"
    hr = np.lib.format.open_memmap(hr_path, mode="w+", dtype=np.uint16,
                                   shape=(len(frames), h, w))
    lr = np.lib.format.open_memmap(lr_path, mode="w+", dtype=np.float32,
                                   shape=(len(frames), lh, lw))

    t0 = time.time()
    for i, name in enumerate(frames):
        raw = np.array(Image.open(src / name))
        hr[i] = raw
        # Noise is added at training time, so pack the noise-free degradation.
        lr[i] = degrade(raw.astype(np.float64), scale=scale,
                        preset=preset, netd_k=0.0).astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(frames)}  {time.time() - t0:.0f}s")

    hr.flush()
    lr.flush()
    (CACHE / f"{split}_index.json").write_text(json.dumps(
        {"frames": frames, "scale": scale, "preset": preset,
         "hr_shape": [h, w], "lr_shape": [lh, lw]}, indent=1))
    print(f"  wrote {hr_path} ({hr.nbytes / 1e9:.2f} GB) "
          f"and {lr_path} ({lr.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "val", "test"]):
        pack(s)

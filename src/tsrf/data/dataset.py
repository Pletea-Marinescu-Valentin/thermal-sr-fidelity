import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .radiometry import RadiometricScaler

CACHE = Path("data/cache")


def load_scaler(path="configs/scaler.json"):
    cfg = json.loads(Path(path).read_text())
    return RadiometricScaler(raw_min=cfg["raw_min"], raw_max=cfg["raw_max"])


class ThermalPatchDataset(Dataset):

    def __init__(self, split, scaler=None, patch_lr=48, scale=4, preset="classic",
                 netd_k=0.0, augment=True, seed=1337, cache=CACHE):
        self.cache = Path(cache)
        meta = json.loads((self.cache / f"{split}_index.json").read_text())
        if meta["scale"] != scale:
            raise ValueError(f"cache built at scale {meta['scale']}, asked for {scale}")

        self.hr = np.load(self.cache / f"{split}_hr_u16.npy", mmap_mode="r")
        self.lr = np.load(self.cache / f"{split}_lr_{preset}_f32.npy", mmap_mode="r")
        self.scaler = scaler or load_scaler()
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
        self.scale = scale
        self.netd_k = netd_k
        self.augment = augment
        self.seed = seed
        self.frames = meta["frames"]

        lh, lw = meta["lr_shape"]
        if patch_lr > min(lh, lw):
            raise ValueError(f"patch_lr {patch_lr} exceeds LR frame {lh}x{lw}")
        self._max_y, self._max_x = lh - patch_lr, lw - patch_lr

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Seeded per (epoch-agnostic) draw: torch's DataLoader reseeds workers, so
        # use torch's generator to stay consistent with its worker seeding.
        y = int(torch.randint(0, self._max_y + 1, (1,)).item())
        x = int(torch.randint(0, self._max_x + 1, (1,)).item())
        p = self.patch_lr

        lr = np.array(self.lr[idx, y:y + p, x:x + p], dtype=np.float32)
        hy, hx, hp = y * self.scale, x * self.scale, self.patch_hr
        hr = np.array(self.hr[idx, hy:hy + hp, hx:hx + hp], dtype=np.float32)

        if self.netd_k > 0:
            # Detector noise is i.i.d., so adding it after cropping is equivalent to
            # adding it to the full frame -- and gives fresh noise every epoch.
            sigma = self.netd_k / 0.04
            lr = lr + torch.randn(lr.shape).numpy() * sigma

        lr = self.scaler.normalize(lr)
        hr = self.scaler.normalize(hr)

        if self.augment:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                lr, hr = np.rot90(lr, k), np.rot90(hr, k)
            if torch.rand(1).item() < 0.5:
                lr, hr = np.fliplr(lr), np.fliplr(hr)

        return (torch.from_numpy(np.ascontiguousarray(lr))[None],
                torch.from_numpy(np.ascontiguousarray(hr))[None])


class ThermalFrameDataset(Dataset):

    def __init__(self, split, scaler=None, scale=4, preset="classic", cache=CACHE):
        self.cache = Path(cache)
        meta = json.loads((self.cache / f"{split}_index.json").read_text())
        self.hr = np.load(self.cache / f"{split}_hr_u16.npy", mmap_mode="r")
        self.lr = np.load(self.cache / f"{split}_lr_{preset}_f32.npy", mmap_mode="r")
        self.scaler = scaler or load_scaler()
        self.frames = meta["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        lr = self.scaler.normalize(np.array(self.lr[idx], dtype=np.float32))
        hr = self.scaler.normalize(np.array(self.hr[idx], dtype=np.float32))
        return torch.from_numpy(lr)[None], torch.from_numpy(hr)[None]

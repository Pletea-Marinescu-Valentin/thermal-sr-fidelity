"""FLAME 3: radiometric thermal UAV imagery, the second instrument pair.

The FLAME 3 release contains two subsets flown with different cameras over
different prescribed burns, and they do not store temperature the same way.
Getting this wrong would silently shift every value reported in Kelvin, so the
two conversions are stated and, for the second one, validated against the
vendor's own product.

CV subset (DJI M30T, Sycan Marsh, Oregon)
    Thermal TIFFs are float32 rasters already in degrees Celsius, so the
    conversion is the definition of the Celsius scale and nothing is inferred.

NADIR plots (Autel Robotics XT709, Hanna Hammock, Florida)
    `raw_thermal_tiff` holds int32 counts around 3000, not Celsius. The release
    also ships `geo_thermal_tiff_celsius`, an orthorectified Celsius product of
    the same frames, which lets the scale be recovered rather than guessed:
    over 296 matched pairs, at quantiles from the first quartile to the
    99.9th percentile, the implied offset

        counts / 10  -  Celsius

    has median exactly 275.00 (spread 0.82, entirely attributable to the
    orthorectification warp changing which pixels are sampled). Hence

        T[C] = counts / 10 - 275.0,      T[K] = counts / 10 - 1.85.

    We read the raw rasters rather than the georeferenced ones. Orthorectifi-
    cation resamples onto a map grid and pads the margins with zeros, about 6%
    of the frame; resampling is a smoothing operation, and a study whose
    subject is texture fidelity cannot take an interpolated raster as its
    measurement.

Normalisation bounds are declared from the instruments rather than fitted to a
split, which keeps the test subset out of the normalisation entirely. The
release documents saturation near 500 C, measured maxima reach 609 C, and the
coldest first-percentile reading observed is -22.8 C.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .radiometry import ZERO_CELSIUS_K

ROOT = Path("data/flame3")

#: Counts-to-Celsius for the Autel XT709 raw rasters, recovered from the
#: vendor's Celsius product (see module docstring).
XT709_COUNTS_PER_DEGREE = 10.0
XT709_OFFSET_C = 275.0

#: Declared normalisation range, in Celsius, common to both instruments.
SCALE_MIN_C = -25.0
SCALE_MAX_C = 610.0

SUBSETS = {
    "cv": ("fire", "nofire"),
    "nadir": ("plot1", "plot2"),
}


def is_nadir(path):
    return "nadir" in Path(path).as_posix().lower().split("/")


def bootstrap_cluster(path):
    """The unit of independent sampling a frame belongs to.

    The two subsets are built differently and must not be clustered the same
    way. The NADIR plots are continuous flight video: neighbouring frames are
    overlapping views of one burn, so the plot is the unit and frames inside it
    are not independent observations.

    The CV subset is not video. It is a curated set of frame pairs sampled
    sparsely across the Sycan Marsh burn, and its frames behave as near
    independent draws: numerically adjacent frames correlate pixelwise at 0.19
    on average rather than near unity, the spread of their consecutive mean
    temperatures, 12.80 K, sits just under the sqrt(2) x 9.31 K independence
    predicts, implying a lag-1 autocorrelation of 0.06, and both the raw
    per-frame metrics and the paired differences between models carry
    integrated autocorrelation times of about one frame
    (`scripts/check_flame3_independence.py`).

    Clustering that subset on its fire / no-fire partition, as an earlier
    version of this code did, leaves two clusters. A paired test over two
    units cannot return a p-value below 0.5 whatever the data say, so every
    comparison came back at p = 1.00 after correction, including differences
    that are in fact large, consistent and visible by eye. That is not a
    conservative choice, it is a wrong one: the partition is a property of the
    scene content, not of how the frames were sampled.
    """
    p = Path(path)
    if is_nadir(p):
        return p.parent.name
    # Qualified by group: the two CV groups number their frames from 00001
    # independently, so the stem alone would silently merge 116 unrelated
    # pairs of frames into shared clusters.
    return f"{p.parent.name}/{p.stem}"


def counts_to_celsius_xt709(counts):
    a = np.asarray(counts, dtype=np.float64)
    return a / XT709_COUNTS_PER_DEGREE - XT709_OFFSET_C


def load_celsius(path):
    """Read one FLAME 3 thermal raster as degrees Celsius."""
    a = np.array(Image.open(path))
    if is_nadir(path):
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"expected integer counts for a NADIR frame: {path}")
        return counts_to_celsius_xt709(a)
    if a.dtype != np.float32:
        raise ValueError(f"expected float32 Celsius for a CV frame: {path}")
    return a.astype(np.float64)


def load_kelvin(path):
    return load_celsius(path) + ZERO_CELSIUS_K


def frames(subset, group=None, root=ROOT):
    """Sorted frame paths for a subset, optionally restricted to one group."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {list(SUBSETS)}")
    groups = (group,) if group else SUBSETS[subset]
    out = []
    for g in groups:
        out.extend(sorted((Path(root) / subset / g).glob("*.TIFF")))
    return out


class Flame3Scaler:
    """Affine map from Celsius to [0, 1] over a declared physical range.

    Mirrors `RadiometricScaler` but is expressed in Celsius, because FLAME 3
    stores temperature rather than sensor counts and there is no gain to apply.
    """

    def __init__(self, min_c=SCALE_MIN_C, max_c=SCALE_MAX_C):
        if not max_c > min_c:
            raise ValueError(f"max_c must exceed min_c, got {min_c}, {max_c}")
        self.min_c = float(min_c)
        self.max_c = float(max_c)

    @property
    def span_kelvin(self):
        return self.max_c - self.min_c

    def normalize(self, celsius, clip=True):
        x = (np.asarray(celsius, dtype=np.float32) - self.min_c) / self.span_kelvin
        return np.clip(x, 0.0, 1.0) if clip else x

    def denormalize(self, x):
        return np.asarray(x, dtype=np.float32) * self.span_kelvin + self.min_c

    def to_kelvin(self, x):
        return self.denormalize(x).astype(np.float64) + ZERO_CELSIUS_K

    def to_celsius(self, x):
        return self.denormalize(x).astype(np.float64)

    def clipping_fraction(self, celsius):
        a = np.asarray(celsius)
        return float(((a < self.min_c) | (a > self.max_c)).mean())

    def __repr__(self):
        return f"Flame3Scaler(min_c={self.min_c}, max_c={self.max_c})"


# --------------------------------------------------------------------- torch

class _Flame3Cached:
    """Common loading for the packed FLAME 3 splits.

    The ADAS datasets read uint16 counts; these read float32 Celsius, so the
    detector-noise amplitude is applied directly in Kelvin instead of being
    divided by a counts-per-Kelvin gain.
    """

    def __init__(self, split, scaler=None, scale=4, preset="classic",
                 cache=Path("data/cache")):
        import json
        self.cache = Path(cache)
        meta = json.loads((self.cache / f"{split}_index.json").read_text())
        if meta["scale"] != scale:
            raise ValueError(
                f"cache built at scale {meta['scale']}, asked for {scale}")
        if meta.get("units") != "celsius":
            raise ValueError(f"{split}: expected a Celsius cache")

        self.hr = np.load(self.cache / f"{split}_hr_c_f32.npy", mmap_mode="r")
        self.lr = np.load(self.cache / f"{split}_lr_{preset}_f32.npy", mmap_mode="r")
        self.scaler = scaler or Flame3Scaler()
        self.frames = meta["frames"]
        self.meta = meta

    def __len__(self):
        return len(self.frames)


class Flame3PatchDataset(_Flame3Cached):
    """Random LR/HR patch pairs for training."""

    def __init__(self, split, scaler=None, patch_lr=48, scale=4,
                 preset="classic", netd_k=0.0, augment=True,
                 cache=Path("data/cache")):
        super().__init__(split, scaler, scale, preset, cache)
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
        self.scale = scale
        self.netd_k = netd_k
        self.augment = augment

        lh, lw = self.meta["lr_shape"]
        if patch_lr > min(lh, lw):
            raise ValueError(f"patch_lr {patch_lr} exceeds LR frame {lh}x{lw}")
        self._max_y, self._max_x = lh - patch_lr, lw - patch_lr

    def __getitem__(self, idx):
        import torch
        y = int(torch.randint(0, self._max_y + 1, (1,)).item())
        x = int(torch.randint(0, self._max_x + 1, (1,)).item())
        p = self.patch_lr

        lr = np.array(self.lr[idx, y:y + p, x:x + p], dtype=np.float32)
        hy, hx, hp = y * self.scale, x * self.scale, self.patch_hr
        hr = np.array(self.hr[idx, hy:hy + hp, hx:hx + hp], dtype=np.float32)

        if self.netd_k > 0:
            # The cache is already in Kelvin-sized units, so NETD applies as is.
            lr = lr + torch.randn(lr.shape).numpy() * self.netd_k

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


class Flame3FrameDataset(_Flame3Cached):
    """Whole normalised frames, for validation."""

    def __getitem__(self, idx):
        import torch
        lr = self.scaler.normalize(np.array(self.lr[idx], dtype=np.float32))
        hr = self.scaler.normalize(np.array(self.hr[idx], dtype=np.float32))
        return torch.from_numpy(lr)[None], torch.from_numpy(hr)[None]

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

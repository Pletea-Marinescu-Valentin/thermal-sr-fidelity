from dataclasses import dataclass

import numpy as np

#: TLinear high-resolution gain, Kelvin per count. See module docstring.
TLINEAR_K_PER_COUNT = 0.04

#: TLinear low-resolution gain, rejected for this dataset (gives ~485 C medians).
TLINEAR_K_PER_COUNT_LOWRES = 0.1

ZERO_CELSIUS_K = 273.15


def raw_to_kelvin(raw, k_per_count=TLINEAR_K_PER_COUNT):
    return np.asarray(raw, dtype=np.float64) * k_per_count


def raw_to_celsius(raw, k_per_count=TLINEAR_K_PER_COUNT):
    return raw_to_kelvin(raw, k_per_count) - ZERO_CELSIUS_K


def kelvin_to_raw(kelvin, k_per_count=TLINEAR_K_PER_COUNT):
    return np.asarray(kelvin, dtype=np.float64) / k_per_count


@dataclass(frozen=True)
class RadiometricScaler:

    raw_min: float
    raw_max: float

    def __post_init__(self):
        if not self.raw_max > self.raw_min:
            raise ValueError(
                f"raw_max must exceed raw_min, got {self.raw_min}, {self.raw_max}")

    @property
    def span_counts(self) -> float:
        return self.raw_max - self.raw_min

    @property
    def span_kelvin(self) -> float:
        return self.span_counts * TLINEAR_K_PER_COUNT

    def normalize(self, raw, clip=True):
        x = (np.asarray(raw, dtype=np.float32) - self.raw_min) / self.span_counts
        return np.clip(x, 0.0, 1.0) if clip else x

    def denormalize(self, x):
        return np.asarray(x, dtype=np.float32) * self.span_counts + self.raw_min

    def to_kelvin(self, x):
        return raw_to_kelvin(self.denormalize(x))

    def to_celsius(self, x):
        return self.to_kelvin(x) - ZERO_CELSIUS_K

    def clipping_fraction(self, raw):
        raw = np.asarray(raw)
        return float(((raw < self.raw_min) | (raw > self.raw_max)).mean())

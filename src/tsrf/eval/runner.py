import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.degradation import degrade
from ..data.radiometry import RadiometricScaler, raw_to_kelvin
from ..data.splits import parse_frame_name
from ..metrics import (
    cold_region_smoothness, gradient_fidelity, hallucination_metrics,
    hotspot_preservation, radiometric_error, texture_correspondence,
    texture_scaling, thermal_ordering,
)
from ..metrics.perceptual import perceptual_metrics


def load_scaler(path="configs/scaler.json"):
    cfg = json.loads(Path(path).read_text())
    return RadiometricScaler(raw_min=cfg["raw_min"], raw_max=cfg["raw_max"])


def load_splits(path="configs/splits.json"):
    return json.loads(Path(path).read_text())


def bicubic_upsampler(scale=4):
    from ..data.degradation import imresize_bicubic
    return lambda lr: imresize_bicubic(lr, scale)


def evaluate_frame(hr_counts, model, scaler, scale=4, preset="classic",
                   deltas_k=(5.0, 10.0, 20.0), rng=None, to_kelvin=None,
                   k_per_count=None):
    """Score one reconstruction against its reference.

    `to_kelvin` converts whatever the dataset stores into Kelvin: sensor counts
    on ADAS, degrees Celsius on FLAME 3. Every metric below reads temperature,
    so this is the only place the two datasets differ.
    """
    to_kelvin = to_kelvin or raw_to_kelvin
    extra = {} if k_per_count is None else {"k_per_count": k_per_count}
    lr = degrade(hr_counts, scale=scale, preset=preset, rng=rng, **extra)
    sr = np.asarray(model(lr), dtype=np.float64)
    if sr.shape != hr_counts.shape:
        raise ValueError(f"model returned {sr.shape}, expected {hr_counts.shape}")

    t_gt = to_kelvin(hr_counts)
    t_sr = to_kelvin(sr)
    # The LR observed at HR scale, used as the plausibility reference for M2.
    from ..data.degradation import imresize_bicubic
    t_lr_up = to_kelvin(imresize_bicubic(lr, scale))

    rec = {}
    rec.update({f"m0_{k}": v for k, v in radiometric_error(t_sr, t_gt).items()})
    rec.update({f"std_{k}": v for k, v in perceptual_metrics(
        scaler.normalize(sr), scaler.normalize(hr_counts),
        include_lpips=True).items()})
    rec.update({f"m3_{k}": v for k, v in thermal_ordering(t_sr, t_gt).items()})
    rec.update({f"m4_{k}": v for k, v in cold_region_smoothness(t_sr, t_gt).items()})
    rec.update({f"m5_{k}": v for k, v in gradient_fidelity(t_sr, t_gt).items()})
    # M6 shares M4's flat cold set, so the pair reads amplitude and scaling of
    # the same pixels. The per-scale ladders are dropped: they are diagnostic
    # curves, not scalars to aggregate, and they would bloat every record.
    rec.update({f"m6_{k}": v for k, v in texture_scaling(t_sr, t_gt).items()
                if not k.endswith("_ladder_gt_k") and not k.endswith("_ladder_sr_k")})
    rec.update({f"m7_{k}": v for k, v in texture_correspondence(t_sr, t_gt).items()})

    for d in deltas_k:
        tag = f"d{int(d)}"
        for k, v in hotspot_preservation(t_sr, t_gt, delta_k=d).items():
            rec[f"m1_{tag}_{k}"] = v
        for k, v in hallucination_metrics(t_sr, t_gt, t_lr_up, delta_k=d).items():
            rec[f"m2_{tag}_{k}"] = v
    return rec


def evaluate_split(model, split="test", split_dir="data/adas/images_thermal_val",
                   scale=4, preset="classic", limit=None, seed=1337,
                   scaler=None, splits=None, progress_every=100):
    scaler = scaler or load_scaler()
    splits = splits or load_splits()
    frames = splits[split][:limit] if limit else splits[split]

    records, names, videos = [], [], []
    for i, name in enumerate(frames, 1):
        raw = np.array(Image.open(Path(split_dir) / "analyticsData" / name),
                       dtype=np.float64)
        # Per-frame seed: reproducible yet decorrelated across frames. blake2b, not
        # hash(), because Python randomises string hashing per process -- the noise
        # would differ between runs and silently break reproducibility.
        digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=4).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        records.append(evaluate_frame(raw, model, scaler, scale, preset, rng=rng))
        names.append(name)
        videos.append(parse_frame_name(name)[0])
        if progress_every and i % progress_every == 0:
            print(f"    {i}/{len(frames)}")
    return records, names, videos


def aggregate(records, videos, keys=None, n_boot=2000):
    from ..stats.bootstrap import cluster_bootstrap_ci
    keys = keys or sorted(records[0])
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in records]
        out[k] = cluster_bootstrap_ci(vals, videos, n_boot=n_boot)
    return out


def evaluate_split_flame3(model, split="f3test", scale=4, preset="classic",
                          limit=None, seed=1337, scaler=None,
                          progress_every=100):
    """Run the protocol over a FLAME 3 split.

    Differs from `evaluate_split` in three ways, all of them dataset
    properties rather than choices: frames are read from disk as Celsius
    rather than counts, the conversion to Kelvin is the Celsius definition,
    and the bootstrap cluster follows how each subset was sampled rather than
    what it contains.
    """
    from ..data.flame3 import Flame3Scaler, bootstrap_cluster
    from ..data.flame3 import frames as flame3_frames
    from ..data.flame3 import load_celsius
    from ..data.radiometry import ZERO_CELSIUS_K

    subset, group = {"f3train": ("nadir", "plot2"),
                     "f3val": ("nadir", "plot1"),
                     "f3test": ("cv", None)}[split]
    scaler = scaler or Flame3Scaler()
    paths = flame3_frames(subset, group)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(
            f"{split}: no frames; run scripts/extract_flame3.py first")

    def to_kelvin(celsius):
        return np.asarray(celsius, dtype=np.float64) + ZERO_CELSIUS_K

    records, names, clusters = [], [], []
    for i, path in enumerate(paths, 1):
        c = load_celsius(path)
        digest = hashlib.blake2b(f"{seed}:{path.name}".encode(),
                                 digest_size=4).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        # The rasters are Celsius, so the sensor NETD applies as a temperature
        # with no counts-per-Kelvin conversion.
        records.append(evaluate_frame(c, model, scaler, scale, preset,
                                      rng=rng, to_kelvin=to_kelvin,
                                      k_per_count=1.0))
        names.append(path.as_posix())
        # The sampling unit differs between the subsets: a flight plot for the
        # NADIR video, an individual frame for the sparsely sampled CV pairs.
        # See `flame3.bootstrap_cluster` for the measurements behind that.
        clusters.append(bootstrap_cluster(path))
        if progress_every and i % progress_every == 0:
            print(f"    {i}/{len(paths)}")
    return records, names, clusters

import os
import sys
import collections

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import open_remote_zip  # noqa: E402

# FLIR ADAS v2 must be obtained from Teledyne FLIR after accepting its image
# licence agreement; this project does not redistribute it or its access URL.
# Point ADAS_URL at your own copy, or set the TSRF_ADAS_URL environment variable.
ADAS_URL = os.environ.get("TSRF_ADAS_URL", "")
IISR_URL = os.environ.get(
    "TSRF_IISR_URL",
    "https://huggingface.co/datasets/yuanzsz/FLIR-IISR/resolve/main/FLIR-IISR.zip")

# TLinear high-resolution gain, inferred rather than read from metadata; validated
# by scripts/validate_radiometric_scale.py.
TLINEAR_K_PER_COUNT = 0.04


def to_celsius(raw):
    return raw.astype(np.float64) * TLINEAR_K_PER_COUNT - 273.15


def zncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def verify_adas(workdir, n=3):
    if not ADAS_URL:
        raise SystemExit(
            "Set TSRF_ADAS_URL to your own copy of the FLIR ADAS v2 archive.\n"
            "Obtain it from Teledyne FLIR under its image licence agreement.")
    os.makedirs(workdir, exist_ok=True)
    hf, zf = open_remote_zip(ADAS_URL)
    print(f"archive: {hf.size / 1e9:.2f} GB")

    names = zf.namelist()
    ext = collections.Counter(
        n_.rsplit(".", 1)[-1].lower() if "." in n_ else "<dir>" for n_ in names
    )
    print("extensions:", dict(ext.most_common(8)))

    tif = sorted(
        p for p in names
        if p.startswith("images_thermal_train/analyticsData/") and p.endswith(".tiff")
    )
    jpg = sorted(
        p for p in names
        if p.startswith("images_thermal_train/data/") and p.endswith(".jpg")
    )
    print(f"thermal TIFF: {len(tif)}   paired AGC JPEG: {len(jpg)}")
    assert len(tif) == len(jpg), "TIFF/JPEG counts must match 1:1"

    for tp, jp in list(zip(tif, jpg))[:n]:
        assert os.path.splitext(os.path.basename(tp))[0] == \
               os.path.splitext(os.path.basename(jp))[0], "filename pairing broken"
        tl = os.path.join(workdir, os.path.basename(tp))
        jl = os.path.join(workdir, os.path.basename(jp))
        if not os.path.exists(tl):
            open(tl, "wb").write(zf.read(tp))
        if not os.path.exists(jl):
            open(jl, "wb").write(zf.read(jp))

        im = Image.open(tl)
        raw = np.array(im)
        agc = cv2.imread(jl, cv2.IMREAD_UNCHANGED).astype(np.float64)
        C = to_celsius(raw)

        print(f"\n--- {os.path.basename(tp)[:44]}")
        print(f"  TIFF mode={im.mode} dtype={raw.dtype} shape={raw.shape} "
              f"bits={im.tag_v2.get(258)} samples={im.tag_v2.get(277)}")
        print(f"  raw  : {raw.min()}..{raw.max()}  mean={raw.mean():.1f}  "
              f"unique={len(np.unique(raw))}")
        print(f"  degC : median={np.median(C):.1f}  p1={np.percentile(C, 1):.1f}  "
              f"p99.9={np.percentile(C, 99.9):.1f}  max={C.max():.1f}")
        print(f"  mean raw -> {raw.mean() * TLINEAR_K_PER_COUNT:.2f} K "
              f"(sanity: outdoor scene mean should sit near 300 K)")

        # AGC information loss
        spread = [np.ptp(C[agc == lv]) for lv in range(1, 255) if (agc == lv).sum() > 50]
        sat = agc >= 254
        print(f"  AGC collapses median {np.median(spread):.1f} K per level "
              f"(max {np.max(spread):.1f} K)")
        if sat.any():
            print(f"  AGC>=254: {sat.mean() * 100:.3f}% px spanning "
                  f"{C[sat].min():.1f}..{C[sat].max():.1f} C "
                  f"({np.ptp(C[sat]):.1f} K destroyed)")


def verify_iisr(workdir, n=3):
    os.makedirs(workdir, exist_ok=True)
    hf, zf = open_remote_zip(IISR_URL)
    print(f"archive: {hf.size / 1e6:.1f} MB")

    infos = [i for i in zf.infolist() if i.file_size]
    by = collections.defaultdict(list)
    for i in infos:
        parts = i.filename.split("/")
        if len(parts) >= 3:
            by[parts[1]].append(i.file_size)
    for k, v in by.items():
        print(f"  {k}: n={len(v)} sizes={set(v)}")

    for idx in range(1, n + 1):
        loaded = {}
        for sub in ("HR", "LR", "LR_4x"):
            member = f"FLIR-IISR/{sub}/{idx:06d}.bmp"
            lp = os.path.join(workdir, f"{sub}_{idx:06d}.bmp")
            if not os.path.exists(lp):
                open(lp, "wb").write(zf.read(member))
            loaded[sub] = cv2.imread(lp, cv2.IMREAD_UNCHANGED)

        HR3 = loaded["HR"]
        identical = np.array_equal(HR3[:, :, 0], HR3[:, :, 1]) and \
                    np.array_equal(HR3[:, :, 1], HR3[:, :, 2])
        HR = HR3[:, :, 0].astype(np.float32)
        LR = loaded["LR"][:, :, 0].astype(np.float32)
        L4 = loaded["LR_4x"][:, :, 0].astype(np.float32)

        print(f"\n--- pair {idx:06d}")
        print(f"  HR dtype={HR3.dtype} shape={HR3.shape} channels_identical={identical}")
        shift, resp = cv2.phaseCorrelate(HR.copy(), LR.copy())
        print(f"  HR<->LR shift = ({shift[0]:+.2f}, {shift[1]:+.2f}) px  "
              f"resp={resp:.3f}   ZNCC={zncc(HR, LR):.4f}")

        h, w = L4.shape
        for nm, itp in [("area", cv2.INTER_AREA), ("bicubic", cv2.INTER_CUBIC),
                        ("bilinear", cv2.INTER_LINEAR)]:
            cand = cv2.resize(LR, (w, h), interpolation=itp)
            print(f"    LR down[{nm:8s}] vs LR_4x: MAE={np.abs(cand - L4).mean():6.3f}")
        cand = cv2.resize(HR, (w, h), interpolation=cv2.INTER_CUBIC)
        print(f"    HR down[bicubic ] vs LR_4x: MAE={np.abs(cand - L4).mean():6.3f}"
              f"   <- large => LR carries REAL degradation")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "adas"
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/_verify_{which}"
    (verify_adas if which == "adas" else verify_iisr)(out)

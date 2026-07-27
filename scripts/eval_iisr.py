import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.degradation import imresize_bicubic  # noqa: E402
from tsrf.data.iisr import (  # noqa: E402
    DEFAULT_DELTAS_LEVELS, ROOT, crop_to_multiple, frame_ids, load_metadata,
    load_pair, saturation_fraction,
)
from tsrf.metrics import (  # noqa: E402
    cold_region_smoothness, gradient_fidelity, hallucination_metrics,
    hotspot_preservation, radiometric_error, thermal_ordering,
)
from tsrf.metrics.perceptual import perceptual_metrics  # noqa: E402
from tsrf.stats import cluster_bootstrap_ci, paired_test  # noqa: E402

SCALE = 4
HEADLINE = [
    ("m0_rmse_k", "error RMSE", "levels"),
    ("std_psnr_db", "PSNR", "dB"),
    ("std_ssim", "SSIM", ""),
    ("m1_d60_mask_iou", "M1 hot mask IoU (+60)", ""),
    ("m1_d60_abs_count_error", "M1 abs count error (+60)", ""),
    ("m2_d60_n_hallucinated", "M2 hallucinated regions (+60)", "/frame"),
    ("m2_d60_n_missed", "M2 missed regions (+60)", "/frame"),
    ("m2_d60_hallucination_rate", "M2 hallucination rate (+60)", ""),
    ("m3_spearman_rho", "M3 ordering rho", ""),
    ("m4_texture_ratio", "M4 flat-region texture ratio", ""),
    ("m5_gradient_ratio", "M5 gradient ratio", ""),
]


def score(sr, hr, lr_up, deltas):
    rec = {}
    rec.update({f"m0_{k}": v for k, v in radiometric_error(sr, hr).items()})
    rec.update({f"std_{k}": v for k, v in
                perceptual_metrics(sr / 255.0, hr / 255.0).items()})
    rec.update({f"m3_{k}": v for k, v in
                thermal_ordering(sr, hr, delta_k=deltas[0]).items()})
    rec.update({f"m4_{k}": v for k, v in cold_region_smoothness(sr, hr).items()})
    rec.update({f"m5_{k}": v for k, v in gradient_fidelity(sr, hr).items()})
    for d in deltas:
        tag = f"d{int(d)}"
        for k, v in hotspot_preservation(sr, hr, delta_k=d).items():
            rec[f"m1_{tag}_{k}"] = v
        for k, v in hallucination_metrics(sr, hr, lr_up, delta_k=d).items():
            rec[f"m2_{tag}_{k}"] = v
    return rec


def main(limit=None, root=ROOT, deltas=DEFAULT_DELTAS_LEVELS, model=None,
         tag="bicubic"):
    ids = frame_ids(root)[:limit] if limit else frame_ids(root)
    meta = load_metadata(root)
    print(f"FLIR-IISR: {len(ids)} frames, model={tag}, "
          f"deltas {deltas} intensity levels\n")

    recs = {"synthetic": [], "real": []}
    sat = []
    t0 = time.time()
    for n, i in enumerate(ids, 1):
        hr, lr_real = load_pair(i, root, kind="LR_4x")
        hr = crop_to_multiple(hr, SCALE)
        lr_real = lr_real[:hr.shape[0] // SCALE, :hr.shape[1] // SCALE]
        sat.append(saturation_fraction(hr))

        lr_syn = imresize_bicubic(hr, 1.0 / SCALE)
        for arm, lr in (("synthetic", lr_syn), ("real", lr_real)):
            up = imresize_bicubic(lr, SCALE)   # plausibility reference for M2
            sr = up if model is None else np.asarray(model(lr), dtype=np.float64)
            recs[arm].append(score(sr, hr, up, deltas))
        if n % 200 == 0:
            print(f"  {n}/{len(ids)}  {time.time() - t0:.0f}s")

    print(f"\nscored {len(ids)} frames x 2 arms in {time.time() - t0:.0f}s")
    print(f"HR saturation at 255: mean {np.mean(sat) * 100:.2f}% of pixels "
          f"-- peak metrics are clipped from above\n")

    # Each frame is its own scene here, so bootstrap over frames.
    clusters = np.arange(len(ids))
    agg = {arm: {k: cluster_bootstrap_ci([r.get(k, np.nan) for r in recs[arm]],
                                         clusters, n_boot=2000)
                 for k, _, _ in HEADLINE} for arm in recs}

    print(f"{'metric':<34} {'synthetic':>20} {'real':>20}")
    print("-" * 78)
    for key, label, unit in HEADLINE:
        s, r = agg["synthetic"][key], agg["real"][key]
        print(f"{label:<34} {s['mean']:>10.4f} [{s['lo']:>6.2f},{s['hi']:>6.2f}] "
              f"{r['mean']:>10.4f} [{r['lo']:>6.2f},{r['hi']:>6.2f}]")

    print(f"\npaired Wilcoxon, real vs synthetic (n={len(ids)} frames):")
    tests = {}
    for key, label, _ in HEADLINE:
        t = paired_test([x.get(key) for x in recs["real"]],
                        [x.get(key) for x in recs["synthetic"]])
        tests[key] = t
        print(f"  {label:<34} median diff {t.get('median_diff', float('nan')):>9.4f}  "
              f"p={t['p_value']:.3g}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"iisr_{tag}.json").write_text(json.dumps(
        {"n_frames": len(ids), "deltas_levels": list(deltas),
         "hr_saturation_mean": float(np.mean(sat)),
         "aggregate": agg, "paired_tests": tests,
         "degradation_counts": {
             str(k): int(sum(1 for i in ids if meta["degradation"].get(i) == k))
             for k in (0, 1)}}, indent=1))
    print(f"\nwrote results/iisr_{tag}.json")


def build_gan_adapter():
    import torch
    from tsrf.models import build_model
    from tsrf.models.registry import MODEL_SPECS
    from tsrf.eval import load_checkpoint

    spec = MODEL_SPECS["esrgan"]
    net = build_model(spec["builder"], scale=SCALE, **spec["kwargs"])
    net, _, _ = load_checkpoint(net, Path(spec["run"]) / spec["checkpoint"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def run(lr_intensity):
        x = np.clip(np.asarray(lr_intensity, dtype=np.float32) / 255.0, 0, 1)
        t = torch.from_numpy(x)[None, None].to(device)
        with torch.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            y = net(t)
        return np.clip(y.float()[0, 0].cpu().numpy() * 255.0, 0, 255)

    return run


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="bicubic", choices=["bicubic", "esrgan"])
    args = ap.parse_args()
    model = build_gan_adapter() if args.model == "esrgan" else None
    main(limit=args.limit, model=model, tag=args.model)

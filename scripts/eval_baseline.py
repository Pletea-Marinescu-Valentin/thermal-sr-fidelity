import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.eval import (  # noqa: E402
    aggregate, bicubic_upsampler, evaluate_split, load_scaler,
)

HEADLINE = [
    ("m0_rmse_k", "M0 radiometric RMSE", "K"),
    ("m0_max_abs_error_k", "M0 max abs error", "K"),
    ("std_psnr_db", "PSNR", "dB"),
    ("std_ssim", "SSIM", ""),
    ("m1_d10_mask_iou", "M1 hotspot mask IoU (10 K)", ""),
    ("m1_d10_abs_count_error", "M1 abs count error (10 K)", ""),
    ("m1_d10_peak_error_on_gt_support_k", "M1 peak error on GT support (10 K)", "K"),
    ("m1_d10_peak_position_error_px", "M1 peak position error (10 K)", "px"),
    ("m2_d10_n_hallucinated", "M2 hallucinated regions (10 K)", "/frame"),
    ("m2_d10_n_missed", "M2 missed regions (10 K)", "/frame"),
    ("m2_d10_miss_rate", "M2 miss rate (10 K)", ""),
    ("m3_spearman_rho", "M3 thermal ordering rho", ""),
    ("m3_top1_preserved", "M3 top-1 hotspot preserved", "rate"),
    ("m4_texture_ratio", "M4 cold-region texture ratio", ""),
    ("m5_gradient_ratio", "M5 gradient magnitude ratio", ""),
    ("m5_orientation_cosine", "M5 gradient orientation cos", ""),
]


def main(limit=None, preset="classic"):
    scaler = load_scaler()
    print(f"scaler: {scaler.raw_min:.0f}..{scaler.raw_max:.0f} counts "
          f"({scaler.span_kelvin:.1f} K span)")
    print(f"degradation preset: {preset}   scale: x4\n")

    t0 = time.time()
    records, names, videos = evaluate_split(
        bicubic_upsampler(4), split="test", preset=preset, limit=limit, scaler=scaler)
    print(f"\nscored {len(records)} frames from {len(set(videos))} videos "
          f"in {time.time() - t0:.0f}s")

    agg = aggregate(records, videos)
    print(f"\n{'metric':<42} {'mean':>10}  {'95% CI':>22}")
    print("-" * 78)
    for key, label, unit in HEADLINE:
        if key not in agg:
            continue
        a = agg[key]
        u = f" {unit}" if unit else ""
        print(f"{label:<42} {a['mean']:>10.4f}  "
              f"[{a['lo']:>9.4f}, {a['hi']:>9.4f}]{u}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    tag = f"bicubic_{preset}" + (f"_n{limit}" if limit else "")
    (out / f"{tag}.json").write_text(json.dumps(
        {"model": "bicubic", "preset": preset, "n_frames": len(records),
         "n_videos": len(set(videos)), "aggregate": agg}, indent=1))
    (out / f"{tag}_per_frame.json").write_text(json.dumps(
        {"names": names, "videos": videos, "records": records}, indent=1))
    print(f"\nwrote results/{tag}.json")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    pre = sys.argv[2] if len(sys.argv) > 2 else "classic"
    main(lim, pre)

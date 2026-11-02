import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.eval import (  # noqa: E402
    aggregate, bicubic_upsampler, evaluate_split, load_checkpoint, load_scaler,
    torch_upsampler,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import DISPLAY_ORDER, MODEL_SPECS, available_models  # noqa: E402
from tsrf.stats import holm_bonferroni, paired_test  # noqa: E402

RESULTS = Path("results")
HEADLINE = [
    ("std_psnr_db", "PSNR (dB)", "high"),
    ("std_ssim", "SSIM", "high"),
    ("std_lpips", "LPIPS", "low"),
    ("m0_rmse_k", "M0 RMSE (K)", "low"),
    ("m0_max_abs_error_k", "M0 max err (K)", "low"),
    ("m1_d10_mask_iou", "M1 IoU@10K", "high"),
    ("m1_d10_peak_error_on_gt_support_k", "M1 peak err (K)", "low"),
    ("m2_d10_n_hallucinated", "M2 halluc/frame", "low"),
    ("m2_d10_miss_rate", "M2 miss rate", "low"),
    ("m3_spearman_rho", "M3 ordering rho", "high"),
    ("m3_top1_preserved", "M3 top-1 kept", "high"),
    ("m4_texture_ratio", "M4 texture ratio", "one"),
    ("m5_gradient_ratio", "M5 gradient ratio", "one"),
    ("m6_flat_delta_hurst", "M6 delta Hurst", "zero"),
    ("m6_flat_dim_sr", "M6 box dim", "gt"),
    ("m7_flat_hf_corr", "M7 texture corr", "high"),
    ("m7_flat_hf_amp_ratio", "M7 texture amp", "one"),
]


def model_callable(name, scaler):
    spec = MODEL_SPECS[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    ckpt = Path(spec["run"]) / spec["checkpoint"]
    net, step, best = load_checkpoint(net, ckpt)
    print(f"  {name}: loaded {ckpt} (step {step})")
    return torch_upsampler(net, scaler)


def eval_one(name, scaler, preset, limit, force):
    per_frame = RESULTS / f"{name}_{preset}_per_frame.json"
    agg_path = RESULTS / f"{name}_{preset}.json"
    if agg_path.exists() and per_frame.exists() and not force:
        print(f"  {name}: cached, skipping")
        return json.loads(per_frame.read_text()), json.loads(agg_path.read_text())

    model = (bicubic_upsampler(4) if name == "bicubic"
             else model_callable(name, scaler))
    t0 = time.time()
    records, names, videos = evaluate_split(
        model, split="test", preset=preset, limit=limit, scaler=scaler,
        progress_every=200)
    print(f"  {name}: {len(records)} frames in {time.time() - t0:.0f}s")

    agg = aggregate(records, videos)
    RESULTS.mkdir(exist_ok=True)
    pf = {"names": names, "videos": videos, "records": records}
    per_frame.write_text(json.dumps(pf, indent=1))
    agg_path.write_text(json.dumps(
        {"model": name, "preset": preset, "n_frames": len(records),
         "n_videos": len(set(videos)), "aggregate": agg}, indent=1))
    return pf, {"aggregate": agg}


def build_comparison(per_frame, aggs, preset):
    models = [m for m in DISPLAY_ORDER if m in aggs]
    ref = "bicubic"
    ref_pf = per_frame[ref]

    tests = {}
    if ref in per_frame:
        for m in models:
            if m == ref:
                continue
            if per_frame[m]["names"] != ref_pf["names"]:
                print(f"  WARN {m}: frame list differs from bicubic, skipping tests")
                continue
            keys = [k for k, _, _ in HEADLINE]
            raw = {k: paired_test([r.get(k) for r in per_frame[m]["records"]],
                                  [r.get(k) for r in ref_pf["records"]],
                                  clusters=per_frame[m]["videos"]) for k in keys}
            _, adj = holm_bonferroni([raw[k]["p_value"] for k in keys])
            tests[m] = {k: {**raw[k], "p_adjusted": float(p)}
                        for k, p in zip(keys, adj)}

    # markdown table
    lines = [f"# Model comparison ({preset} degradation, x4)\n",
             f"Test split, cluster-bootstrap 95% CI over videos. n_frames="
             f"{aggs[models[0]]['aggregate'][HEADLINE[0][0]]['n']}, "
             f"videos={aggs[models[0]]['aggregate'][HEADLINE[0][0]]['n_clusters']}.\n"]
    header = "| metric | " + " | ".join(MODEL_SPECS.get(m, {}).get("label", m)
                                        for m in models) + " |"
    lines += [header, "|" + "---|" * (len(models) + 1)]
    for key, label, _ in HEADLINE:
        row = [label]
        for m in models:
            a = aggs[m]["aggregate"].get(key, {})
            mean = a.get("mean", float("nan"))
            row.append(f"{mean:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines) + "\n"
    (RESULTS / f"comparison_{preset}.md").write_text(md)
    (RESULTS / f"comparison_{preset}.json").write_text(json.dumps(
        {"preset": preset, "models": models,
         "aggregate": {m: aggs[m]["aggregate"] for m in models},
         "paired_tests_vs_bicubic": tests}, indent=1))
    print("\n" + md)
    print(f"wrote results/comparison_{preset}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--only", default=None, help="evaluate just this model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scaler = load_scaler()
    wanted = [args.only] if args.only else (["bicubic"] + available_models())
    print(f"evaluating: {wanted}")

    per_frame, aggs = {}, {}
    for name in wanted:
        pf, agg = eval_one(name, scaler, args.preset, args.limit, args.force)
        per_frame[name] = pf
        aggs[name] = agg

    # Always (re)load bicubic for the comparison, even under --only.
    for name in ["bicubic"] + available_models():
        if name in aggs:
            continue
        p = RESULTS / f"{name}_{args.preset}_per_frame.json"
        a = RESULTS / f"{name}_{args.preset}.json"
        if p.exists() and a.exists():
            per_frame[name] = json.loads(p.read_text())
            aggs[name] = json.loads(a.read_text())

    if len(aggs) > 1 and "bicubic" in aggs:
        build_comparison(per_frame, aggs, args.preset)


if __name__ == "__main__":
    main()

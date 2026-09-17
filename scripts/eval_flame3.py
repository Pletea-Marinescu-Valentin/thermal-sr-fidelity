"""Run the fidelity protocol over the FLAME 3 test split.

Mirrors `eval_all.py` for the second dataset. The test split is the CV subset,
flown with a DJI M30T over Sycan Marsh; every model was trained on the NADIR
plots, flown with an Autel XT709 over a different burn in a different state.
The cross-instrument transfer is therefore part of the main result here rather
than a separate stress test, and the numbers should be read in that light.

Bootstrap clusters follow how the CV subset was sampled. Its frames are not a
video: they are curated pairs drawn sparsely across the burn, empirically
independent of one another, so each frame is its own unit. See
`tsrf.data.flame3.bootstrap_cluster`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.flame3 import Flame3Scaler, bootstrap_cluster  # noqa: E402
from tsrf.eval import (  # noqa: E402
    aggregate, bicubic_upsampler, evaluate_split_flame3, load_checkpoint,
    torch_upsampler,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import DISPLAY_ORDER, flame3_specs  # noqa: E402
from tsrf.stats import holm_bonferroni, paired_test  # noqa: E402

RESULTS = Path("results")
HEADLINE = [
    ("std_psnr_db", "PSNR (dB)"),
    ("std_ssim", "SSIM"),
    ("std_lpips", "LPIPS"),
    ("m0_rmse_k", "M0 RMSE (K)"),
    ("m0_max_abs_error_k", "M0 max err (K)"),
    ("m1_d10_mask_iou", "M1 IoU@10K"),
    ("m1_d10_peak_error_on_gt_support_k", "M1 peak err (K)"),
    ("m2_d10_n_hallucinated", "M2 halluc/frame"),
    ("m2_d10_miss_rate", "M2 miss rate"),
    ("m3_spearman_rho", "M3 ordering rho"),
    ("m4_texture_ratio", "M4 texture ratio"),
    ("m5_gradient_ratio", "M5 gradient ratio"),
    ("m6_flat_delta_hurst", "M6 delta Hurst"),
    ("m6_flat_dim_sr", "M6 box dimension"),
    ("m7_flat_hf_amp_ratio", "M7 texture amp"),
    ("m7_flat_hf_corr", "M7 texture corr"),
]


def model_callable(name, scaler):
    spec = flame3_specs()[name]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    ckpt = Path(spec["run"]) / spec["checkpoint"]
    net, step, _ = load_checkpoint(net, ckpt)
    print(f"  {name}: loaded {ckpt} (step {step})")
    return torch_upsampler(net, scaler)


def eval_one(name, scaler, preset, limit, force):
    per_frame = RESULTS / f"f3_{name}_{preset}_per_frame.json"
    agg_path = RESULTS / f"f3_{name}_{preset}.json"
    if per_frame.exists() and not force:
        print(f"  {name}: reusing cached per-frame records")
        pf = json.loads(per_frame.read_text())
        records, names = pf["records"], pf["names"]
        # Clusters are always rederived from the frame paths rather than read
        # back from the cache, so a correction to the sampling-unit rule takes
        # effect on existing results without re-running every model.
        clusters = [bootstrap_cluster(n) for n in names]
    else:
        model = (bicubic_upsampler(4) if name == "bicubic"
                 else model_callable(name, scaler))
        t0 = time.time()
        records, names, clusters = evaluate_split_flame3(
            model, "f3test", preset=preset, limit=limit, scaler=scaler,
            progress_every=200)
        print(f"  {name}: {len(records)} frames in {time.time() - t0:.0f}s")

    agg = aggregate(records, clusters)
    RESULTS.mkdir(exist_ok=True)
    pf = {"names": names, "clusters": clusters, "records": records}
    per_frame.write_text(json.dumps(pf, indent=1))
    agg_path.write_text(json.dumps(
        {"model": name, "preset": preset, "dataset": "flame3",
         "n_frames": len(records), "n_clusters": len(set(clusters)),
         "aggregate": agg}, indent=1))
    return pf, {"aggregate": agg}


def build_comparison(per_frame, aggs, preset):
    models = [m for m in DISPLAY_ORDER if m in aggs]
    ref, ref_pf = "bicubic", per_frame["bicubic"]

    tests = {}
    for m in models:
        if m == ref or per_frame[m]["names"] != ref_pf["names"]:
            continue
        keys = [k for k, _ in HEADLINE]
        raw = {k: paired_test([r.get(k) for r in per_frame[m]["records"]],
                              [r.get(k) for r in ref_pf["records"]],
                              clusters=per_frame[m]["clusters"]) for k in keys}
        _, adj = holm_bonferroni([raw[k]["p_value"] for k in keys])
        tests[m] = {k: {**raw[k], "p_adjusted": float(p)}
                    for k, p in zip(keys, adj)}

    meta = aggs[models[0]]["aggregate"][HEADLINE[0][0]]
    lines = [f"# FLAME 3 comparison ({preset} degradation, x4)\n",
             f"Test split: CV subset (DJI M30T), models trained on the NADIR "
             f"plots (Autel XT709). n={meta['n']} frames, "
             f"{meta['n_clusters']} clusters.\n",
             "| metric | " + " | ".join(models) + " |",
             "|" + "---|" * (len(models) + 1)]
    for key, label in HEADLINE:
        row = [label]
        for m in models:
            a = aggs[m]["aggregate"].get(key, {})
            row.append(f"{a.get('mean', float('nan')):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines) + "\n"
    (RESULTS / f"f3_comparison_{preset}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"f3_comparison_{preset}.json").write_text(json.dumps(
        {"preset": preset, "dataset": "flame3", "models": models,
         "aggregate": {m: aggs[m]["aggregate"] for m in models},
         "paired_tests_vs_bicubic": tests}, indent=1))
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scaler = Flame3Scaler()
    wanted = ["bicubic"] + [m for m in DISPLAY_ORDER if m != "bicubic"
                            and (Path(flame3_specs()[m]["run"])
                                 / flame3_specs()[m]["checkpoint"]).exists()]
    print(f"evaluating on FLAME 3: {wanted}")

    per_frame, aggs = {}, {}
    for name in wanted:
        pf, agg = eval_one(name, scaler, args.preset, args.limit, args.force)
        per_frame[name], aggs[name] = pf, agg

    if len(aggs) > 1:
        build_comparison(per_frame, aggs, args.preset)


if __name__ == "__main__":
    main()

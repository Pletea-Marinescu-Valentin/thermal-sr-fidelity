"""Ablation: what does the texture-scaling loss actually buy?

The loss of `tsrf.train.losses` is the metric of M6 turned into an objective.
It should therefore be able to fix what M6 measures. The question this script
answers is whether fixing it also fixes fidelity, or whether the two come
apart: a model can be pushed to emit texture with the reference's amplitude and
the reference's scaling without emitting the reference's texture, because the
loss is a statistic of each image separately and never compares them.

Writes results/ablation_texture.json and a markdown summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.eval import (  # noqa: E402
    aggregate, bicubic_upsampler, evaluate_split, load_checkpoint, load_scaler,
    torch_upsampler,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402
from tsrf.stats import holm_bonferroni, paired_test  # noqa: E402

RESULTS = Path("results")

ROWS = [
    ("std_psnr_db", "PSNR (dB)"),
    ("std_lpips", "LPIPS"),
    ("m0_rmse_k", "M0 RMSE (K)"),
    ("m4_texture_ratio", "M4 texture ratio"),
    ("m6_flat_delta_hurst", "M6 delta Hurst"),
    ("m6_flat_dim_sr", "M6 box dimension"),
    ("m7_flat_hf_amp_ratio", "M7 texture amplitude"),
    ("m7_flat_hf_corr", "M7 texture correspondence"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/edsr_tex_x4_classic")
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default="EDSR + texture loss")
    args = ap.parse_args()

    # Key the outputs to the run, so a sweep over lambda does not have each
    # arm silently overwrite the previous one. Run directories already end in
    # the preset (runs/edsr_tex_x4_classic), so drop it rather than repeat it.
    tag = Path(args.run).name
    if tag.endswith(f"_{args.preset}"):
        tag = tag[: -len(args.preset) - 1]
    scaler = load_scaler()
    spec = MODEL_SPECS["edsr_lite"]
    net = build_model(spec["builder"], scale=4, **spec["kwargs"])
    net, step, _ = load_checkpoint(net, Path(args.run) / args.checkpoint)
    print(f"loaded {args.run}/{args.checkpoint} (step {step})")

    t0 = time.time()
    records, names, videos = evaluate_split(
        torch_upsampler(net, scaler), split="test", preset=args.preset,
        limit=args.limit, scaler=scaler, progress_every=200)
    print(f"{len(records)} frames in {time.time() - t0:.0f}s")

    agg = aggregate(records, videos)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{tag}_{args.preset}_per_frame.json").write_text(
        json.dumps({"names": names, "videos": videos, "records": records}, indent=1))

    # Compare against the same architecture trained with L1 alone: the only
    # difference between the two arms is the objective.
    baseline = json.loads(
        (RESULTS / f"edsr_lite_{args.preset}_per_frame.json").read_text())
    tests = {}
    if baseline["names"][:len(names)] == names:
        keys = [k for k, _ in ROWS]
        raw = {k: paired_test([r.get(k) for r in records],
                              [r.get(k) for r in baseline["records"][:len(names)]],
                              clusters=videos) for k in keys}
        _, adj = holm_bonferroni([raw[k]["p_value"] for k in keys])
        tests = {k: {**raw[k], "p_adjusted": float(p)} for k, p in zip(keys, adj)}
    else:
        print("WARN: frame lists differ, skipping paired tests")

    others = {}
    for name in ("bicubic", "edsr_lite", "esrgan"):
        f = RESULTS / f"{name}_{args.preset}.json"
        if f.exists():
            others[name] = json.loads(f.read_text())["aggregate"]

    (RESULTS / f"{tag}_{args.preset}.json").write_text(json.dumps(
        {"run": args.run, "step": step, "preset": args.preset,
         "n_frames": len(records), "n_videos": len(set(videos)),
         "aggregate": agg, "paired_tests_vs_edsr_l1": tests}, indent=1))

    cols = [("bicubic", "Bicubic"), ("edsr_lite", "EDSR (L1)"),
            ("esrgan", "ESRGAN")]
    lines = [f"# Texture-loss ablation ({args.preset}, x4)\n",
             f"n={len(records)} frames / {len(set(videos))} videos. "
             f"Checkpoint {args.run}/{args.checkpoint} at step {step}.\n",
             "| metric | " + " | ".join(l for _, l in cols)
             + f" | {args.label} | p vs EDSR (L1) |",
             "|" + "---|" * (len(cols) + 3)]
    for key, label in ROWS:
        row = [label]
        for name, _ in cols:
            a = others.get(name, {}).get(key)
            row.append(f"{a['mean']:.3f}" if a else "--")
        row.append(f"{agg[key]['mean']:.3f} [{agg[key]['lo']:.3f}, "
                   f"{agg[key]['hi']:.3f}]")
        row.append(f"{tests[key]['p_adjusted']:.1e}" if key in tests else "--")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines) + "\n"
    (RESULTS / f"{tag}_{args.preset}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()

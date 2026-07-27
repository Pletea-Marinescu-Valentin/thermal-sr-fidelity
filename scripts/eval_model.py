import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.eval import (  # noqa: E402
    aggregate, evaluate_split, load_checkpoint, load_scaler, torch_upsampler,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.stats import holm_bonferroni, paired_test  # noqa: E402

from eval_baseline import HEADLINE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--model", default="edsr_lite")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--feats", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", default=None,
                    help="per-frame json of a reference model for paired tests")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    scaler = load_scaler()
    net = build_model(args.model, scale=args.scale,
                      n_feats=args.feats, n_blocks=args.blocks)
    net, step, best = load_checkpoint(net, args.checkpoint)
    print(f"{args.model} from {args.checkpoint} (step {step}, val PSNR {best})")

    t0 = time.time()
    records, names, videos = evaluate_split(
        torch_upsampler(net, scaler), split="test", preset=args.preset,
        limit=args.limit, scaler=scaler)
    print(f"scored {len(records)} frames / {len(set(videos))} videos "
          f"in {time.time() - t0:.0f}s")

    agg = aggregate(records, videos)
    print(f"\n{'metric':<42} {'mean':>10}  {'95% CI':>22}")
    print("-" * 78)
    for key, label, unit in HEADLINE:
        if key in agg:
            a = agg[key]
            print(f"{label:<42} {a['mean']:>10.4f}  "
                  f"[{a['lo']:>9.4f}, {a['hi']:>9.4f}]{' ' + unit if unit else ''}")

    tests = {}
    if args.compare:
        ref = json.loads(Path(args.compare).read_text())
        if ref["names"] != names:
            raise SystemExit("reference was scored on a different frame list")
        keys = [k for k, _, _ in HEADLINE if k in agg]
        raw = {k: paired_test([r.get(k) for r in records],
                              [r.get(k) for r in ref["records"]], clusters=videos)
               for k in keys}
        rejected, adj = holm_bonferroni([raw[k]["p_value"] for k in keys])
        print(f"\npaired Wilcoxon vs {Path(args.compare).name} "
              f"(video-level, Holm-Bonferroni):")
        print(f"{'metric':<42} {'median diff':>12} {'p_adj':>10}  sig")
        print("-" * 74)
        for k, rej, p in zip(keys, rejected, adj):
            tests[k] = {**raw[k], "p_adjusted": float(p), "significant": bool(rej)}
            md = raw[k].get("median_diff", float("nan"))
            print(f"{k:<42} {md:>12.4f} {p:>10.4g}  {'yes' if rej else 'no'}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    tag = args.tag or f"{args.model}_{args.preset}"
    (out / f"{tag}.json").write_text(json.dumps(
        {"model": args.model, "checkpoint": args.checkpoint, "step": step,
         "preset": args.preset, "n_frames": len(records),
         "n_videos": len(set(videos)), "aggregate": agg, "tests": tests}, indent=1))
    (out / f"{tag}_per_frame.json").write_text(json.dumps(
        {"names": names, "videos": videos, "records": records}, indent=1))
    print(f"\nwrote results/{tag}.json")


if __name__ == "__main__":
    main()

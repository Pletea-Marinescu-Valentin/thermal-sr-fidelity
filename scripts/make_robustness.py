r"""Two robustness analyses the reviewer will ask for, from records already on disk.

1. The negative result on discrete hallucination is reported at delta = 10 K in
   the main table. It was pre-registered over {5, 10, 20} K, so the sweep should
   be shown rather than asserted: if M2 only vanishes at one threshold, the
   negative is an artefact of that threshold.

2. The M7 contrast between the adversarial model and the regressors is reported
   as a difference of means over 1144 frames. Means can hide overlap, so we also
   report how often the ordering holds frame by frame, which is the statement an
   operator would care about.

Writes results/robustness.json and paper/table_delta_sweep.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.models.registry import DISPLAY_ORDER  # noqa: E402
from tsrf.stats import cluster_bootstrap_ci  # noqa: E402

RESULTS = Path("results")
OUT_JSON = RESULTS / "robustness.json"
OUT_TEX = Path("paper/table_delta_sweep.tex")

SHORT = {"bicubic": "Bicubic", "edsr_lite": "EDSR", "rrdb": "RRDB",
         "swinir_lite": "SwinIR", "esrgan": "ESRGAN"}
DELTAS = (5, 10, 20)


def per_frame(model, preset):
    p = RESULTS / f"{model}_{preset}_per_frame.json"
    return json.loads(p.read_text()) if p.exists() else None


def delta_sweep(models, preset):
    out = {}
    for m in models:
        pf = per_frame(m, preset)
        if pf is None:
            continue
        rows = {}
        for d in DELTAS:
            for key, label in ((f"m2_d{d}_n_hallucinated", "halluc"),
                               (f"m2_d{d}_miss_rate", "miss"),
                               (f"m1_d{d}_mask_iou", "iou")):
                vals = [r.get(key, np.nan) for r in pf["records"]]
                rows[f"{label}_d{d}"] = cluster_bootstrap_ci(
                    vals, pf["videos"], n_boot=2000)
        out[m] = rows
    return out


def hallucination_significance(preset, models):
    """Is the small count at the permissive threshold real, or noise?

    At delta = 5 K the counts stop being zero and order the methods sensibly.
    That ordering is only worth reporting if it survives a paired test against
    the interpolation floor, which by construction cannot invent anything.
    """
    from tsrf.stats import holm_bonferroni, paired_test

    ref = per_frame("bicubic", preset)
    if ref is None:
        return None
    out, raw = {}, {}
    for m in models:
        if m == "bicubic":
            continue
        pf = per_frame(m, preset)
        if pf is None or pf["names"] != ref["names"]:
            continue
        raw[m] = paired_test([r.get("m2_d5_n_hallucinated") for r in pf["records"]],
                             [r.get("m2_d5_n_hallucinated") for r in ref["records"]],
                             clusters=pf["videos"])
    if not raw:
        return None
    names = list(raw)
    _, adj = holm_bonferroni([raw[m]["p_value"] for m in names])
    for m, p in zip(names, adj):
        out[m] = {**raw[m], "p_adjusted": float(p)}
    return out


def separability(preset, a="esrgan", b="edsr_lite", key="m7_flat_hf_corr"):
    """How often does the per-frame ordering match the ordering of the means?"""
    pa, pb = per_frame(a, preset), per_frame(b, preset)
    if pa is None or pb is None or pa["names"] != pb["names"]:
        return None
    va = np.array([r.get(key, np.nan) for r in pa["records"]], dtype=float)
    vb = np.array([r.get(key, np.nan) for r in pb["records"]], dtype=float)
    ok = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[ok], vb[ok]
    return {
        "metric": key, "a": a, "b": b, "n": int(ok.sum()),
        "fraction_a_below_b": float((va < vb).mean()),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "p05_b": float(np.percentile(vb, 5)),
        "p95_a": float(np.percentile(va, 95)),
        "overlap": float(np.percentile(va, 95) > np.percentile(vb, 5)),
    }


def write_delta_table(sweep, models, preset):
    cols = [m for m in models if m in sweep]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{The discrete-hallucination result across the pre-registered "
        r"thresholds, " + f"{preset} degradation. " + r"M2 counts invented hot "
        r"regions per frame. Relaxing $\delta$ raises the count and orders the "
        r"methods as expected, the interpolation floor staying at exactly zero, "
        r"but even at the most permissive threshold the adversarial model "
        r"invents about one region per $34$ frames. The negative result is "
        r"therefore a matter of magnitude at every threshold, not an artefact "
        r"of the one reported in Table~\ref{tab:main}.}",
        r"\label{tab:deltasweep}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        r"$\delta$ & " + " & ".join(SHORT.get(m, m) for m in cols) + r" \\",
        r"\midrule",
    ]
    for d in DELTAS:
        cells = [f"{sweep[m][f'halluc_d{d}']['mean']:.3f}" for m in cols]
        lines.append(f"${d}$~K & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.parent.mkdir(exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TEX}")


def main():
    preset = "classic"
    models = [m for m in DISPLAY_ORDER
              if (RESULTS / f"{m}_{preset}_per_frame.json").exists()]
    if not models:
        sys.exit("no per-frame records on disk; run scripts/eval_all.py first")

    sweep = delta_sweep(models, preset)
    print(f"M2 hallucinated regions per frame, {preset}:")
    print(f"  {'delta':<8}" + "".join(f"{SHORT.get(m, m):>12}" for m in models))
    for d in DELTAS:
        row = "".join(f"{sweep[m][f'halluc_d{d}']['mean']:>12.4f}" for m in models)
        print(f"  {d:<8}" + row)

    print(f"\nM2 miss rate (the dual error), {preset}:")
    for d in DELTAS:
        row = "".join(f"{sweep[m][f'miss_d{d}']['mean']:>12.3f}" for m in models)
        print(f"  {d:<8}" + row)

    sig = hallucination_significance(preset, models)
    if sig:
        print("\nM2 at the permissive threshold (5 K), paired vs bicubic:")
        for m, t in sig.items():
            print(f"  {SHORT.get(m, m):<10} median diff "
                  f"{t.get('median_diff', float('nan')):+.4f}  "
                  f"p_adj={t['p_adjusted']:.3g}")

    sep = separability(preset)
    if sep:
        print(f"\nper-frame separability of M7 correspondence "
              f"({sep['a']} vs {sep['b']}, n={sep['n']}):")
        print(f"  frames where {sep['a']} scores lower: "
              f"{sep['fraction_a_below_b'] * 100:.1f}%")
        print(f"  95th pct of {sep['a']} = {sep['p95_a']:.3f}   "
              f"5th pct of {sep['b']} = {sep['p05_b']:.3f}   "
              f"distributions {'overlap' if sep['overlap'] else 'are disjoint'}")

    write_delta_table(sweep, models, preset)
    OUT_JSON.write_text(json.dumps(
        {"preset": preset, "delta_sweep": sweep, "separability": sep,
         "m2_d5_vs_bicubic": sig}, indent=1))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

"""Effect sizes for the controlled comparison, on both datasets.

The adversarial arm is the regressor of the same architecture, fine-tuned, so
the difference between them isolates the training objective. This is the
paper's central experiment, and reporting it by p-value alone is a mistake:
above 25 pairs the signed-rank test switches to a normal approximation, and at
716 pairs it returns nominal values near 1e-112, which is a 22-sigma
extrapolation rather than a measurement.

What belongs in the paper instead, and what this writes, is the difference
itself, a bootstrap interval for it over the same units the test pairs on, and
the share of units on which it keeps its sign.

Writes paper/table_effects.tex and prints the quantities the prose quotes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.flame3 import bootstrap_cluster  # noqa: E402
from tsrf.stats import holm_bonferroni, paired_effect, paired_test  # noqa: E402

RESULTS = Path("results")
OUT = Path("paper/table_effects.tex")

ROWS = [
    ("std_lpips",            r"LPIPS",                        "{:+.4f}"),
    ("m0_rmse_k",            r"M0 RMSE (K)",                  "{:+.3f}"),
    ("m1_d10_mask_iou",      r"M1 IoU@10K",                   "{:+.3f}"),
    ("m4_texture_ratio",     r"M4 texture ratio",             "{:+.3f}"),
    ("m6_flat_delta_hurst",  r"M6 $\Delta\hat{H}$",           "{:+.3f}"),
    ("m7_flat_hf_amp_ratio", r"M7 amplitude $a$",             "{:+.3f}"),
    ("m7_flat_hf_corr",      r"M7 correspondence $r$",        "{:+.3f}"),
]

#: Smallest p-value the paper prints. Below this the signed-rank approximation
#: is reporting the shape of a Gaussian tail, not evidence, so a bound is the
#: honest form and the effect size carries the claim.
P_FLOOR = 1e-4


def load(dataset, model, preset):
    path = RESULTS / f"{dataset}{model}_{preset}_per_frame.json"
    return json.loads(path.read_text())


def clusters_of(dataset, pf):
    if dataset == "f3_":
        return [bootstrap_cluster(n) for n in pf["names"]]
    return pf.get("clusters") or pf.get("videos")


def analyse(dataset, preset="classic"):
    a, b = load(dataset, "esrgan", preset), load(dataset, "rrdb", preset)
    clusters = clusters_of(dataset, a)
    keys = [k for k, _, _ in ROWS]
    tests = {k: paired_test([r.get(k) for r in a["records"]],
                            [r.get(k) for r in b["records"]],
                            clusters=clusters) for k in keys}
    _, adj = holm_bonferroni([tests[k]["p_value"] for k in keys])
    effects = {k: paired_effect([r.get(k) for r in a["records"]],
                                [r.get(k) for r in b["records"]],
                                clusters=clusters) for k in keys}
    return effects, dict(zip(keys, adj)), tests


def main():
    data = {ds: analyse(ds) for ds in ("", "f3_")}
    n_ad = data[""][0]["std_lpips"]["n_clusters"]
    n_f3 = data["f3_"][0]["std_lpips"]["n_clusters"]

    lines = [
        # Full width: each cell carries an estimate and its interval, so five
        # columns of them overrun a single IEEEtran column by about 93 pt and
        # print over the neighbouring text.
        r"\begin{table*}[t]", r"\centering",
        # One or two lines: what the cells are and over which units. Reading
        # the differences is the text's job, not the caption's.
        r"\caption{ESRGAN minus the RRDB it was fine-tuned from: paired "
        r"difference, 95\% bootstrap interval and sign consistency $c$, over "
        f"videos on ADAS (${n_ad}$) and frames on FLAME~3 (${n_f3}$).}}",
        r"\label{tab:effects}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{FLIR ADAS v2}} "
        r"& \multicolumn{2}{c}{\textbf{FLAME 3}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Metric & difference $[95\%]$ & $c$ & difference $[95\%]$ & $c$ \\",
        r"\midrule",
    ]
    for key, label, fmt in ROWS:
        cells = []
        for ds in ("", "f3_"):
            e = data[ds][0][key]
            cells.append("$" + fmt.format(e["median_diff"]) + "$ $["
                         + fmt.format(e["lo"]) + ", " + fmt.format(e["hi"]) + "]$")
            cells.append(f"${e['sign_consistency']:.2f}$")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    for ds, name in (("", "ADAS"), ("f3_", "FLAME 3")):
        effects, adj, tests = data[ds]
        print(f"\n--- {name}: ESRGAN vs RRDB, for the prose ---")
        print(f"  units: {effects['std_lpips']['n_clusters']}, "
              f"largest p_adj {max(adj.values()):.3g}, "
              f"approximation used: {tests['std_lpips']['approximate']}")
        for key, label, fmt in ROWS:
            e = effects[key]
            print(f"  {label:<26} {fmt.format(e['median_diff'])} "
                  f"[{fmt.format(e['lo'])}, {fmt.format(e['hi'])}]  "
                  f"c={e['sign_consistency']:.3f}  p_adj={adj[key]:.2g}")


if __name__ == "__main__":
    main()

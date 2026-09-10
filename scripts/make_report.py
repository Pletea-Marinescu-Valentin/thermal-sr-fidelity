import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.models.registry import DISPLAY_ORDER, MODEL_SPECS  # noqa: E402

RESULTS = Path("results")
KEY_METRICS = [
    ("std_psnr_db", "PSNR (dB)", "{:.2f}"),
    ("std_ssim", "SSIM", "{:.4f}"),
    ("std_lpips", "LPIPS", "{:.4f}"),
    ("m0_rmse_k", "M0 RMSE (K)", "{:.3f}"),
    ("m1_d10_peak_error_on_gt_support_k", "M1 peak err (K)", "{:.2f}"),
    ("m1_d10_mask_iou", "M1 IoU@10K", "{:.3f}"),
    ("m2_d10_n_hallucinated", "M2 halluc/frame", "{:.3f}"),
    ("m2_d10_miss_rate", "M2 miss rate", "{:.3f}"),
    ("m3_top1_preserved", "M3 top-1 kept", "{:.3f}"),
    ("m4_texture_ratio", "M4 texture ratio", "{:.3f}"),
    ("m5_gradient_ratio", "M5 gradient ratio", "{:.3f}"),
    ("m6_flat_delta_hurst", "M6 delta Hurst", "{:+.3f}"),
    ("m6_flat_dim_sr", "M6 box dimension", "{:.3f}"),
    ("m7_flat_hf_amp_ratio", "M7 texture amplitude", "{:.3f}"),
    ("m7_flat_hf_corr", "M7 texture correspondence", "{:.3f}"),
]


def fmt_ci(a, fmt):
    if not a or a.get("mean") is None:
        return "-"
    return f"{fmt.format(a['mean'])} [{fmt.format(a['lo'])}, {fmt.format(a['hi'])}]"


def section_comparison(preset):
    path = RESULTS / f"comparison_{preset}.json"
    if not path.exists():
        return f"## {preset} degradation\n\n_(no comparison yet)_\n"
    data = json.loads(path.read_text())
    models = [m for m in DISPLAY_ORDER if m in data["aggregate"]]
    labels = [MODEL_SPECS.get(m, {}).get("label", m) for m in models]

    lines = [f"## {preset} degradation (x4)\n",
             "| metric | " + " | ".join(labels) + " |",
             "|" + "---|" * (len(models) + 1)]
    for key, label, fmt in KEY_METRICS:
        row = [label]
        for m in models:
            row.append(fmt_ci(data["aggregate"][m].get(key), fmt))
        lines.append("| " + " | ".join(row) + " |")

    tests = data.get("paired_tests_vs_bicubic", {})
    if tests:
        lines.append("\n**Paired Wilcoxon vs bicubic (video-level, Holm-Bonferroni "
                     "adjusted p):**\n")
        lines.append("| model | " + " | ".join(l for _, l, _ in KEY_METRICS) + " |")
        lines.append("|" + "---|" * (len(KEY_METRICS) + 1))
        for m in models:
            if m == "bicubic" or m not in tests:
                continue
            cells = [MODEL_SPECS.get(m, {}).get("label", m)]
            for key, _, _ in KEY_METRICS:
                t = tests[m].get(key, {})
                p = t.get("p_adjusted")
                cells.append("n/a" if p is None else
                             ("**{:.1e}**".format(p) if p < 0.05 else f"{p:.2f}"))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def section_iisr():
    path = RESULTS / "iisr_bicubic.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    a = d["aggregate"]
    lines = ["## FLIR-IISR: real vs synthetic degradation (bicubic floor)\n",
             f"n={d['n_frames']} frames. Intensity levels, not Kelvin (8-bit AGC).\n",
             "| metric | synthetic | real |", "|---|---|---|"]
    for key, label, fmt in [("std_psnr_db", "PSNR (dB)", "{:.2f}"),
                            ("m1_d60_mask_iou", "M1 IoU@+60", "{:.3f}"),
                            ("m2_d60_n_missed", "M2 missed/frame", "{:.2f}"),
                            ("m3_spearman_rho", "M3 ordering rho", "{:.3f}")]:
        s, r = a["synthetic"].get(key, {}), a["real"].get(key, {})
        lines.append(f"| {label} | {fmt_ci(s, fmt)} | {fmt_ci(r, fmt)} |")
    return "\n".join(lines) + "\n"


def section_control():
    path = RESULTS / "positive_control.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    arms = ["attenuated", "fabricated_global", "fabricated_local"]
    lines = ["## Positive control: constructed references\n",
             f"n={d['n_frames']} frames. Predictions whose answer is known by "
             "construction; the fabricated arms carry no measured signal.\n",
             "| metric | " + " | ".join(arms) + " |", "|" + "---|" * 4]
    for key, label, fmt in [("m4_texture_ratio", "M4 texture ratio", "{:.3f}"),
                            ("m6_flat_delta_hurst", "M6 delta Hurst", "{:+.3f}"),
                            ("m7_flat_hf_amp_ratio", "M7 amplitude", "{:.3f}"),
                            ("m7_flat_hf_corr", "M7 correspondence", "{:.3f}")]:
        cells = [fmt_ci(d["aggregate"][a].get(key), fmt) for a in arms]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def section_robustness():
    path = RESULTS / "robustness.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    sweep = d["delta_sweep"]
    models = [m for m in DISPLAY_ORDER if m in sweep]
    labels = [MODEL_SPECS.get(m, {}).get("label", m) for m in models]
    lines = ["## Threshold sweep: M2 hallucinated regions per frame\n",
             "| delta | " + " | ".join(labels) + " |",
             "|" + "---|" * (len(models) + 1)]
    for dk in (5, 10, 20):
        cells = [f"{sweep[m][f'halluc_d{dk}']['mean']:.4f}" for m in models]
        lines.append(f"| {dk} K | " + " | ".join(cells) + " |")
    sep = d.get("separability")
    if sep:
        lines.append(
            f"\nPer-frame separability of M7 ({sep['a']} vs {sep['b']}): "
            f"{sep['fraction_a_below_b'] * 100:.1f}% of {sep['n']} frames; "
            f"95th pct {sep['p95_a']:.3f} vs 5th pct {sep['p05_b']:.3f}.\n")
    return "\n".join(lines) + "\n"


def main():
    parts = ["# Thermal SR fidelity — results report\n",
             "Auto-generated by scripts/make_report.py from results/*.json.\n"]
    for preset in ("classic", "realistic"):
        parts.append(section_comparison(preset))
    for extra in (section_control(), section_robustness(), section_iisr()):
        if extra:
            parts.append(extra)

    figs = sorted((RESULTS / "figures").glob("*.png")) if (RESULTS / "figures").exists() else []
    if figs:
        parts.append("## Qualitative figures\n\n" +
                     "\n".join(f"- `{f.as_posix()}`" for f in figs) + "\n")

    report = "\n".join(parts)
    # Explicit encoding: the default on Windows is cp1252, which writes the
    # em dash as a byte no UTF-8 reader can decode.
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nwrote results/REPORT.md")


if __name__ == "__main__":
    main()

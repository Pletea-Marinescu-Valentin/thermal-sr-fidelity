r"""Emit the paper's result tables straight from results/*.json.

Transcribing numbers by hand is how a table drifts from the data it claims to
report, and it double-rounds: 0.474475 becomes 0.4745 and then 0.475. This
writes paper/table_main.tex and paper/table_texture.tex, which the paper
\input{}s, so every printed cell is formatted exactly once from the aggregate.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.models.registry import DISPLAY_ORDER  # noqa: E402

RESULTS = Path("results")
OUT_TEXTURE = Path("paper/table_texture.tex")
OUT_MAIN = Path("paper/table_main.tex")

SHORT = {"bicubic": "Bicubic", "edsr_lite": "EDSR", "rrdb": "RRDB",
         "swinir_lite": "SwinIR", "esrgan": "ESRGAN"}

UP, DOWN, TO1 = r"$\uparrow$", r"$\downarrow$", r"$\rightarrow 1$"

MAIN_ROWS = [
    ("std_psnr_db",        "PSNR (dB) " + UP,                 "{:.2f}", "high"),
    ("std_ssim",           "SSIM " + UP,                      "{:.4f}", "high"),
    ("std_lpips",          "LPIPS " + DOWN,                   "{:.4f}", "low"),
    (None, None, None, None),
    ("m0_rmse_k",          "M0 RMSE (K) " + DOWN,             "{:.3f}", "low"),
    ("m0_max_abs_error_k", "M0 max error (K) " + DOWN,        "{:.2f}", "low"),
    ("m1_d10_mask_iou",    "M1 IoU@10K " + UP,                "{:.3f}", "high"),
    ("m1_d10_peak_error_on_gt_support_k",
     "M1 peak error (K) " + DOWN,                             "{:.2f}", "low"),
    ("m2_d10_n_hallucinated",
     "M2 hallucinated / frame " + DOWN,                       "{:.3f}", "low"),
    ("m2_d10_miss_rate",   "M2 miss rate " + DOWN,            "{:.3f}", "low"),
    ("m3_spearman_rho",    r"M3 ordering $\rho_s$ " + UP,     "{:.3f}", "high"),
    ("m3_top1_preserved",  "M3 top-1 preserved " + UP,        "{:.3f}", "high"),
    ("m4_texture_ratio",   "M4 texture ratio " + TO1,         "{:.3f}", "one"),
    ("m5_gradient_ratio",  "M5 gradient ratio " + TO1,        "{:.3f}", "one"),
]

TEXTURE_ROWS = [
    ("m6_flat_hurst_sr",     r"$\hat{H}$",                    "{:.3f}",  None),
    ("m6_flat_delta_hurst",  r"$\Delta\hat{H}$ " + r"$\rightarrow 0$", "{:+.3f}", None),
    ("m6_flat_dim_sr",       r"$D$",                          "{:.3f}",  None),
    ("m7_flat_hf_amp_ratio", r"$a$ " + TO1,                   "{:.3f}",  None),
    ("m7_flat_hf_corr",      r"$r$ " + UP,                    "{:.3f}",  None),
]


def load(preset, models):
    return {m: json.loads((RESULTS / f"{m}_{preset}.json").read_text())["aggregate"]
            for m in models}


def best_cells(values, direction, fmt):
    """Indices of the best cells, by the row's own notion of best.

    Comparison is on the *printed* value, not the underlying float: two cells
    that a reader sees as identical must be bolded identically, otherwise the
    table appears to rank 0.9953 above 0.9953.
    """
    shown = [float(fmt.format(v)) for v in values]
    if direction == "high":
        target = max(shown)
    elif direction == "low":
        target = min(shown)
    else:
        target = min(shown, key=lambda v: abs(v - 1.0))
    return {i for i, v in enumerate(shown) if v == target}


def write_main_table(models):
    aggs = {p: load(p, models) for p in ("classic", "realistic")}
    meta = json.loads((RESULTS / f"{models[0]}_classic.json").read_text())
    n, head = len(models), " & ".join(SHORT.get(m, m) for m in models)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Fidelity protocol at $\times 4$ on the FLIR ADAS v2 radiometric "
        f"test split ({meta['n_frames']} frames, {meta['n_videos']} videos). "
        r"Arrows give the preferred direction; $\rightarrow 1$ denotes that both "
        r"over- and under-shoot are errors. Best value per row and degradation in "
        r"bold. Perceptual metrics (top block) and fidelity metrics (bottom block) "
        r"disagree on the adversarial model.}",
        r"\label{tab:main}",
        # Bold digits are wider than regular ones, and marking ties honestly
        # bolds more cells than picking one arbitrary winner would.
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{l " + "c" * n + " c " + "c" * n + "}",
        r"\toprule",
        r"& \multicolumn{" + str(n) + r"}{c}{\textbf{Classic degradation}} & & "
        r"\multicolumn{" + str(n) + r"}{c}{\textbf{Realistic degradation}} \\",
        r"\cmidrule{2-" + str(n + 1) + r"} \cmidrule{" + str(n + 3) + "-"
        + str(2 * n + 2) + "}",
        f"Metric & {head} & & {head} " + r"\\",
        r"\midrule",
    ]
    for key, label, fmt, direction in MAIN_ROWS:
        if key is None:
            lines.append(r"\midrule")
            continue
        blocks = []
        for preset in ("classic", "realistic"):
            vals = [aggs[preset][m][key]["mean"] for m in models]
            best = best_cells(vals, direction, fmt)
            blocks.append([(r"\textbf{" + fmt.format(v) + "}") if i in best
                           else fmt.format(v) for i, v in enumerate(vals)])
        lines.append(f"{label} & " + " & ".join(blocks[0]) + " & & "
                     + " & ".join(blocks[1]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    OUT_MAIN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MAIN}")


def write_texture_table(models, preset):
    aggs = load(preset, models)
    meta = json.loads((RESULTS / f"{models[0]}_{preset}.json").read_text())
    ref_h = aggs[models[0]]["m6_flat_hurst_gt"]["mean"]
    ref_d = aggs[models[0]]["m6_flat_dim_gt"]["mean"]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Texture character in the flat cold set $F$, "
        f"{preset} degradation, " + r"$\times 4$, "
        f"{meta['n_frames']} frames. " + r"$\hat{H}$ is the variance-scaling "
        r"exponent of \eqref{eq:powerlaw} and $D = 3 - \hat{H}$; $a$ and $r$ are "
        r"the amplitude and correspondence of \eqref{eq:m7}. The reference reads "
        + f"$\\hat{{H}} = {ref_h:.3f}$ and $D = {ref_d:.3f}$. "
        + r"The adversarial model matches the reference statistics and carries "
        r"the least of its signal.}",
        r"\label{tab:texture}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Quantity & " + " & ".join(SHORT.get(m, m) for m in models) + r" \\",
        r"\midrule",
    ]
    for key, label, fmt, _ in TEXTURE_ROWS:
        cells = [fmt.format(aggs[m][key]["mean"]) for m in models]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEXTURE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TEXTURE}")

    print("\n--- for the prose ---")
    for m in models:
        a = aggs[m]
        print(f"  {SHORT.get(m, m):8s} dH={a['m6_flat_delta_hurst']['mean']:+.3f} "
              f"[{a['m6_flat_delta_hurst']['lo']:+.3f}, "
              f"{a['m6_flat_delta_hurst']['hi']:+.3f}]  "
              f"r={a['m7_flat_hf_corr']['mean']:.3f} "
              f"[{a['m7_flat_hf_corr']['lo']:.3f}, {a['m7_flat_hf_corr']['hi']:.3f}]  "
              f"a={a['m7_flat_hf_amp_ratio']['mean']:.3f}  "
              f"M4={a['m4_texture_ratio']['mean']:.3f}")
    ref = aggs[models[0]]
    print(f"  reference H={ref_h:.3f} [{ref['m6_flat_hurst_gt']['lo']:.3f}, "
          f"{ref['m6_flat_hurst_gt']['hi']:.3f}] D={ref_d:.3f}  "
          f"fit resid={ref['m6_flat_fit_resid_gt']['mean']:.3f}")


OUT_SENS = Path("paper/table_sensitivity.tex")


def write_sensitivity_table():
    """Design-choice sweep and the scale-resolved correspondence."""
    src = RESULTS / "sensitivity.json"
    if not src.exists():
        print("sensitivity table: results/sensitivity.json missing, skipped")
        return
    d = json.loads(src.read_text())
    models = d["models"]
    head = " & ".join(SHORT.get(m, m) for m in models)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Design-choice sweep and scale-resolved correspondence, "
        + f"{d['n_frames']} test frames. " + r"Top: $\Delta\hat{H}$ over five "
        r"window ladders and four definitions of the flat set. Bottom: M7 "
        r"correspondence as the high-pass window is varied, which locates the "
        r"scale below which no method recovers the measurement. The separation "
        r"between the adversarial model and the regressors survives every "
        r"choice.}",
        r"\label{tab:sensitivity}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Setting & " + head + r" \\",
        r"\midrule",
        r"\multicolumn{" + str(len(models) + 1)
        + r"}{l}{\emph{$\Delta\hat{H}$, window ladder (px)}} \\",
    ]
    for label, row in d["ladders"].items():
        cells = [f"{row[m]:+.3f}" for m in models]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(models) + 1)
                 + r"}{l}{\emph{$\Delta\hat{H}$, flat-set percentile}} \\")
    for label, row in d["dh_by_percentile"].items():
        cells = [f"{row[m]:+.3f}" for m in models]
        lines.append(f"\\quad {float(label):.0f} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(models) + 1)
                 + r"}{l}{\emph{M7 $r$, high-pass window (px)}} \\")
    for label, row in d["corr_by_window"].items():
        cells = [f"{row[m]:.3f}" for m in models]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_SENS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_SENS}")

    # Ranges the prose quotes, so they are not eyeballed off the table.
    for block in ("ladders", "dh_by_percentile", "r_by_percentile"):
        for m in models:
            vals = [row[m] for row in d[block].values()]
            print(f"  {block:<18}{SHORT.get(m, m):<10}"
                  f"[{min(vals):+.3f}, {max(vals):+.3f}]")


IISR_ROWS = [
    ("m4_texture_ratio",     r"M4 $R_{\mathrm{tex}}$ " + TO1,             "{:.2f}"),
    ("m6_flat_delta_hurst",  r"M6 $\Delta\hat{H}$ " + r"$\rightarrow 0$", "{:+.2f}"),
    ("m7_flat_hf_amp_ratio", "M7 $a$ " + TO1,                            "{:.2f}"),
    ("m7_flat_hf_corr",      "M7 $r$ " + UP,                             "{:.3f}"),
]

OUT_IISR = Path("paper/table_iisr.tex")


def write_iisr_table():
    """Cross-sensor table: two methods times two low-resolution origins."""
    files = {t: RESULTS / f"iisr_{t}.json" for t in ("bicubic", "esrgan")}
    if not all(f.exists() for f in files.values()):
        print("iisr table: results missing, skipped")
        return
    d = {t: json.loads(f.read_text()) for t, f in files.items()}
    n = d["bicubic"]["n_frames"]
    ref_h = d["bicubic"]["aggregate"]["synthetic"]["m6_flat_hurst_gt"]["mean"]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-sensor stress test on FLIR-IISR "
        f"({n} frames, 8-bit intensity levels, not Kelvin). " + r"Only the origin "
        r"of the low-resolution input differs between the columns of each pair. "
        r"The reference here reads $\hat{H} = " + f"{ref_h:.2f}$, a much smoother "
        r"field than the radiometric reference of Table~\ref{tab:texture}. Under "
        r"this out-of-domain transfer the adversarial model no longer matches the "
        r"reference scaling, so M6 detects what it cannot detect in domain.}",
        r"\label{tab:iisr}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Bicubic} & \multicolumn{2}{c}{ESRGAN} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Metric & Synth. & Real & Synth. & Real \\",
        r"\midrule",
    ]
    for key, label, fmt in IISR_ROWS:
        cells = [fmt.format(d[t]["aggregate"][arm][key]["mean"])
                 for t in ("bicubic", "esrgan") for arm in ("synthetic", "real")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_IISR.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_IISR}")


CONTROL_ROWS = [
    ("m4_texture_ratio",     r"M4 $R_{\mathrm{tex}}$ " + TO1,             "{:.3f}"),
    ("m6_flat_delta_hurst",  r"M6 $\Delta\hat{H}$ " + r"$\rightarrow 0$", "{:+.3f}"),
    ("m7_flat_hf_amp_ratio", "M7 $a$ " + TO1,                            "{:.3f}"),
    ("m7_flat_hf_corr",      "M7 $r$ " + UP,                             "{:.3f}"),
]

CONTROL_COLUMNS = [
    ("attenuated",        "Attenuated"),
    ("fabricated_global", "Fabricated (global)"),
    ("fabricated_local",  "Fabricated (local)"),
]

OUT_CONTROL = Path("paper/table_control.tex")


def write_control_table():
    """Table for the constructed references of scripts/positive_control.py."""
    src = RESULTS / "positive_control.json"
    if not src.exists():
        print("control table: results/positive_control.json missing, skipped")
        return
    d = json.loads(src.read_text())
    agg = d["aggregate"]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{The protocol on references whose answer is known by "
        r"construction, " + f"{d['n_frames']} radiometric test frames. All three "
        r"keep the reference's low-frequency content and differ only in the fine "
        r"texture: one attenuates the measured texture, two replace it with a "
        r"phase-randomised surrogate carrying no measured signal at all. M4 and "
        r"M6 rank the locally matched fabrication as the more faithful of the "
        r"two failures; only M7 rejects it.}",
        r"\label{tab:control}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(CONTROL_COLUMNS) + "}",
        r"\toprule",
        "Metric & " + " & ".join(lbl for _, lbl in CONTROL_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for key, label, fmt in CONTROL_ROWS:
        cells = [fmt.format(agg[arm][key]["mean"]) for arm, _ in CONTROL_COLUMNS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_CONTROL.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CONTROL}")


ABLATION_ROWS = [
    ("std_psnr_db",          "PSNR (dB) " + UP,          "{:.2f}"),
    ("std_lpips",            "LPIPS " + DOWN,            "{:.4f}"),
    ("m0_rmse_k",            "M0 RMSE (K) " + DOWN,      "{:.3f}"),
    ("m4_texture_ratio",     r"M4 $R_{\mathrm{tex}}$ " + TO1, "{:.3f}"),
    ("m6_flat_delta_hurst",  r"M6 $\Delta\hat{H}$ " + r"$\rightarrow 0$", "{:+.3f}"),
    ("m7_flat_hf_amp_ratio", "M7 $a$ " + TO1,            "{:.3f}"),
    ("m7_flat_hf_corr",      "M7 $r$ " + UP,             "{:.3f}"),
]

#: (results stem, column label) for the ablation table, in increasing lambda.
ABLATION_COLUMNS = [
    ("edsr_lite_classic",    r"$L_1$"),
    ("edsr_tex2_x4_classic", r"$\lambda = 2\!\times\!10^{-3}$"),
    ("edsr_tex_x4_classic",  r"$\lambda = 5\!\times\!10^{-3}$"),
    ("esrgan_classic",       "ESRGAN"),
]

OUT_ABLATION = Path("paper/table_ablation.tex")


def write_ablation_table():
    """Table for the texture-loss sweep, skipping arms not yet evaluated."""
    cols = [(stem, label) for stem, label in ABLATION_COLUMNS
            if (RESULTS / f"{stem}.json").exists()]
    if len(cols) < 2:
        print("ablation table: not enough arms on disk, skipped")
        return
    aggs = {stem: json.loads((RESULTS / f"{stem}.json").read_text())["aggregate"]
            for stem, _ in cols}

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Adding the texture objective \eqref{eq:ltex} to the $L_1$ "
        r"regressor, at two weights, against the same architecture trained with "
        r"$L_1$ alone and the adversarial model. The statistics of the emitted "
        r"texture move to the reference and past it, while correspondence to the "
        r"measurement falls throughout.}",
        r"\label{tab:ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        r"& \multicolumn{" + str(len(cols) - 1) + r"}{c}{EDSR} & \\",
        r"\cmidrule{2-" + str(len(cols)) + "}",
        "Metric & " + " & ".join(label for _, label in cols) + r" \\",
        r"\midrule",
    ]
    for key, label, fmt in ABLATION_ROWS:
        cells = [fmt.format(aggs[stem][key]["mean"]) for stem, _ in cols]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_ABLATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_ABLATION}")
    for key, label, fmt in ABLATION_ROWS:
        vals = "  ".join(f"{fmt.format(aggs[s][key]['mean']):>9}" for s, _ in cols)
        print(f"  {label[:26]:28s}{vals}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic")
    args = ap.parse_args()

    models = [m for m in DISPLAY_ORDER
              if (RESULTS / f"{m}_{args.preset}.json").exists()]
    if not models:
        sys.exit(f"no results for preset {args.preset}")
    OUT_MAIN.parent.mkdir(exist_ok=True)

    write_texture_table(models, args.preset)
    write_main_table(models)
    if args.preset == "classic":
        write_ablation_table()
        write_control_table()
        write_iisr_table()
        write_sensitivity_table()


if __name__ == "__main__":
    main()

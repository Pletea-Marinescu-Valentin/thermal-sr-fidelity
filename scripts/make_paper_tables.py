r"""Emit the paper's result tables straight from results/*.json.

Transcribing numbers by hand is how a table drifts from the data it claims to
report, and it double-rounds: 0.474475 becomes 0.4745 and then 0.475. This
writes the paper/table_*.tex files the paper \input{}s, so every printed cell
is formatted exactly once from the aggregate. Captions stay at one or two
lines; what a table shows is argued in the text, not in its caption.
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

#: Format of a row that prints a total over the test split rather than a mean.
#: Invented hot regions are rare enough that a per-frame rate prints as a row
#: of 0.000, which reads as a rounding artefact; the integer count is exact.
COUNT = "count"

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
    # Unbolded: a tie at zero, and bicubic's zero holds by construction, so
    # marking a winner would rank nothing.
    ("m2_d10_n_hallucinated",
     "M2 invented regions (total) " + DOWN,                   COUNT, None),
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


def cell(aggregate, fmt, n_frames):
    """One printed cell; a COUNT row prints the mean times the frame count."""
    if fmt != COUNT:
        return fmt.format(aggregate["mean"])
    total = aggregate["mean"] * n_frames
    # A fractional total would mean the mean skipped frames, and the integer
    # printed would then not be the count it claims to be.
    if abs(total - round(total)) > 1e-6:
        sys.exit(f"count row does not total to a whole number: {total}")
    return str(round(total))


def best_cells(values, direction, fmt):
    """Indices of the best cells, by the row's own notion of best.

    Comparison is on the *printed* value, not the underlying float: two cells
    that a reader sees as identical must be bolded identically, otherwise the
    table appears to rank 0.9953 above 0.9953.
    """
    if direction is None:
        # Some rows have no single best cell because they are only meaningful
        # in pairs: M7's amplitude and correspondence trade off against each
        # other, and the blurriest reconstruction can hold the highest
        # correspondence precisely because it adds nothing. Bolding one of them
        # would assert a ranking the metric does not support.
        return set()
    shown = [float(fmt.format(v)) for v in values]
    if direction == "high":
        target = max(shown)
    elif direction == "low":
        target = min(shown)
    else:
        # "one" for ratios, "zero" for differences such as M6's delta. Both are
        # two-sided: overshoot and undershoot are equally wrong, which is the
        # whole point of those metrics and must not be bolded as if larger were
        # better.
        ideal = 0.0 if direction == "zero" else 1.0
        target = min(shown, key=lambda v: abs(v - ideal))
    return {i for i, v in enumerate(shown) if v == target}


def write_main_table(models):
    aggs = {p: load(p, models) for p in ("classic", "realistic")}
    meta = json.loads((RESULTS / f"{models[0]}_classic.json").read_text())
    n, head = len(models), " & ".join(SHORT.get(m, m) for m in models)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Results at $\times 4$ on the FLIR ADAS v2 test split "
        f"({meta['n_frames']} frames, {meta['n_videos']} videos). Best value per "
        r"row and degradation in bold; $\rightarrow 1$ marks a two-sided ratio.}",
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
            shown = [cell(aggs[preset][m][key], fmt, meta["n_frames"])
                     for m in models]
            blocks.append([(r"\textbf{" + s + "}") if i in best else s
                           for i, s in enumerate(shown)])
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
        r"\caption{Texture statistics in the flat cold set $F$ on FLIR ADAS v2 "
        f"({preset} degradation, {meta['n_frames']} frames); the reference "
        + f"reads $\\hat{{H}} = {ref_h:.3f}$ and $D = {ref_d:.3f}$.}}",
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
        r"\caption{Texture metrics on constructed references with a known "
        f"outcome ({d['n_frames']} FLIR ADAS v2 test frames).}}",
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

#: (results prefix, block heading) of each track the ablation was run on; ""
#: is ADAS and "f3_" FLAME 3, matching how every other artefact is named.
ABLATION_TRACKS = [("", "FLIR ADAS v2"), ("f3_", "FLAME~3")]


def write_ablation_table():
    """Texture-loss sweep on both datasets, stacked in one table.

    Arms not yet evaluated are skipped, and a track with fewer than two arms
    on disk is left out rather than printed half empty.
    """
    tracks = []
    for prefix, heading in ABLATION_TRACKS:
        cols = [(prefix + stem, label) for stem, label in ABLATION_COLUMNS
                if (RESULTS / f"{prefix}{stem}.json").exists()]
        if len(cols) < 2:
            print(f"ablation ({heading}): not enough arms on disk, skipped")
            continue
        tracks.append((heading, cols))
    if not tracks:
        return
    # One header serves both blocks, so both must carry the same arms.
    labels = [label for _, label in tracks[0][1]]
    if any([label for _, label in cols] != labels for _, cols in tracks):
        sys.exit("ablation tracks carry different arms; one header cannot serve both")
    n = len(labels)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Texture objective \eqref{eq:ltex} added to EDSR at two "
        r"weights, against $L_1$ training and ESRGAN (classic degradation).}",
        r"\label{tab:ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r"& \multicolumn{" + str(n - 1) + r"}{c}{EDSR} & \\",
        r"\cmidrule{2-" + str(n) + "}",
        "Metric & " + " & ".join(labels) + r" \\",
    ]
    for heading, cols in tracks:
        aggs = {stem: json.loads((RESULTS / f"{stem}.json").read_text())["aggregate"]
                for stem, _ in cols}
        lines += [r"\midrule",
                  r"\multicolumn{" + str(n + 1) + r"}{l}{\emph{" + heading + r"}} \\"]
        for key, label, fmt in ABLATION_ROWS:
            cells = [fmt.format(aggs[stem][key]["mean"]) for stem, _ in cols]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        print(f"  -- {heading}")
        for key, label, fmt in ABLATION_ROWS:
            vals = "  ".join(f"{fmt.format(aggs[s][key]['mean']):>9}" for s, _ in cols)
            print(f"  {label[:26]:28s}{vals}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_ABLATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_ABLATION}")


OUT_FLAME3 = Path("paper/table_flame3.tex")

#: The FLAME 3 table carries the metrics the paper's claims rest on, not every
#: metric. The full sweep lives in results/f3_comparison_*.md.
F3_ROWS = [
    ("std_lpips",          "LPIPS " + DOWN,                   "{:.4f}", "low"),
    (None, None, None, None),
    ("m0_rmse_k",          "M0 RMSE (K) " + DOWN,             "{:.3f}", "low"),
    ("m1_d10_mask_iou",    "M1 IoU@10K " + UP,                "{:.3f}", "high"),
    ("m2_d10_n_hallucinated",
     "M2 invented regions (total) " + DOWN,                   COUNT, None),
    ("m2_d10_miss_rate",   "M2 miss rate " + DOWN,            "{:.3f}", "low"),
    ("m3_spearman_rho",    r"M3 ordering $\rho_s$ " + UP,     "{:.3f}", "high"),
    ("m4_texture_ratio",   "M4 texture ratio " + TO1,         "{:.3f}", "one"),
    ("m5_gradient_ratio",  "M5 gradient ratio " + TO1,        "{:.3f}", "one"),
    (None, None, None, None),
    ("m6_flat_delta_hurst", r"M6 $\Delta\hat{H}$ $\rightarrow 0$", "{:+.3f}", None),
    ("m7_flat_hf_amp_ratio", r"M7 amplitude $a$ " + TO1,      "{:.3f}", None),
    ("m7_flat_hf_corr",    r"M7 correspondence $r$ " + UP,    "{:.3f}", None),
]


def load_f3(preset, models):
    return {m: json.loads(
        (RESULTS / f"f3_{m}_{preset}.json").read_text())["aggregate"]
        for m in models}


def write_flame3_table(models):
    aggs = {p: load_f3(p, models) for p in ("classic", "realistic")}
    meta = json.loads((RESULTS / f"f3_{models[0]}_classic.json").read_text())
    n, head = len(models), " & ".join(SHORT.get(m, m) for m in models)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Results at $\times 4$ on FLAME~3 "
        f"({meta['n_frames']} test frames; trained on the XT709 camera, tested "
        r"on the M30T camera). Best value per row and degradation in bold; the "
        r"texture block is unbolded, as $a$ and $r$ rank only jointly.}",
        r"\label{tab:flame3}",
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
    for key, label, fmt, direction in F3_ROWS:
        if key is None:
            lines.append(r"\midrule")
            continue
        blocks = []
        for preset in ("classic", "realistic"):
            vals = [aggs[preset][m][key]["mean"] for m in models]
            best = best_cells(vals, direction, fmt)
            shown = [cell(aggs[preset][m][key], fmt, meta["n_frames"])
                     for m in models]
            blocks.append([(r"\textbf{" + s + "}") if i in best else s
                           for i, s in enumerate(shown)])
        lines.append(f"{label} & " + " & ".join(blocks[0]) + " & & "
                     + " & ".join(blocks[1]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    OUT_FLAME3.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_FLAME3}")

    print("\n--- FLAME 3, for the prose (classic) ---")
    a = aggs["classic"]
    for m in models:
        d = a[m]
        print(f"  {SHORT.get(m, m):8s} "
              f"LPIPS={d['std_lpips']['mean']:.4f}  "
              f"M0={d['m0_rmse_k']['mean']:.3f}  "
              f"M4={d['m4_texture_ratio']['mean']:.3f}  "
              f"dH={d['m6_flat_delta_hurst']['mean']:+.3f} "
              f"[{d['m6_flat_delta_hurst']['lo']:+.3f}, "
              f"{d['m6_flat_delta_hurst']['hi']:+.3f}]  "
              f"a={d['m7_flat_hf_amp_ratio']['mean']:.3f}  "
              f"r={d['m7_flat_hf_corr']['mean']:.3f} "
              f"[{d['m7_flat_hf_corr']['lo']:.3f}, {d['m7_flat_hf_corr']['hi']:.3f}]")
    ref = a[models[0]]
    print(f"  reference H={ref['m6_flat_hurst_gt']['mean']:.3f}  "
          f"sigma_w3={ref['m6_flat_ladder_gt_k']['mean'] if 'm6_flat_ladder_gt_k' in ref else float('nan')}")


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
        f3 = [m for m in DISPLAY_ORDER
              if (RESULTS / f"f3_{m}_classic.json").exists()
              and (RESULTS / f"f3_{m}_realistic.json").exists()]
        if len(f3) > 1:
            write_flame3_table(f3)
        else:
            print("skipping FLAME 3 table: need both presets for >1 model")


if __name__ == "__main__":
    main()

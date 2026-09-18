"""Check numbers quoted in the paper's prose against results/*.json.

Tables are generated (`make_paper_tables.py`), so they cannot drift. Prose is
written by hand, and an earlier revision of this paper carried seven cells that
had been double-rounded in transcription. Every quantity stated in the text
should therefore be asserted against the file it came from.

Each claim below pairs the value as printed in the paper with the expression
that produces it. A mismatch is either a stale number in the paper or a changed
result; both need a human, so this exits non-zero and names the claim.

Run: .venv/Scripts/python.exe scripts/check_paper_numbers.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.flame3 import bootstrap_cluster  # noqa: E402
from tsrf.stats import (  # noqa: E402
    holm_bonferroni, paired_effect, paired_test,
)

RESULTS = Path("results")
_cache = {}


def agg(dataset, preset, model, key, field="mean"):
    """One aggregate field. `dataset` is "" for ADAS or "f3_" for FLAME 3."""
    path = RESULTS / f"{dataset}{model}_{preset}.json"
    if path not in _cache:
        _cache[path] = json.loads(path.read_text())["aggregate"]
    return _cache[path][key][field]


def meta(dataset, preset, model, key):
    return json.loads((RESULTS / f"{dataset}{model}_{preset}.json").read_text())[key]


def controlled_p(keys, preset="classic"):
    """Holm-adjusted p-values for ESRGAN against RRDB on FLAME 3."""
    a = json.loads((RESULTS / f"f3_esrgan_{preset}_per_frame.json").read_text())
    b = json.loads((RESULTS / f"f3_rrdb_{preset}_per_frame.json").read_text())
    clusters = [bootstrap_cluster(n) for n in a["names"]]
    raw = [paired_test([r.get(k) for r in a["records"]],
                       [r.get(k) for r in b["records"]],
                       clusters=clusters)["p_value"] for k in keys]
    return holm_bonferroni(raw)[1]


def controlled_effects(preset="classic"):
    """Paired effect sizes for ESRGAN against RRDB on FLAME 3."""
    a = json.loads((RESULTS / f"f3_esrgan_{preset}_per_frame.json").read_text())
    b = json.loads((RESULTS / f"f3_rrdb_{preset}_per_frame.json").read_text())
    clusters = [bootstrap_cluster(n) for n in a["names"]]
    return {k: paired_effect([r.get(k) for r in a["records"]],
                             [r.get(k) for r in b["records"]],
                             clusters=clusters) for k in CONTROLLED_KEYS}


def abl(stem, key):
    """One aggregate field of an ablation arm, keyed by its run stem."""
    return json.loads((RESULTS / f"{stem}.json").read_text())["aggregate"][key]["mean"]


def _m2():
    return json.loads((RESULTS / "m2_sensitivity.json").read_text())


def m2(model, axis, value):
    """Hallucinated regions per frame at one point of the threshold sweep."""
    return _m2()["hallucinated_per_frame"][f"{model}|{axis}|{value}"]


def m2_worst(exclude):
    """The largest rate over every axis but one.

    The paper claims three of the four thresholds leave the negative result
    untouched and names the worst case; this is that worst case, so the claim
    cannot quietly become false if a sweep point changes.
    """
    return max(v for k, v in _m2()["hallucinated_per_frame"].items()
               if k.split("|")[1] != exclude)


def ceiling(dataset, field, radius=2):
    """The attainable-correspondence ceiling, from scripts/m7_ceiling.py."""
    suffix = "" if radius == 2 else f"_r{radius}"
    path = RESULTS / f"{dataset}m7_ceiling{suffix}.json"
    return json.loads(path.read_text())[f"linear_ceiling_{field}"]


def first_n(dataset, model, key, n=200):
    """A model's mean over the first n test frames.

    The ceiling is fitted and scored on those frames, so anything compared
    against it has to be measured on them too; the full-split mean is a
    different quantity and mixing the two would not be a comparison.
    """
    path = RESULTS / f"{dataset}{model}_classic_per_frame.json"
    records = json.loads(path.read_text())["records"][:n]
    return float(np.nanmean([r.get(key, np.nan) for r in records]))


def control(dataset, arm, key):
    d = json.loads((RESULTS / f"{dataset}positive_control.json").read_text())
    v = d["aggregate"][arm][key]
    return v["mean"] if isinstance(v, dict) else v


CONTROLLED_KEYS = ["std_lpips", "m0_rmse_k", "m1_d10_mask_iou",
                   "m4_texture_ratio", "m6_flat_delta_hurst",
                   "m7_flat_hf_amp_ratio", "m7_flat_hf_corr"]


def claims():
    """(description, printed value, computed value, absolute tolerance)."""
    eff = controlled_effects()
    rows = [
        # -- Setup: the reference the metrics measure against
        ("FLAME 3 reference texture in F (K)", 0.324,
         agg("f3_", "classic", "bicubic", "m4_gt_local_std_k"), 5e-4),
        ("ADAS reference texture in F (K)", 0.155,
         agg("", "classic", "bicubic", "m4_gt_local_std_k"), 5e-4),
        ("FLAME 3 reference exponent", 0.618,
         agg("f3_", "classic", "bicubic", "m6_flat_hurst_gt"), 5e-4),
        ("FLAME 3 reference exponent, CI low", 0.549,
         agg("f3_", "classic", "bicubic", "m6_flat_hurst_gt", "lo"), 5e-3),
        ("FLAME 3 reference exponent, CI high", 0.692,
         agg("f3_", "classic", "bicubic", "m6_flat_hurst_gt", "hi"), 5e-3),
        ("ADAS reference exponent", 0.475,
         agg("", "classic", "bicubic", "m6_flat_hurst_gt"), 5e-4),

        # -- Results: the headline conflict on FLAME 3
        ("FLAME 3 ESRGAN LPIPS", 0.0318,
         agg("f3_", "classic", "esrgan", "std_lpips"), 5e-5),
        ("FLAME 3 RRDB LPIPS", 0.0463,
         agg("f3_", "classic", "rrdb", "std_lpips"), 5e-5),
        ("FLAME 3 ESRGAN M0 (K)", 4.256,
         agg("f3_", "classic", "esrgan", "m0_rmse_k"), 5e-4),
        ("FLAME 3 RRDB M0 (K)", 3.733,
         agg("f3_", "classic", "rrdb", "m0_rmse_k"), 5e-4),

        # -- Results: the controlled comparison, seven paired differences.
        # These are medians over units, matched to the bootstrap interval the
        # paper prints beside them, not means of the aggregates.
        ("controlled LPIPS gain", 0.011, -eff["std_lpips"]["estimate"], 5e-4),
        ("controlled LPIPS gain, CI low", 0.010, -eff["std_lpips"]["hi"], 5e-4),
        ("controlled LPIPS gain, CI high", 0.012, -eff["std_lpips"]["lo"], 5e-4),
        ("controlled M0 cost (K)", 0.506, eff["m0_rmse_k"]["estimate"], 5e-4),
        ("controlled M1 IoU cost", 0.025, -eff["m1_d10_mask_iou"]["estimate"], 5e-4),
        ("controlled M4 rise", 0.147, eff["m4_texture_ratio"]["estimate"], 5e-4),
        ("controlled M6 move", 0.288, -eff["m6_flat_delta_hurst"]["estimate"], 5e-4),
        ("controlled M7 amplitude rise", 0.179,
         eff["m7_flat_hf_amp_ratio"]["estimate"], 5e-4),
        ("controlled M7 correspondence fall", 0.065,
         -eff["m7_flat_hf_corr"]["estimate"], 5e-4),
        ("controlled M7 correspondence, CI low", 0.063,
         -eff["m7_flat_hf_corr"]["hi"], 5e-4),
        ("controlled M7 correspondence, CI high", 0.068,
         -eff["m7_flat_hf_corr"]["lo"], 5e-4),
        ("M7 correspondence sign consistency", 0.95,
         eff["m7_flat_hf_corr"]["sign_consistency"], 5e-3),
        ("lowest sign consistency of the seven", 0.89,
         min(e["sign_consistency"] for e in eff.values()), 5e-3),
        ("highest sign consistency of the seven", 1.00,
         max(e["sign_consistency"] for e in eff.values()), 1e-9),

        # -- Results: texture block
        ("FLAME 3 ESRGAN dHurst", 0.036,
         agg("f3_", "classic", "esrgan", "m6_flat_delta_hurst"), 5e-4),
        ("FLAME 3 ESRGAN dHurst CI low", -0.041,
         agg("f3_", "classic", "esrgan", "m6_flat_delta_hurst", "lo"), 5e-3),
        ("FLAME 3 ESRGAN dHurst CI high", 0.111,
         agg("f3_", "classic", "esrgan", "m6_flat_delta_hurst", "hi"), 5e-3),
        ("FLAME 3 EDSR dHurst (lowest regressor)", 0.140,
         agg("f3_", "classic", "edsr_lite", "m6_flat_delta_hurst"), 5e-4),
        ("FLAME 3 bicubic dHurst (highest)", 0.491,
         agg("f3_", "classic", "bicubic", "m6_flat_delta_hurst"), 5e-4),
        ("FLAME 3 ESRGAN correspondence", 0.128,
         agg("f3_", "classic", "esrgan", "m7_flat_hf_corr"), 5e-4),
        ("FLAME 3 ESRGAN correspondence CI low", 0.122,
         agg("f3_", "classic", "esrgan", "m7_flat_hf_corr", "lo"), 5e-3),
        ("FLAME 3 ESRGAN correspondence CI high", 0.133,
         agg("f3_", "classic", "esrgan", "m7_flat_hf_corr", "hi"), 5e-3),

        # -- Results: the two stated differences from ADAS
        ("ADAS ESRGAN M4", 1.113,
         agg("", "classic", "esrgan", "m4_texture_ratio"), 5e-4),
        ("ADAS ESRGAN M7 amplitude", 1.141,
         agg("", "classic", "esrgan", "m7_flat_hf_amp_ratio"), 5e-4),
        ("FLAME 3 ESRGAN M4", 0.682,
         agg("f3_", "classic", "esrgan", "m4_texture_ratio"), 5e-4),
        ("FLAME 3 ESRGAN M7 amplitude", 0.625,
         agg("f3_", "classic", "esrgan", "m7_flat_hf_amp_ratio"), 5e-4),
        ("FLAME 3 bicubic correspondence", 0.221,
         agg("f3_", "classic", "bicubic", "m7_flat_hf_corr"), 5e-4),
        ("FLAME 3 bicubic M7 amplitude", 0.279,
         agg("f3_", "classic", "bicubic", "m7_flat_hf_amp_ratio"), 5e-4),
        ("ADAS bicubic correspondence", 0.143,
         agg("", "classic", "bicubic", "m7_flat_hf_corr"), 5e-4),
        ("ADAS regressor correspondence", 0.173,
         agg("", "classic", "rrdb", "m7_flat_hf_corr"), 5e-4),

        # -- Results: hot regions on FLAME 3
        ("FLAME 3 RRDB hallucinated / frame", 0.0014,
         agg("f3_", "classic", "rrdb", "m2_d10_n_hallucinated"), 5e-5),
        ("FLAME 3 RRDB hallucination CI high", 0.0041,
         agg("f3_", "classic", "rrdb", "m2_d10_n_hallucinated", "hi"), 5e-4),
        ("FLAME 3 ESRGAN miss rate", 0.155,
         agg("f3_", "classic", "esrgan", "m2_d10_miss_rate"), 5e-4),
        ("FLAME 3 RRDB miss rate", 0.167,
         agg("f3_", "classic", "rrdb", "m2_d10_miss_rate"), 5e-4),

        # -- Results: positive control replicated on FLAME 3
        ("FLAME 3 control attenuated M4", 0.447,
         control("f3_", "attenuated", "m4_texture_ratio"), 5e-4),
        ("FLAME 3 control attenuated r", 0.911,
         control("f3_", "attenuated", "m7_flat_hf_corr"), 5e-4),
        ("FLAME 3 control fabricated M4", 1.082,
         control("f3_", "fabricated_local", "m4_texture_ratio"), 5e-4),
        ("FLAME 3 control fabricated r", 0.012,
         control("f3_", "fabricated_local", "m7_flat_hf_corr"), 5e-4),

        # -- Results: the texture-objective ablation on ADAS. The run named
        # edsr_tex2 carries lambda = 2e-3 and edsr_tex carries 5e-3, which is
        # the opposite of what the names suggest, so the mapping is asserted
        # here rather than trusted.
        ("ablation L1 dHurst", 0.744, abl("edsr_lite_classic", "m6_flat_delta_hurst"), 5e-4),
        ("ablation lambda=2e-3 dHurst", -0.121,
         abl("edsr_tex2_x4_classic", "m6_flat_delta_hurst"), 5e-4),
        ("ablation lambda=5e-3 dHurst", -0.142,
         abl("edsr_tex_x4_classic", "m6_flat_delta_hurst"), 5e-4),
        ("ablation L1 amplitude", 0.279,
         abl("edsr_lite_classic", "m7_flat_hf_amp_ratio"), 5e-4),
        ("ablation lambda=2e-3 amplitude", 1.811,
         abl("edsr_tex2_x4_classic", "m7_flat_hf_amp_ratio"), 5e-4),
        ("ablation lambda=5e-3 amplitude", 1.683,
         abl("edsr_tex_x4_classic", "m7_flat_hf_amp_ratio"), 5e-4),
        ("ablation L1 correspondence", 0.173,
         abl("edsr_lite_classic", "m7_flat_hf_corr"), 5e-4),
        ("ablation lambda=2e-3 correspondence", 0.030,
         abl("edsr_tex2_x4_classic", "m7_flat_hf_corr"), 5e-4),
        ("ablation lambda=5e-3 correspondence", 0.032,
         abl("edsr_tex_x4_classic", "m7_flat_hf_corr"), 5e-4),
        ("ablation L1 M0 (K)", 0.474, abl("edsr_lite_classic", "m0_rmse_k"), 5e-4),
        ("ablation lambda=5e-3 M0 (K)", 0.538,
         abl("edsr_tex_x4_classic", "m0_rmse_k"), 5e-4),

        # -- Results: robustness of the negative result on discrete hallucination
        ("M2 worst rate over the three inert axes", 0.015,
         m2_worst(exclude="lr_relax"), 5e-4),
        ("M2 EDSR at lambda = 1", 0.280, m2("edsr_lite", "lr_relax", 1.0), 5e-4),
        ("M2 ESRGAN at lambda = 1", 0.275, m2("esrgan", "lr_relax", 1.0), 5e-4),

        # -- Results: the attainable-correspondence ceiling
        ("ADAS linear ceiling r", 0.173, ceiling("", "corr"), 5e-4),
        ("ADAS linear ceiling amplitude", 0.196, ceiling("", "amp"), 5e-4),
        ("FLAME 3 linear ceiling r", 0.197, ceiling("f3_", "corr"), 5e-4),
        ("FLAME 3 linear ceiling amplitude", 0.212, ceiling("f3_", "amp"), 5e-4),
        ("ceiling shift from 3x3 to 5x5 support", 0.007,
         ceiling("", "corr") - ceiling("", "corr", radius=1), 5e-4),
        # The comparisons are over the same 200 frames the ceiling is fitted on,
        # not the full split, or the numbers would not be commensurable.
        ("ADAS regressors on the ceiling frames", 0.201,
         first_n("", "rrdb", "m7_flat_hf_corr"), 5e-4),
        ("ADAS ESRGAN amplitude on those frames", 1.077,
         first_n("", "esrgan", "m7_flat_hf_amp_ratio"), 5e-4),
        ("ADAS ESRGAN amplitude over ceiling", 5.5,
         first_n("", "esrgan", "m7_flat_hf_amp_ratio") / ceiling("", "amp"), 5e-2),
        ("FLAME 3 bicubic on the ceiling frames", 0.195,
         first_n("f3_", "bicubic", "m7_flat_hf_corr"), 5e-4),
        ("FLAME 3 ESRGAN on the ceiling frames", 0.109,
         first_n("f3_", "esrgan", "m7_flat_hf_corr"), 5e-4),

        # -- Setup: sample sizes
        ("ADAS test frames", 1144, meta("", "classic", "bicubic", "n_frames"), 0),
        ("FLAME 3 test frames", 738,
         meta("f3_", "classic", "bicubic", "n_frames"), 0),
        ("total test frames across datasets", 1882,
         meta("", "classic", "bicubic", "n_frames")
         + meta("f3_", "classic", "bicubic", "n_frames"), 0),
        ("FLAME 3 clusters", 738,
         meta("f3_", "classic", "bicubic", "n_clusters"), 0),
    ]
    return rows


def main():
    rows = claims()
    bad = []
    for desc, printed, computed, tol in rows:
        ok = abs(float(printed) - float(computed)) <= tol
        if not ok:
            bad.append((desc, printed, computed))
        print(f"  {'ok ' if ok else 'BAD'}  {desc:<42} paper {printed!s:>9}"
              f"   results {computed:.6g}")

    # The paper states a bound, not a value: below 1e-4 the signed-rank
    # approximation is reporting the shape of a Gaussian tail.
    worst = float(np.max(controlled_p(CONTROLLED_KEYS)))
    ok = worst < 1e-4
    print(f"  {'ok ' if ok else 'BAD'}  {'controlled p_adj all < 1e-4':<42} "
          f"largest {worst:.3g}")
    if not ok:
        bad.append(("controlled p_adj bound", "< 1e-4", worst))

    if bad:
        print(f"\n{len(bad)} claim(s) do not match the results:")
        for desc, printed, computed in bad:
            print(f"  {desc}: paper says {printed}, results give {computed}")
        return 1
    print(f"\nall {len(rows) + 1} claims match the results")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Thermal fidelity evaluation for thermal image super-resolution

Code for *Radiometric Fidelity in Thermal Image Super-Resolution: An Evaluation
Protocol for Measurement Preservation* (V. Pletea-Marinescu, M. Ciuciulete,
G. Stamatescu; under review).

Thermal super-resolution is normally evaluated with PSNR, SSIM and LPIPS, none of
which measures whether a reconstruction preserves the physical quantity a thermal
image encodes. This repository implements a deterministic protocol of eight fidelity
metrics defined on radiometric data and expressed in kelvin, together with the
training and evaluation pipeline used to compare five reconstruction methods on two
independent radiometric datasets recorded by three cameras: 1144 automotive frames
from a FLIR Tau 2, and 738 airborne wildfire frames from a DJI M30T reconstructed by
models trained on an Autel XT709.

## Quick start

The metric suite needs only NumPy, SciPy, OpenCV and Pillow:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .          # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m pip install pytest
.venv/Scripts/python.exe -m pytest -q                  # tests needing PyTorch are skipped
```

To score your own reconstruction, pass temperatures in kelvin on the
high-resolution grid. No model and no dataset from this study is needed:

```python
from tsrf.metrics import (
    radiometric_error, hotspot_preservation, hallucination_metrics,
    thermal_ordering, cold_region_smoothness, gradient_fidelity,
    texture_scaling, texture_correspondence,
)

# t_gt: radiometric reference, t_sr: its reconstruction, t_lr_up: the
# low-resolution input bicubically upsampled to the same shape (float arrays, K)
m0 = radiometric_error(t_sr, t_gt)                          # rmse_k, mae_k, bias_k
m1 = hotspot_preservation(t_sr, t_gt, delta_k=10.0)         # mask_iou, peak_error_on_gt_support_k
m2 = hallucination_metrics(t_sr, t_gt, t_lr_up, delta_k=10.0)  # n_hallucinated, miss_rate
m3 = thermal_ordering(t_sr, t_gt)                           # spearman_rho, top1_preserved
m4 = cold_region_smoothness(t_sr, t_gt)                     # texture_ratio
m5 = gradient_fidelity(t_sr, t_gt)                          # gradient_ratio, orientation_cosine
m6 = texture_scaling(t_sr, t_gt)                            # flat_delta_hurst
m7 = texture_correspondence(t_sr, t_gt)                     # flat_hf_amp_ratio (a), flat_hf_corr (r)
```

`src/tsrf/eval/runner.py` (`evaluate_frame`) shows the full per-frame protocol,
including PSNR, SSIM and LPIPS, as used for the paper.

## Where each result in the paper comes from

Every table is generated from `results/*.json` by a script, never typed by hand,
and every number quoted in the text is asserted against the same files by
`scripts/check_paper_numbers.py`.

| Paper | Produced by | Read from |
|---|---|---|
| Table I, FLIR ADAS v2 | `eval_all.py --preset classic` and `--preset realistic`, then `make_paper_tables.py` | `results/{model}_{preset}.json` |
| Table II, texture statistics | `make_paper_tables.py` | `results/{model}_classic.json` |
| Table III, constructed references | `positive_control.py`, `make_paper_tables.py` | `results/positive_control.json` |
| Table IV, FLAME 3 | `pipeline_flame3.sh`, `eval_flame3.py`, `make_paper_tables.py` | `results/f3_{model}_{preset}.json` |
| Table V, controlled pair | `make_effects_table.py` | per-frame records (see below) |
| Table VI, texture-objective ablation | `train.py --texture-weight`, `eval_ablation.py`, `make_paper_tables.py` | `results/{,f3_}edsr_tex*_x4_classic.json` |
| Fig. 1, protocol | `make_protocol_figure.py` | none |
| Fig. 2, a flat region close up | `make_texture_figure.py` | an ADAS test frame and the checkpoints |
| Fig. 3, ADAS texture | `make_scaling_figure.py` | `results/adas_figure_ladders.json`, `m7_ceiling.json` |
| Fig. 4, FLAME 3 replication | `make_flame3_figure.py` | `results/f3_figure_ladders.json`, `f3_m7_ceiling.json`, `f3_positive_control.json` |
| Sec. V-B, threshold and M2 sweeps | `make_robustness.py`, `m2_sensitivity.py` | `results/robustness.json`, `m2_sensitivity.json` |
| Sec. V-G, design choices, attainable correspondence, FLIR-IISR | `make_sensitivity.py`, `m7_ceiling.py`, `eval_iisr.py` | `results/sensitivity.json`, `m7_ceiling*.json`, `iisr_*.json` |

All scripts are in `scripts/`. The aggregated results are committed, so every
table except Table V can be regenerated from this repository alone. Table V, the
paired statistics and `check_paper_numbers.py` also need the per-frame records
(`results/*_per_frame.json`, 90 MB), which the evaluation scripts write and which
are kept out of git for their size.

## Metric suite (`src/tsrf/metrics/`)

| | metric | unit |
|---|---|---|
| M0 | radiometric error (RMSE / MAE / bias) | K |
| M1 | hot-region preservation (IoU, count error, peak position and temperature error) | px, K |
| M2 | hallucinated hot regions, and missed ones | per frame |
| M3 | thermal ordering (does the hottest region stay hottest) | rho |
| M4 | texture fabrication in flat cold regions | ratio |
| M5 | thermal-boundary gradient fidelity | ratio |
| M6 | texture scaling: variance-scaling (Hurst) exponent and box dimension of the emitted texture | — |
| M7 | texture correspondence: does the emitted texture match the measurement, or only its statistics | r, ratio |

PSNR, SSIM and LPIPS are reported alongside for contrast.

M4, M6 and M7 are three views of the same pixels and are meant to be read
together: M4 gives the amount of fine variation a model emits, M6 how that
variation scales with distance, and M7 whether it corresponds to what the
sensor recorded. A model can match the first two while failing the third, which
is what an adversarial model does here.

M7 is a pair, `(a, r)`, and neither coordinate ranks reconstructions on its own:
the blurriest one can hold the highest correspondence precisely because it adds
nothing. The pair is also not aimed at `(1, 1)`. A x4 decimation removes most of
the band this window measures, so the reachable target is the ceiling measured by
`scripts/m7_ceiling.py`: `(0.196, 0.173)` on ADAS and `(0.212, 0.197)` on FLAME 3.
Read against that point, an amplitude near 1 is not a model approaching fidelity
but one emitting several times the fine structure its input can account for.

## Protocol validation

The metrics are checked against references whose answer is known by
construction, rather than only applied to model outputs:

```bash
python scripts/positive_control.py     # phase-randomised surrogates
python scripts/make_robustness.py      # threshold sweep, per-frame separability
python scripts/make_sensitivity.py     # M6/M7 against their own design choices
python scripts/m2_sensitivity.py       # M2 against its four thresholds, not just delta
python scripts/m7_ceiling.py           # how much correspondence is attainable at all
```

`positive_control.py` builds predictions that keep the reference's
low-frequency content and either attenuate the measured texture or replace it
with a phase-randomised surrogate carrying no measured signal at all. The
surrogate preserves the power spectrum exactly, so M4 and M6 rank it as the
more faithful reconstruction and only M7 rejects it; the trained models land on
these controls. `make_sensitivity.py` sweeps the window ladder and the flat-set
definition, and resolves M7 by scale to locate the point below which no method
recovers the measurement.

`m2_sensitivity.py` answers whether the negative result on discrete hallucination
is a property of the models or of M2's thresholds: three of the four leave it at
zero, and the fourth only raises it by mislabelling attenuated real regions as
invented. `m7_ceiling.py` fits the optimal shift-invariant linear predictor of the
reference fine texture from the same input, which is what turns M7 from a purely
comparative score into one with a calibrated reference point, as described above.
It is a ceiling over shift-invariant linear filters only, so a spatially adaptive
model can exceed it, and the L1 regressors do; that is the useful reading, since
falling well below it is then not explained by the information gap.

Outputs are `results/positive_control.json`, `robustness.json`, `sensitivity.json`,
`m2_sensitivity.json` and `m7_ceiling.json`, plus the `f3_` counterparts of the
last two checks, which run on both datasets.

## Data

None of the three datasets is redistributed here.

- **FLIR ADAS v2** (primary, radiometric). Obtain from Teledyne FLIR
  (<https://oem.flir.com/solutions/automotive/adas-dataset-form/>) under its image
  licence agreement, which permits research use and publication but not
  redistribution. The 16-bit T-linear TIFFs in `analyticsData/` are the radiometric
  measurements this study uses; apparent temperature is `raw * 0.04` K.
- **FLAME 3** (second radiometric dataset). Airborne wildfire imagery from two
  cameras. The NADIR plots (Autel XT709) are the training and validation split and
  ship integer rasters whose scale is recovered empirically, `counts/10 - 275.0` in
  Celsius; the CV subset (DJI M30T) is the test split and ships float32 Celsius
  directly. Testing therefore crosses instruments by construction. Obtain from
  IEEE DataPort, <https://doi.org/10.21227/w0mz-aq48>.
- **FLIR-IISR** (secondary, real cross-sensor degradation, 8-bit), from
  <https://github.com/JZD151/Real-IISR>. Used only for the stress test in
  Section V-G; no claims in kelvin are made from it.

## Statistics (`src/tsrf/stats/`)

**The sampling unit differs between the datasets**, and it sets how the bootstrap
resamples and how the paired tests pair. ADAS frames come from continuous video, so
the video is the unit, 17 in the test split. The FLAME 3 CV subset is not video but a
sparse set of frame pairs, empirically independent, so the frame is the unit, 738 of
them; `scripts/check_flame3_independence.py` reports the three measurements behind
that, and `tsrf.data.flame3.bootstrap_cluster` is the single place the rule lives.
This is not a cosmetic choice: clustering FLAME 3 on its fire / no-fire partition
leaves two units, and a paired test over two units cannot return p below 0.5
whatever the data contain, so every comparison came back at p = 1.00 for effects
that are in fact large and consistent.

**Effect sizes carry the claims, not p-values.** Beyond about 25 pairs the
signed-rank test switches to a normal approximation, and on the 738-unit split that
produces nominal values near 1e-112, a 22-sigma extrapolation rather than a
measurement. `paired_effect` therefore reports the paired difference, a bootstrap
interval for that same statistic, and `sign_consistency`, the share of units on
which the difference keeps its sign. The last is what distinguishes a shift present
on almost every frame from an identical average produced by a handful of outliers,
which no p-value can. Significance is reported as a bound, `p_adj < 1e-4`.

**Undefined tests stay undefined.** A comparison between two arms that differ on no
unit at all has no p-value; `holm_bonferroni` keeps such entries as NaN and excludes
them from the multiplicity count, rather than ranking them among the tests that were
performed, where the step-down loop would hand them the running maximum.

## Environment

The results in the paper were produced with Python 3.13.12 on Windows 11 and a
single NVIDIA RTX 5050 GPU. `requirements.txt` pins the exact versions used:

```bash
python -m venv .venv
# PyTorch first, from the CUDA 12.8 index (required for Blackwell, sm_120):
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
```

`pip install -e .[full]` installs the same packages without pinning them.

## Reproducing the results

Once the FLIR ADAS archive is available locally:

```bash
python scripts/extract_adas_thermal.py <archive.zip>   # thermal split only, ~5 GB
python scripts/validate_radiometric_scale.py           # checks the 0.04 K/count gain
python scripts/build_splits.py                         # video-disjoint splits + scaler
python scripts/pack_dataset.py                         # memory-mapped LR/HR arrays
bash   scripts/pipeline.sh                             # train, evaluate, figures, report
```

`pipeline.sh` is resume-safe: re-running it continues from the last checkpoint.
It covers training, the two radiometric evaluations, the qualitative figures
and the report; the cross-sensor track, the protocol validation above and the
paper artefacts are run separately. Individual stages:

```bash
python scripts/verify_datasets.py               # archives present and readable
python scripts/train.py edsr_lite --steps 100000 --resume
python scripts/train_gan.py --pretrained runs/rrdb_x4_classic/best.pt \
    --steps 50000 --batch 8 --patch 32 --resume   # ESRGAN, from the converged RRDB
python scripts/eval_all.py --preset classic     # + paired significance tests
python scripts/eval_iisr.py --model esrgan      # cross-sensor stress test
python scripts/make_figures.py                  # qualitative overlays
python scripts/make_texture_figure.py           # Fig. 2, a flat region close up
python scripts/make_scaling_figure.py           # M6/M7 ladder and plane
python scripts/positive_control.py              # constructed references (protocol check)
python scripts/make_robustness.py               # threshold sweep + per-frame separability
python scripts/make_sensitivity.py              # M6/M7 vs their design choices
python scripts/eval_ablation.py --run runs/edsr_tex_x4_classic   # texture-loss arm
python scripts/m7_ceiling.py                    # attainable correspondence
python scripts/m2_sensitivity.py                # M2 against its own thresholds
python scripts/make_paper_tables.py             # LaTeX tables from results/*.json
python scripts/make_report.py                   # assembles results/REPORT.md
python scripts/check_tex_escapes.py             # guards against eaten LaTeX backslashes
python scripts/check_paper_numbers.py           # every number quoted in the prose
python scripts/check_cross_refs.py              # no unchecked manual subsection letters
python scripts/check_build_warnings.py          # overfull boxes, undefined refs
python scripts/make_effects_table.py            # paired effect sizes, both datasets
python scripts/make_flame3_figure.py            # the replication figure
python scripts/make_protocol_figure.py          # Fig. 1, the protocol end to end
python scripts/make_overleaf_zip.py             # bundles paper/ into overleaf.zip
```

For the second dataset:

```bash
python scripts/extract_flame3.py CVSubset.zip NADIRPlots.zip
python scripts/validate_flame3_scale.py         # recovers the XT709 offset, 275.0
python scripts/check_flame3_independence.py     # is the CV subset a video? no
python scripts/pack_flame3.py
bash   scripts/pipeline_flame3.sh               # train and evaluate the five arms
python scripts/eval_flame3.py --preset classic  # re-aggregates from cached records
python scripts/positive_control.py --dataset flame3
```

`eval_flame3.py` reuses cached per-frame records and rederives the clusters from
the frame paths, so a correction to the sampling-unit rule re-scores every model
without re-running any of them.

The paper-facing scripts write into `paper/`, which is not part of this repository
(see below), so they are useful only alongside a local copy of the manuscript.

Splits are fixed and committed (`configs/splits.json`, `configs/scaler.json`), so
evaluation runs on exactly the frames reported in the paper.

Four checks guard the manuscript against drift, because the failure modes are
silent rather than loud. `check_tex_escapes.py` catches LaTeX macros whose
backslash was eaten by a Python escape, which renders as plain text without
failing the build. `check_paper_numbers.py` asserts every quantity quoted in the
prose against `results/*.json`; tables are generated and cannot drift, prose is
written by hand and did. `check_cross_refs.py` guards the subsection references:
IEEEtran numbers subsections by position, so a hand-written letter shifts when a
subsection is inserted and keeps resolving, to the wrong target. The paper now
references labels instead, and the checker fails on any manual letter whose
target has not been checked and recorded.
`check_build_warnings.py` reads the LaTeX log, which latexmk exits zero on
regardless: an overfull table printing over the neighbouring column, an undefined
reference rendering as [?], a label defined twice sending every reference to
whichever won.

## What is and is not in this repository

Included: the metric suite, models, training and evaluation code, experiment
configurations, split lists, the aggregated results behind every table, and the
protocol-validation outputs (`positive_control.json`, `robustness.json`,
`sensitivity.json`, `m2_sensitivity.json`, `m7_ceiling.json`, and their `f3_`
counterparts where the check runs on both datasets).

Not included, and regenerable with the commands above: dataset files, per-frame
result records, rendered figures (derived from the source imagery), and trained
checkpoints (the largest exceeds GitHub's per-file limit). Checkpoints are available
from the authors on request. `configs/checkpoints.sha256` records the SHA-256 of
the twelve checkpoints behind the paper's numbers (`best.pt` for the regressors,
`last.pt` for ESRGAN and for the ablation arms), so a copy obtained later can be
checked with `sha256sum -c configs/checkpoints.sha256` from the repository root.

Also not included: the `paper/` directory holding the manuscript, its figures
and its generated tables. The code that produces those artefacts is here, but
the manuscript itself is kept out of the public repository.

## Training with the texture objective

The multi-scale texture loss of `src/tsrf/train/losses.py` is off by default;
every model in the main table is trained with L1 alone. The ablation fine-tunes
the selected L1 regressor at two weights, on each dataset:

```bash
# ADAS: 15000 steps, 15% of that track's 100k budget
python scripts/train.py edsr_lite --steps 15000 --lr 5e-5 --texture-weight 0.005 \
    --init-from runs/edsr_lite_x4_classic/best.pt --out runs/edsr_tex_x4_classic
python scripts/eval_ablation.py --run runs/edsr_tex_x4_classic

# FLAME 3: 7500 steps, the same fraction of a 50k budget
python scripts/train.py edsr_lite --dataset flame3 --steps 7500 --val-every 500 \
    --texture-weight 0.005 --init-from runs/edsr_lite_x4_classic_flame3/best.pt \
    --out runs/edsr_tex_x4_classic_flame3
python scripts/eval_ablation.py --dataset flame3 --run runs/edsr_tex_x4_classic_flame3
```

Two naming traps. The run `edsr_tex_*` carries the **larger** weight, 5e-3, and
`edsr_tex2_*` the smaller, 2e-3, which is the opposite of what the names suggest;
the mapping is asserted in `check_paper_numbers.py` rather than remembered. And
these arms are evaluated at `last.pt`, not `best.pt`: selecting by validation PSNR
would select against the objective under test, since the texture term buys
statistics by spending pixel accuracy.

## Tests

```bash
python -m pytest tests/ -q
```

156 tests. The metric suite is exercised on synthetic scenes with known answers,
the fractal metrics against synthetic fractional Brownian surfaces of known Hurst
exponent and against the two limits bracketing them, white detector noise (H = 0)
and a smooth bandlimited field (H = 1). The rest cover the degradation pipeline,
the models, the statistics, the FLAME 3 radiometry and sampling-unit rule, and the
three manuscript checkers.

Several tests exist because the bug they pin was real and silent:

- a NaN p-value inheriting the running maximum of the Holm correction, reporting
  an untestable comparison as `p_adj = 2e-59`;
- a bootstrap resampling the mean while the median was printed, putting the
  estimate outside its own interval on skewed differences;
- two FLAME 3 subsets numbering their frames from 00001 independently, which an
  unqualified cluster label merged into shared units;
- a texture loss that scores two independent draws from the same distribution as
  a near-perfect match, which is the property the paper's argument rests on.

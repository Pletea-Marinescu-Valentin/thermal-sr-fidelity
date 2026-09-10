# Thermal fidelity evaluation for thermal image super-resolution

Code for *Perceptual Metrics Reward Thermal Texture Fabrication: A Radiometric
Fidelity Protocol for Thermal Image Super-Resolution*.

Thermal super-resolution is normally evaluated with PSNR, SSIM and LPIPS, none of
which measures whether a reconstruction preserves the physical quantity a thermal
image encodes. This repository implements a deterministic protocol of eight fidelity
metrics defined on radiometric data and expressed in Kelvin, together with the
training and evaluation pipeline used to compare five reconstruction methods.

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

## Data

Neither dataset is redistributed here.

- **FLIR ADAS v2** (primary, radiometric). Obtain from Teledyne FLIR under its image
  licence agreement, which permits research use and publication but not
  redistribution. The 16-bit T-linear TIFFs in `analyticsData/` are the radiometric
  measurements this study uses; apparent temperature is `raw * 0.04` K.
- **FLIR-IISR** (secondary, real cross-sensor degradation, 8-bit). Used only for the
  stress test in Section V-C; no claims in Kelvin are made from it.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
# PyTorch for Blackwell (RTX 5050, sm_120):
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

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
Individual stages:

```bash
python scripts/train.py edsr_lite --steps 100000 --resume
python scripts/eval_all.py --preset classic     # + paired significance tests
python scripts/eval_iisr.py --model esrgan      # cross-sensor stress test
python scripts/make_figures.py                  # qualitative overlays
python scripts/make_report.py                   # assembles results/REPORT.md
```

Splits are fixed and committed (`configs/splits.json`, `configs/scaler.json`), so
evaluation runs on exactly the frames reported in the paper.

## What is and is not in this repository

Included: the metric suite, models, training and evaluation code, experiment
configurations, split lists, and the aggregated results behind every table.

Not included, and regenerable with the commands above: dataset files, per-frame
result records, rendered figures (derived from the source imagery), and trained
checkpoints (the largest exceeds GitHub's per-file limit). Checkpoints are available
from the authors on request.

## Tests

```bash
python -m pytest tests/ -q
```

86 tests covering the metric suite on synthetic scenes with known answers, the
degradation pipeline, the models, and the statistics. The fractal metrics are
validated against synthetic fractional Brownian surfaces of known Hurst
exponent and against the two limits bracketing them: white detector noise
(H = 0) and a smooth bandlimited field (H = 1).

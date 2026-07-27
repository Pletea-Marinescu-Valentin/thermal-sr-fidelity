#!/usr/bin/env bash
# End-to-end pipeline: train every model, evaluate them, make figures and a report.
# One command, no manual steps. Safe to re-run: finished work is skipped, so after a
# GPU crash you just launch it again and it resumes where it stopped.
#
#   bash scripts/pipeline.sh
#
# Stages, in order:
#   0  GPU health check (abort early with a clear message if the GPU is down)
#   1  train edsr_lite -> swinir_lite -> rrdb (L1) -> esrgan (adversarial)
#   2  evaluate all models on classic degradation  (+ paired tests vs bicubic)
#   3  evaluate all models on realistic degradation (robustness, no retraining)
#   4  qualitative hallucination-localisation figures
#   5  assemble results/REPORT.md
#
# Every training stage is resume-safe (--resume). The adversarial stage needs the
# L1-trained RRDB, so RRDB precedes it.

set -u
PY=".venv/Scripts/python.exe"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }

gpu_ok() {
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1
}

require_gpu() {
  if ! gpu_ok; then
    log "ERROR: GPU is not accessible (nvidia-smi failed)."
    log "       The RTX 5050 dropped off the bus earlier; this needs a Windows reboot."
    log "       After rebooting, just run this script again -- it resumes from the"
    log "       last checkpoint, nothing is lost."
    exit 2
  fi
}

# finished <run_dir> <target_steps> : true if training reached the target
finished() {
  [ -f "$1/history.json" ] && "$PY" -c "
import json, sys
h = json.load(open(r'$1/history.json'))
sys.exit(0 if h and h[-1]['step'] >= $2 else 1)
" 2>/dev/null
}

# train_stage <label> <run_dir> <target_steps> <cmd...>
train_stage() {
  local label="$1" dir="$2" steps="$3"; shift 3
  if finished "$dir" "$steps"; then
    log "SKIP train:$label (already at $steps steps)"
    return 0
  fi
  local attempt
  for attempt in 1 2 3; do
    require_gpu
    log "START train:$label attempt $attempt -> $LOG_DIR/$label.log"
    "$@" >>"$LOG_DIR/$label.log" 2>&1
    if finished "$dir" "$steps"; then
      log "DONE train:$label"
      return 0
    fi
    log "train:$label did not reach $steps (attempt $attempt); checking GPU"
    gpu_ok || { log "GPU lost during train:$label"; require_gpu; }
    sleep 5
  done
  log "FAILED train:$label after 3 attempts -- see $LOG_DIR/$label.log"
  return 1
}

# eval_stage <preset>  (idempotent: eval_all skips models it already scored)
eval_stage() {
  local preset="$1"
  require_gpu
  log "START eval:$preset"
  if "$PY" -u scripts/eval_all.py --preset "$preset" >>"$LOG_DIR/eval_$preset.log" 2>&1; then
    log "DONE eval:$preset"
  else
    log "FAILED eval:$preset -- see $LOG_DIR/eval_$preset.log"
  fi
}

log "==== pipeline start ===="
require_gpu

# ---- stage 1: training queue -------------------------------------------------
train_stage edsr_lite runs/edsr_lite_x4_classic 100000 \
  "$PY" -u scripts/train.py edsr_lite --steps 100000 --batch 16 --patch 48 \
        --val-every 5000 --resume

train_stage swinir_lite runs/swinir_lite_x4_classic 100000 \
  "$PY" -u scripts/train.py swinir_lite --steps 100000 --batch 16 --patch 48 \
        --val-every 5000 --resume

train_stage rrdb runs/rrdb_x4_classic 100000 \
  "$PY" -u scripts/train.py rrdb --steps 100000 --batch 16 --patch 48 \
        --val-every 5000 --resume

if [ -f runs/rrdb_x4_classic/best.pt ]; then
  train_stage esrgan runs/esrgan_x4_classic 50000 \
    "$PY" -u scripts/train_gan.py --pretrained runs/rrdb_x4_classic/best.pt \
          --steps 50000 --batch 8 --patch 32 --val-every 2500 --resume
else
  log "SKIP train:esrgan (rrdb best.pt missing)"
fi

# ---- stage 2 + 3: evaluation -------------------------------------------------
eval_stage classic
eval_stage realistic

# ---- stage 4: qualitative figures --------------------------------------------
require_gpu
log "START figures"
if "$PY" -u scripts/make_figures.py --frames 6 >>"$LOG_DIR/figures.log" 2>&1; then
  log "DONE figures"
else
  log "FAILED figures -- see $LOG_DIR/figures.log"
fi

# ---- stage 5: report (cheap, no GPU) -----------------------------------------
log "START report"
"$PY" -u scripts/make_report.py >>"$LOG_DIR/report.log" 2>&1 && log "DONE report"

log "==== pipeline finished ===="
log "See results/REPORT.md, results/comparison_*.md and results/figures/"

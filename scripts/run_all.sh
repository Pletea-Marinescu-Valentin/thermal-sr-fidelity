#!/usr/bin/env bash
# Sequential training queue: one model at a time, so the GPU is never oversubscribed
# and wall-clock timings stay comparable between architectures.
#
# Order matters: the adversarial stage initialises its generator from the L1-trained
# RRDB, so RRDB must finish first.
#
# Every stage is resume-safe (--resume picks up runs/<name>/last.pt), so the queue can
# be killed and restarted without losing work.
#
# Usage: bash scripts/run_all.sh

set -u
PY=".venv/Scripts/python.exe"
LOGS="runs/logs"
mkdir -p "$LOGS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

finished() {   # finished <run_dir> <target_steps>
  [ -f "$1/history.json" ] && "$PY" -c "
import json, sys
h = json.load(open(r'$1/history.json'))
sys.exit(0 if h and h[-1]['step'] >= $2 else 1)
" 2>/dev/null
}

wait_for() {   # wait_for <run_dir> <target_steps> <label>
  log "waiting for $3 to reach $2 steps..."
  until finished "$1" "$2"; do sleep 60; done
  log "$3 complete"
}

run_stage() {  # run_stage <label> <run_dir> <steps> <command...>
  local label="$1" dir="$2" steps="$3"; shift 3
  if finished "$dir" "$steps"; then
    log "SKIP $label (already at $steps steps)"
    return 0
  fi
  log "START $label -> $LOGS/$label.log"
  "$@" >>"$LOGS/$label.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED $label (exit $rc) -- see $LOGS/$label.log"
  else
    log "DONE $label"
  fi
  return $rc
}

# ---- stage 0: the EDSR run already in flight
wait_for runs/edsr_lite_x4_classic 100000 "edsr_lite"

# ---- stage 1: transformer
run_stage swinir_lite runs/swinir_lite_x4_classic 100000 \
  "$PY" -u scripts/train.py swinir_lite --steps 100000 --batch 16 --patch 48 \
        --val-every 5000 --resume

# ---- stage 2: RRDB, L1 only (the GAN's starting point)
run_stage rrdb_l1 runs/rrdb_x4_classic 100000 \
  "$PY" -u scripts/train.py rrdb --steps 100000 --batch 16 --patch 48 \
        --val-every 5000 --resume

# ---- stage 3: adversarial fine-tuning
if [ -f runs/rrdb_x4_classic/best.pt ]; then
  run_stage esrgan runs/esrgan_x4_classic 50000 \
    "$PY" -u scripts/train_gan.py --pretrained runs/rrdb_x4_classic/best.pt \
          --steps 50000 --batch 8 --patch 32 --val-every 2500 --resume
else
  log "SKIP esrgan: runs/rrdb_x4_classic/best.pt missing"
fi

log "queue finished"

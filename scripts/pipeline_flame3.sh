#!/usr/bin/env bash
# Train the five FLAME 3 arms: three regressors, then the adversarial model
# initialised from the converged RRDB, exactly as on the ADAS track so the
# RRDB-versus-ESRGAN comparison stays controlled.
set -u
PY=".venv/Scripts/python.exe"
LOG="runs/logs"
STEPS=${STEPS:-50000}
GAN_STEPS=${GAN_STEPS:-25000}

log() { echo "[$(date +%H:%M:%S)] $*"; }

for m in edsr_lite swinir_lite rrdb; do
  out="runs/${m}_x4_classic_flame3"
  if [ -f "$out/best.pt" ]; then log "SKIP $m (already trained)"; continue; fi
  log "TRAIN $m ($STEPS steps)"
  "$PY" -u scripts/train.py "$m" --dataset flame3 --steps "$STEPS" \
      --batch 16 --patch 48 --val-every 2500 --resume \
      >>"$LOG/${m}_flame3.log" 2>&1 \
    && log "DONE $m" || { log "FAILED $m"; exit 1; }
done

out="runs/esrgan_x4_classic_flame3"
if [ -f "$out/last.pt" ]; then
  log "SKIP esrgan (already trained)"
else
  log "TRAIN esrgan ($GAN_STEPS steps)"
  "$PY" -u scripts/train_gan.py --dataset flame3 \
      --pretrained runs/rrdb_x4_classic_flame3/best.pt \
      --steps "$GAN_STEPS" --batch 8 --patch 32 --val-every 2500 --resume \
      >>"$LOG/esrgan_flame3.log" 2>&1 \
    && log "DONE esrgan" || { log "FAILED esrgan"; exit 1; }
fi
log "ALL FLAME 3 TRAINING COMPLETE"

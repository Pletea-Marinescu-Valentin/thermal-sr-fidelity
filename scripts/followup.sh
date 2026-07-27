#!/usr/bin/env bash
# Follow-up runs after the main pipeline: add LPIPS to the radiometric comparison
# and apply the GAN to the FLIR-IISR cross-sensor gap. Resumable in spirit -- each
# step overwrites its own result file.
#
#   bash scripts/followup.sh

set -u
PY=".venv/Scripts/python.exe"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/followup.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_ok() { nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; }

log "==== followup start ===="
gpu_ok || { log "ERROR: GPU not accessible -- reboot and rerun"; exit 2; }

# 1 + 2: recompute the radiometric comparison with LPIPS on both degradations.
log "START eval:classic (with LPIPS)"
"$PY" -u scripts/eval_all.py --preset classic --force >>"$LOG_DIR/eval_classic_lpips.log" 2>&1 \
  && log "DONE eval:classic" || log "FAILED eval:classic"

log "START eval:realistic (with LPIPS)"
"$PY" -u scripts/eval_all.py --preset realistic --force >>"$LOG_DIR/eval_realistic_lpips.log" 2>&1 \
  && log "DONE eval:realistic" || log "FAILED eval:realistic"

# 3: GAN on the FLIR-IISR cross-sensor gap (200 frames, matching the bicubic run).
log "START iisr:esrgan (cross-sensor stress)"
"$PY" -u scripts/eval_iisr.py --model esrgan --limit 200 >>"$LOG_DIR/iisr_esrgan.log" 2>&1 \
  && log "DONE iisr:esrgan" || log "FAILED iisr:esrgan"

# 4: refresh the report.
log "START report"
"$PY" -u scripts/make_report.py >>"$LOG_DIR/report.log" 2>&1 && log "DONE report"

log "==== followup finished ===="

#!/bin/bash
# Run exactly one (cv_iter, model_i) training task.
# Args:
#   $1: TASK_ID (0 .. N_CV_ITERS*N_MODELS_PER_CV_ITER - 1)
#   $2: CONFIG_FP
#   $3: OUTPUT_DIR
#   $4: N_CV_ITERS
#   $5: N_MODELS_PER_CV_ITER

set -euo pipefail

TASK_ID="$1"
CONFIG_FP="$2"
OUTPUT_DIR="$3"
N_CV_ITERS="$4"
N_MODELS_PER_CV_ITER="$5"

export TF_CPP_MIN_LOG_LEVEL=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/train/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/src/train/train_model.py

CV_ITER=$(( TASK_ID / N_MODELS_PER_CV_ITER ))
MODEL_I=$(( TASK_ID % N_MODELS_PER_CV_ITER ))

if [[ "$CV_ITER" -ge "$N_CV_ITERS" ]]; then
  echo "TASK_ID=$TASK_ID maps to CV_ITER=$CV_ITER which is >= N_CV_ITERS=$N_CV_ITERS. Exiting."
  exit 0
fi

CV_ITER_DIR="$OUTPUT_DIR/cv_iter_${CV_ITER}"
MODELS_DIR="$CV_ITER_DIR/models"
MODEL_DIR="$MODELS_DIR/model${MODEL_I}"
mkdir -p "$MODEL_DIR"

CV_LOG_FP="$CV_ITER_DIR/cv_iter_${CV_ITER}.log"
echo "Starting TASK_ID=$TASK_ID => CV_ITER=$CV_ITER MODEL_I=$MODEL_I at $(date)" | tee -a "$CV_LOG_FP"

LOG_FP_TRAIN_MODEL="$MODEL_DIR/train_model_${MODEL_I}.log"

# Skip if already trained (supports retries/requeues)
if [[ -f "$MODEL_DIR/model.keras" ]]; then
  echo "[SKIP] model already exists: $MODEL_DIR/model.keras" | tee -a "$LOG_FP_TRAIN_MODEL"
  exit 0
fi

# Create per-model config for this CV fold
python "$SETUP_CV_ITER_FP" \
  --cv_iter="$CV_ITER" \
  --config_fp="$CONFIG_FP" \
  --output_dir="$MODEL_DIR" &>> "$LOG_FP_TRAIN_MODEL"

CV_ITER_CONFIG_FP="$MODEL_DIR/config_cv.yaml"

# Train
python "$TRAIN_MODEL_SCRIPT_FP" \
  --config_fp="$CV_ITER_CONFIG_FP" \
  --model_dir="$MODEL_DIR" &>> "$LOG_FP_TRAIN_MODEL"

# Success marker
if [[ -f "$MODEL_DIR/model.keras" ]]; then
  touch "$MODEL_DIR/TRAIN_DONE"
  echo "[DONE] CV_ITER=$CV_ITER MODEL_I=$MODEL_I" | tee -a "$CV_LOG_FP"
else
  echo "[ERROR] model.keras missing after training for CV_ITER=$CV_ITER MODEL_I=$MODEL_I" | tee -a "$CV_LOG_FP"
  exit 1
fi

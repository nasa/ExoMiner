#!/bin/bash
# Create/evaluate/predict ensemble for a single CV iteration.
# Args:
#   $1: CV_ITER
#   $2: CONFIG_FP
#   $3: OUTPUT_DIR
#   $4: N_MODELS_PER_CV_ITER

set -euo pipefail

CV_ITER="$1"
CONFIG_FP="$2"
OUTPUT_DIR="$3"
N_MODELS_PER_CV_ITER="$4"

SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/train/setup_cv_iter.py
CREATE_ENSEMBLE_MODEL_SCRIPT_FP=$PYTHONPATH/models/create_ensemble_avg_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict/predict_model.py

CV_ITER_DIR="$OUTPUT_DIR/cv_iter_${CV_ITER}"
MODELS_DIR="$CV_ITER_DIR/models"
ENSEMBLE_MODEL_DIR="$CV_ITER_DIR/ensemble_model"
mkdir -p "$ENSEMBLE_MODEL_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR/cv_iter_${CV_ITER}_ensemble.log"

echo "[INFO] Waiting for $N_MODELS_PER_CV_ITER trained models for CV_ITER=$CV_ITER" | tee -a "$LOG_FP_CV_ITER"

# Wait for all models (robust barrier for the fold)
for ((i=0; i<N_MODELS_PER_CV_ITER; i++)); do
  MODEL_DIR="$MODELS_DIR/model${i}"
  while [[ ! -f "$MODEL_DIR/model.keras" ]]; do
    echo "[WAIT] CV_ITER=$CV_ITER waiting for model$i (missing $MODEL_DIR/model.keras)" | tee -a "$LOG_FP_CV_ITER"
    sleep 120
  done
  echo "[OK] model$i ready" | tee -a "$LOG_FP_CV_ITER"
done

# Setup ensemble config
LOG_FP_SETUP="$ENSEMBLE_MODEL_DIR/setup_ensemble_config.log"
python "$SETUP_CV_ITER_FP" \
  --cv_iter="$CV_ITER" \
  --config_fp="$CONFIG_FP" \
  --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_SETUP"

CV_ITER_CONFIG_FP="$ENSEMBLE_MODEL_DIR/config_cv.yaml"
ENSEMBLE_MODEL_FP="$ENSEMBLE_MODEL_DIR/ensemble_avg_model.keras"

# Create ensemble
LOG_FP_CREATE="$ENSEMBLE_MODEL_DIR/create_ensemble_model.log"
python "$CREATE_ENSEMBLE_MODEL_SCRIPT_FP" \
  --config_fp="$CV_ITER_CONFIG_FP" \
  --models_dir="$MODELS_DIR" \
  --ensemble_fp="$ENSEMBLE_MODEL_FP" &>> "$LOG_FP_CREATE"

echo "[DONE] Created ensemble for CV_ITER=$CV_ITER" | tee -a "$LOG_FP_CV_ITER"

# Evaluate
LOG_FP_EVAL="$ENSEMBLE_MODEL_DIR/eval_ensemble_model.log"
python "$EVAL_MODEL_SCRIPT_FP" \
  --config_fp="$CV_ITER_CONFIG_FP" \
  --model_fp="$ENSEMBLE_MODEL_FP" \
  --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_EVAL"

# Predict
LOG_FP_PRED="$ENSEMBLE_MODEL_DIR/predict_ensemble_model.log"
python "$PREDICT_MODEL_SCRIPT_FP" \
  --config_fp="$CV_ITER_CONFIG_FP" \
  --model_fp="$ENSEMBLE_MODEL_FP" \
  --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_PRED"

echo "[DONE] Evaluated and predicted with ensemble for CV_ITER=$CV_ITER" | tee -a "$LOG_FP_CV_ITER"

# Run job for average ensemble based on a set of trained models.

# Path to the codebase root directory
export PYTHONPATH=

MODELS_DIR=
CONFIG_FP=
ENSEMBLE_MODEL_DIR=

CREATE_ENSEMBLE_MODEL_SCRIPT_FP=$PYTHONPATH/models/create_ensemble_avg_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict_model.py

mkdir -p "$ENSEMBLE_MODEL_DIR"
LOG_FP="$ENSEMBLE_MODEL_DIR"/run_ensemble_model.log
ENSEMBLE_MODEL_FP="$ENSEMBLE_MODEL_DIR"/ensemble_avg_model.keras

echo "Creating ensemble model in $ENSEMBLE_MODEL_FP for models in $MODELS_DIR..." >> "$LOG_FP"

# create ensemble model
python "$CREATE_ENSEMBLE_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --models_dir="$MODELS_DIR" --ensemble_fp="$ENSEMBLE_MODEL_FP" &>> "$LOG_FP"

echo "Created ensemble model in $ENSEMBLE_MODEL_FP for models in $MODELS_DIR." >> "$LOG_FP"

# evaluate ensemble model
echo "Started evaluating ensemble model..." >> "$LOG_FP"

# evaluate and predict with ensemble model
LOG_FP_EVAL_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/eval_ensemble_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_EVAL_ENSEMBLE_MODEL"

echo "Evaluated ensemble model." >> "$LOG_FP"

# run inference with ensemble model
echo "Started running inference with ensemble model..." >> "$LOG_FP"

# evaluate and predict with ensemble model
LOG_FP_PREDICT_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/predict_ensemble_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_PREDICT_ENSEMBLE_MODEL"

echo "Ran inference with ensemble model." >> "$LOG_FP"

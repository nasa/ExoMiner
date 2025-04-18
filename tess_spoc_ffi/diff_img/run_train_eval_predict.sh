# Run job for one CV iteration for transfer learning of TESS SPOC FFI with fine-tuning
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the 2-min run
# $4: File path to configuration yaml file for the FFI run
# $5: Path to output directory
# $6: Number of GPUs per node
# $7: Number of trained models per CV iteration
# $8: Check GPU usage (default is false)
# $9: Number of CV iterations (default is -1)

# External arguments
#GNU_PARALLEL_INDEX="$1"
#JOB_ARRAY_INDEX="$2"
#CONFIG_FP_2MIN="$3"
#CONFIG_FP_FFI="$4"
#CV_DIR="$5"
#N_GPUS_PER_NODE="$6"
#N_MODELS_PER_CV_ITER="$7"
#CHECK_GPU=${8:-0}
#N_CV_ITERS=${9:--1}


PYTHONPATH=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# define paths to python scripts in the codebase
SETUP_ITER_FP=$PYTHONPATH/tess_spoc_ffi/diff_img/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/tess_spoc_ffi/diff_img/train_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/tess_spoc_ffi/diff_img/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/tess_spoc_ffi/diff_img/predict_model.py
CONFIG_FP=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/tess_spoc_ffi/diff_img/config_cv_train.yaml

EXPERIMENT_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/diff_img/original_diff_img_branch_regression_tce_dikco_msky_mae_8-14-2025_2157/

mkdir -p "$EXPERIMENT_DIR"

LOG_EXPERIMENT="$EXPERIMENT_DIR"/output.log

echo "Starting experiment $EXPERIMENT_DIR ..." > "$LOG_EXPERIMENT"

# setup run
echo "Setting up configuration for model..." >> "$LOG_EXPERIMENT"
python "$SETUP_ITER_FP" --config_fp="$CONFIG_FP" --output_dir="$EXPERIMENT_DIR" --model_i=0 --cv_iter=0 &>> "$LOG_EXPERIMENT"
CONFIG_FP=$EXPERIMENT_DIR/config_cv.yaml

MODEL_FP=$EXPERIMENT_DIR/model.keras
if [ -f "$MODEL_FP" ]; then
    echo "Model already exists. Skipping training..." >> "$LOG_EXPERIMENT"
else
    echo "Training model..." >> "$LOG_EXPERIMENT"
    python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_dir="$EXPERIMENT_DIR" &>> "$LOG_EXPERIMENT"
    echo "Finished training model." >> "$LOG_EXPERIMENT"
fi

# evaluate ensemble model
echo "Started evaluating model..." >> "$LOG_EXPERIMENT"

python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$EXPERIMENT_DIR" &>> "$LOG_EXPERIMENT"

echo "Evaluated model." >> "$LOG_EXPERIMENT"

# run inference on FFI data with ensemble model of FFI trained models
echo "Started running inference..." >> "$LOG_EXPERIMENT"

python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$EXPERIMENT_DIR" &>> "$LOG_EXPERIMENT"

echo "Ran inference with model." >> "$LOG_EXPERIMENT"

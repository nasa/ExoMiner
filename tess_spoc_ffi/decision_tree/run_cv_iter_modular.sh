# Run job for one CV iteration for transfer learning of TESS SPOC FFI with fine-tuning
# Args:
# $1: CV iteration index
# $2: File path to configuration yaml file for the FFI run
# $3: Path to output directory
# $4: Number of CV iterations (default is -1)

# External arguments
CV_ITER="$1"
CONFIG_FP="$2"
CV_DIR="$3"
N_CV_ITERS=${4:--1}

# define paths to python scripts in the codebase
SETUP_CV_ITER_FP=$PYTHONPATH/tess_spoc_ffi/decision_tree/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/tess_spoc_ffi/decision_tree/train_model.py

# Check if N_CV_ITERS is different from -1 and CV_ITER is greater than or equal to N_CV_ITERS
if [ "$N_CV_ITERS" -ne -1 ] && [ "$CV_ITER" -ge "$N_CV_ITERS" ]; then
    echo "Stopping script: CV iteration ($CV_ITER) is greater than or equal to total number of CV iterations ($N_CV_ITERS)"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=''

#if [ "$CV_ITER" -ne 9 ]; then  #  && [ "$CV_ITER" -ne 8 ]; then
#  echo "CV_ITER must be 9. Exiting."
#  exit 1
#fi

CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR"/cv_iter_"$CV_ITER"_run.log

echo "Starting job for CV iteration $CV_ITER..." > "$LOG_FP_CV_ITER"

# train set of models
MODELS_DIR="$CV_ITER_DIR"/models
mkdir -p "$MODELS_DIR"
MODEL_I=0
# run second training step on  data
MODEL_DIR="$MODELS_DIR"/model$MODEL_I
mkdir -p "$MODEL_DIR"
LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model_"$MODEL_I".log

# setup run
echo "Setting up configuration for model in CV iteration $CV_ITER" >> "$LOG_FP_CV_ITER"
python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$MODEL_DIR" --model_i="$MODEL_I" &>> "$LOG_FP_TRAIN_MODEL"
CV_ITER_CONFIG_FP=$MODEL_DIR/config_cv.yaml

#if [ -f "$MODEL_DIR/model.ydf" ]; then
#    echo "Model in CV iteration $CV_ITER already exists. Skipping..."
#    exit 0
#fi

echo "Training model in CV iteration $CV_ITER for data..." >> "$LOG_FP_CV_ITER"
python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_dir="$MODEL_DIR" &> "$LOG_FP_TRAIN_MODEL"

echo "Finished training model in CV iteration $CV_ITER" >> "$LOG_FP_CV_ITER"

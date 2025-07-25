# Run job for one trained model iteration.
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the run
# $4: Path to output directory
# $5: Number of GPUs per node
# $6: Number of trained models

# External arguments
MODELS_DIR="$4"
CONFIG_FP="$3"
N_GPUS_PER_NODE="$5"
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
N_MODELS="$6"

# if needed
source /path/to/bash

# initialize conda and activate conda environment
source activate exoplnt_dl_gh

# path to codebase root directory
export PYTHONPATH=codebase/

# Paths
SETUP_RUN_FP=$PYTHONPATH/src/train/setup_train.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/src/train/train_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict/predict_model.py

# set model id based on received indexes
MODEL_I=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))

# Check if MODEL_I is greater than or equal to N_MODELS
if [ "$MODEL_I" -ge "$N_MODELS" ]; then
    echo "Stopping script: model ID ($MODEL_I) is greater than or equal to total number of models ($N_MODELS)."
    exit 0
fi

MODEL_DIR="$MODELS_DIR"/model$MODEL_I
mkdir -p "$MODEL_DIR"

LOG_FP_MAIN="$MODEL_DIR"/model_run_"$MODEL_I".log

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for model $MODEL_I..." > "$LOG_FP_MAIN"

GPU_ID=$(($MODEL_I % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_MAIN"

# setup run
echo "Setting up config file for model $MODEL_I out of $N_MODELS models..." >> "$LOG_FP_MAIN"
python "$SETUP_RUN_FP" --config_fp="$CONFIG_FP" --output_dir="$MODEL_DIR" &> "$LOG_FP_MAIN"

# get config yaml filepath
MODEL_CONFIG_FP=$MODEL_DIR/config_run.yaml

# train model
LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model_"$MODEL_I".log
echo "Training model $MODEL_I out of $N_MODELS..." >> "$LOG_FP_MAIN"
python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$MODEL_CONFIG_FP" --model_dir="$MODEL_DIR" &> "$LOG_FP_TRAIN_MODEL"
echo "Finished training model $MODEL_I out of $N_MODELS." >> "$LOG_FP_MAIN"

# get model filepath
MODEL_FP=$MODEL_DIR/model.keras

# evaluate model
echo "Evaluating model $MODEL_I out of $N_MODELS..." >> "$LOG_FP_MAIN"
LOG_FP_EVAL_MODEL="$MODEL_DIR"/eval_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$MODEL_CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$MODEL_DIR" &> "$LOG_FP_EVAL_MODEL"
echo "Evaluated model $MODEL_I out of $N_MODELS models." >> "$LOG_FP_MAIN"

# run inference with model
echo "Running inference for model $MODEL_I out of $N_MODELS models..." >> "$LOG_FP_MAIN"
LOG_FP_PREDICT_MODEL="$MODEL_DIR"/predict_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$MODEL_CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$MODEL_DIR" &> "$LOG_FP_PREDICT_MODEL"

echo "Ran inference for model $MODEL_I out of $N_MODELS models." >> "$LOG_FP_MAIN"

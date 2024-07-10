# Run job train, eval, and run inference on trained model
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the run
# $4: Path to output directory
# $5: Number of GPUs per node

# External arguments
EXP_DIR="$4"
CONFIG_FP="$3"
N_GPUS_PER_NODE="$5"
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"

source "$HOME"/.bashrc

#conda activate exoplnt_dl
conda activate exoplnt_dl_tf2_13

export PYTHONPATH=/Users/agiri1/Desktop/ExoPlanet

# Paths
SETUP_RUN_FP=$PYTHONPATH/src/setup_train.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/src_cv/train_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src_cv/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src_cv/predict_model.py

MODEL_I=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))

MODELS_DIR="$EXP_DIR"/models
mkdir -p "$MODELS_DIR"

MODEL_DIR="$MODELS_DIR"/model$MODEL_I
mkdir -p "$MODEL_DIR"

LOG_FP_MODEL_I="$MODEL_DIR"/run_model_"$MODEL_I".log

echo "Starting job for model $MODEL_I..." > "$LOG_FP_MODEL_I"

GPU_ID=$(("$MODEL_I" % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_MODEL_I"

# setup run
echo "Setting up train iteration." >> "$LOG_FP_MODEL_I"
python "$SETUP_RUN_FP" --config_fp="$CONFIG_FP" --output_dir="$MODEL_DIR" &>> "$LOG_FP_MODEL_I"
CONFIG_FP=$MODEL_DIR/config_run.yaml

# train model
echo "Started training model $MODEL_I..." >> "$LOG_FP_MODEL_I"

LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model.log
echo "Training model $MODEL_I..." >> "$LOG_FP_MODEL_I"
python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_dir="$MODEL_DIR" &> "$LOG_FP_TRAIN_MODEL"
echo "Finished training model $MODEL_I" >> "$LOG_FP_MODEL_I"

# evaluate model
echo "Started evaluating model $MODEL_I..." >> "$LOG_FP_MODEL_I"

# evaluate model
LOG_FP_EVAL_MODEL="$MODEL_DIR"/eval_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$MODEL_DIR" &>> "$LOG_FP_EVAL_MODEL"

echo "Evaluated model $MODEL_I.." >> "$LOG_FP_MODEL_I"

# run inference with model
echo "Started running inference for model $MODEL_I..." >> "$LOG_FP_MODEL_I"

# predict with model
LOG_FP_PREDICT_MODEL="$MODEL_DIR"/predict_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CONFIG_FP" --model_fp="$MODEL_FP" --output_dir="$MODEL_DIR" &>> "$LOG_FP_PREDICT_MODEL"

echo "Ran inference with model $MODEL_I." >> "$LOG_FP_MODEL_I"

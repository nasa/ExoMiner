# Run job for one CV iteration for transfer learning of TESS SPOC FFI with fine-tuning
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the 2-min run
# $4: File path to configuration yaml file for the FFI run
# $5: Path to output directory
# $6: Number of GPUs per node
# $7: Number of CV iterations
# $8: Number of trained models per CV iteration

# External arguments
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
CONFIG_FP_2MIN="$3"
CONFIG_FP_FFI="$4"
CV_DIR="$5"
N_GPUS_PER_NODE="$6"
N_MODELS_PER_CV_ITER="$7"
CHECK_GPU=${8:-0}

# define paths to python scripts in the codebase
SETUP_CV_ITER_FP=$PYTHONPATH/tess_spoc_ffi/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/tess_spoc_ffi/train_model.py
CREATE_ENSEMBLE_MODEL_SCRIPT_FP=$PYTHONPATH/models/create_ensemble_avg_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict/predict_model.py

# set CV iteration ID
CV_ITER=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))

CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR"/cv_iter_"$CV_ITER"_run_"$GNU_PARALLEL_INDEX"_jobarray_"$JOB_ARRAY_INDEX".log

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for CV iteration $CV_ITER..." > "$LOG_FP_CV_ITER"

GPU_ID=$(($CV_ITER % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_CV_ITER"

if [ "$CHECK_GPU" -eq 1 ]; then
    PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
    until [ "$PROC_IN_GPU" == "" ]
    do
        PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
        echo "Current process in GPU $GPU_ID: $PROC_IN_GPU"
        sleep 60
    done

    echo "GPU $GPU_ID is available. Resuming CV iteration." >> "$LOG_FP_CV_ITER"
fi

# train set of models
MODELS_DIR_2MIN="$CV_ITER_DIR"/models_2min
MODELS_DIR_FFI="$CV_ITER_DIR"/models_ffi
mkdir -p "$MODELS_DIR_2MIN"
mkdir -p "$MODELS_DIR_FFI"

echo "Started training $N_MODELS_PER_CV_ITER models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

for ((MODEL_I=0; MODEL_I<$N_MODELS_PER_CV_ITER; MODEL_I++))
do

    # run first training step on 2-min data
    MODEL_DIR_2MIN="$MODELS_DIR_2MIN"/model$MODEL_I
    mkdir -p "$MODEL_DIR_2MIN"
    LOG_FP_TRAIN_MODEL_2MIN="$MODEL_DIR_2MIN"/train_model_2min_"$MODEL_I".log

    # setup run
    echo "Setting up configuration for model $MODEL_I in CV iteration $CV_ITER for 2-min data." >> "$LOG_FP_CV_ITER"
    python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP_2MIN" --output_dir="$MODEL_DIR_2MIN" --model_i="$MODEL_I" &>> "$LOG_FP_TRAIN_MODEL_2MIN"
    CV_ITER_CONFIG_FP_2MIN=$MODEL_DIR_2MIN/config_cv.yaml

    echo "Training model $MODEL_I in CV iteration $CV_ITER for 2-min data..." >> "$LOG_FP_CV_ITER"
    python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP_2MIN" --model_dir="$MODEL_DIR_2MIN" &> "$LOG_FP_TRAIN_MODEL_2MIN"
    echo "Finished training model $MODEL_I in CV iteration $CV_ITER for 2-min data" >> "$LOG_FP_CV_ITER"

    # run second training step on FFI data
    MODEL_DIR_FFI="$MODELS_DIR_FFI"/model$MODEL_I
    mkdir -p "$MODEL_DIR_FFI"
    LOG_FP_TRAIN_MODEL_FFI="$MODEL_DIR_FFI"/train_model_ffi_"$MODEL_I".log

    # setup run
    echo "Setting up configuration for model $MODEL_I in CV iteration $CV_ITER for FFI data." >> "$LOG_FP_CV_ITER"
    python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP_FFI" --output_dir="$MODEL_DIR_FFI" --model_i="$MODEL_I" &>> "$LOG_FP_TRAIN_MODEL_FFI"
    CV_ITER_CONFIG_FP_FFI=$MODEL_DIR_FFI/config_cv.yaml

    echo "Training model $MODEL_I in CV iteration $CV_ITER for FFI data..." >> "$LOG_FP_CV_ITER"
    python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP_FFI" --model_dir="$MODEL_DIR_FFI" --model_fp="$MODEL_DIR_2MIN"/model.keras &> "$LOG_FP_TRAIN_MODEL_FFI"
    echo "Finished training model $MODEL_I in CV iteration $CV_ITER for FFI data" >> "$LOG_FP_CV_ITER"

done

echo "Finished training all models for CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

echo "Creating ensemble model in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

ENSEMBLE_MODEL_DIR="$CV_ITER_DIR"/ensemble_model
mkdir -p "$ENSEMBLE_MODEL_DIR"
LOG_FP_CREATE_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/create_ensemble_model.log
ENSEMBLE_MODEL_FP="$ENSEMBLE_MODEL_DIR"/ensemble_avg_model.keras

# create ensemble model
python "$CREATE_ENSEMBLE_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP_FFI" --models_dir="$MODELS_DIR_FFI" --ensemble_fp="$ENSEMBLE_MODEL_FP" &> "$LOG_FP_CREATE_ENSEMBLE_MODEL"

echo "Created ensemble model in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# evaluate ensemble model
echo "Started evaluating ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_EVAL_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/eval_ensemble_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &> "$LOG_FP_EVAL_ENSEMBLE_MODEL"

echo "Evaluated ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# run inference with ensemble model
echo "Started running inference with ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_PREDICT_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/predict_ensemble_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP_FFI" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &> "$LOG_FP_PREDICT_ENSEMBLE_MODEL"

echo "Ran inference with ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# Run job for one CV iteration
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the run
# $4: Path to output directory
# $5: Number of GPUs per node
# $6: Root directory for model file paths
# $7: Boolean variable that if set to true it will preprocess data before running inference
# $8: File path to configuration yaml file for the preprocessing the data before running inference
# $9: Boolean variable that if set to true it will delete preprocessed data after running inference if variable
# `PREPROCESS_DATA` is also true

# External arguments
CV_DIR="$4"
CONFIG_FP="$3"
N_GPUS_PER_NODE="$5"
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
MODELS_CV_ROOT_DIR="$6"
PREPROCESS_DATA="$7"
CONFIG_PREPROCESS_FP="$8"
DELETE_DATA_AFTER_INFERENCE="$9"
CHECK_GPU=${10:-0}

# set up Python scripts
SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/predict/setup_cv_iter_predict.py
PREPROCESS_SCRIPT=$PYTHONPATH/src_cv/preprocessing/preprocess_cv_folds_predict_trecord_dataset.py
CREATE_CV_FOLDS_SCRIPT=$PYTHONPATH/src_cv/preprocessing/create_cv_folds_yaml_from_dir.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict/predict_model.py

# set CV iteration id
CV_ITER=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))

# create directory for CV iteration
CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

# set main log file
LOG_FP_CV_ITER="$CV_ITER_DIR"/cv_run_"$GNU_PARALLEL_INDEX"_jobarray_"$JOB_ARRAY_INDEX".log

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for CV iteration $CV_ITER..." > "$LOG_FP_CV_ITER"

GPU_ID=$(($CV_ITER % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_CV_ITER"

if [ "$CHECK_GPU" -eq 1 ]; then
    # check if GPU is free
    PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
    until [ "$PROC_IN_GPU" == "" ]
    do
        PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
        echo "Current process in GPU $GPU_ID: $PROC_IN_GPU"
        sleep 60
    done

    echo "GPU $GPU_ID is available. Resuming CV iteration." >> "$LOG_FP_CV_ITER"
fi

# preprocess data
if [ "$PREPROCESS_DATA" = true ] ; then
    echo "Started preprocessing data for CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

    export DATA_DIR="$CV_DIR"/data
    mkdir -p "$DATA_DIR"

    python "$PREPROCESS_SCRIPT" --rank=$CV_ITER --config_fp="$CONFIG_PREPROCESS_FP" --output_dir="$DATA_DIR" &>> "$LOG_FP_CV_ITER"

    python "$CREATE_CV_FOLDS_SCRIPT" --config_fp="$CONFIG_FP" --data_dir="$DATA_DIR"/cv_iter_$CV_ITER --cv_iter=$CV_ITER &>> "$LOG_FP_CV_ITER"
    CONFIG_FP=$DATA_DIR/cv_iter_$CV_ITER/config_cv.yaml

    echo "Finished preprocessing data for CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
fi

# setup run
echo "Setting up CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$CV_ITER_DIR" &>> "$LOG_FP_CV_ITER"
CV_ITER_CONFIG_FP=$CV_ITER_DIR/config_cv.yaml

# run inference with ensemble model
echo "Started running inference with ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

# get model file path
ENSEMBLE_MODEL_FP="$MODELS_CV_ROOT_DIR/cv_iter_$CV_ITER/ensemble_model/ensemble_avg_model.keras"

# evaluate and predict with ensemble model
LOG_FP_PREDICT_ENSEMBLE_MODEL="$CV_ITER_DIR"/predict_ensemble_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$CV_ITER_DIR" &> "$LOG_FP_PREDICT_ENSEMBLE_MODEL"

echo "Ran inference with ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# delete preprocessed data after running inference
if [ "$DELETE_DATA_AFTER_INFERENCE" = true ] && [ "$PREPROCESS_DATA" = true ]; then
    echo "Deleting data in $DATA_DIR/cv_iter_$CV_ITER for CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
    rm -r "$DATA_DIR"/cv_iter_$CV_ITER
fi

# Run job for one CV iteration
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: Number of sub-jobs
# $4: File path to configuration yaml file for the run
# $5: Path to output directory
# $6: Number of GPUs per node
# $7: Number of CV iterations
# $8: Number of trained models per CV iteration
# $9: Check if GPU is being used already; set to true by default

# External arguments
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
N_JOBS="$3"
CONFIG_FP="$4"
CV_DIR="$5"
N_GPUS_PER_NODE="$6"
N_CV_ITERS="$7"
N_MODELS_PER_CV_ITER="$8"
CHECK_GPU=${9:-0}

export TF_CPP_MIN_LOG_LEVEL=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# define paths to python scripts in the codebase
SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/train/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/src/train/train_model.py
CREATE_ENSEMBLE_MODEL_SCRIPT_FP=$PYTHONPATH/models/create_ensemble_avg_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict/predict_model.py

# set CV iteration ID
CV_ITER=$((GNU_PARALLEL_INDEX + JOB_ARRAY_INDEX * N_JOBS))

# Check if CV_ITER is greater than or equal to N_CV_ITERS
if [ "$CV_ITER" -ge "$N_CV_ITERS" ]
then
  echo "CV iteration $CV_ITER is above total number of iterations ($N_CV_ITERS). Ending process."
  exit 0
fi

CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR"/cv_iter_"$CV_ITER"_run.log
touch "$LOG_FP_CV_ITER"

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

GPU_ID=$((CV_ITER % N_GPUS_PER_NODE))
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

## setup run
#echo "Setting up CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
#python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$CV_ITER_DIR" &>> "$LOG_FP_CV_ITER"
#CV_ITER_CONFIG_FP=$CV_ITER_DIR/config_cv.yaml

## process data
#echo "Started preprocessing the data for CV iteration $CV_ITER..." >> "$LOG_FP"
#
#PREPROCESS_DATA_DIR="$OUTPUT_DIR"/norm_data
#mkdir -p "$PREPROCESS_DATA_DIR"
#LOG_FP_PREPROCESS_DATA="$PREPROCESS_DATA_DIR"/preprocess.log
#
#python $PREPROCESS_SCRIPT_FP --rank=0 --config_fp="$CONFIG_FP_PREPROCESS" --output_dir="$PREPROCESS_DATA_DIR" &>> "$LOG_FP_PREPROCESS_DATA"
#
#echo "Preprocessed data for CV iteration $CV_ITER." >> "$LOG_FP"

# train set of models
MODELS_DIR="$CV_ITER_DIR"/models
mkdir -p "$MODELS_DIR"

echo "Started training $N_MODELS_PER_CV_ITER models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

for ((MODEL_I=0; MODEL_I<N_MODELS_PER_CV_ITER; MODEL_I++))
do
    MODEL_DIR="$MODELS_DIR"/model$MODEL_I
    mkdir -p "$MODEL_DIR"
    LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model_"$MODEL_I".log

    if [ -f "$MODEL_DIR/model.keras" ]; then
        echo "Model $MODEL_I in CV iteration $CV_ITER already exists. Skipping training..." >> "$LOG_FP_CV_ITER"
        # continue
    else
      # setup run
      echo "Setting up configuration for model $MODEL_I in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
      python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$MODEL_DIR" &>> "$LOG_FP_TRAIN_MODEL"
      CV_ITER_CONFIG_FP=$MODEL_DIR/config_cv.yaml

      echo "Training model $MODEL_I in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"
      python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_dir="$MODEL_DIR" &>> "$LOG_FP_TRAIN_MODEL"  # &
      echo "Finished training model $MODEL_I in CV iteration $CV_ITER" >> "$LOG_FP_CV_ITER"
    fi
done

#PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
#until [ "$PROC_IN_GPU" == "" ]
#do
#    PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
#    echo "Current process in GPU $GPU_ID: $PROC_IN_GPU"
#    sleep 60
#done
#
#echo "GPU $GPU_ID is available. Resuming CV iteration." >> "$LOG_FP_CV_ITER"

echo "Finished training all models for CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

echo "Creating ensemble model in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

ENSEMBLE_MODEL_DIR="$CV_ITER_DIR"/ensemble_model
mkdir -p "$ENSEMBLE_MODEL_DIR"
LOG_FP_CREATE_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/create_ensemble_model.log
ENSEMBLE_MODEL_FP="$ENSEMBLE_MODEL_DIR"/ensemble_avg_model.keras

# setup run for ensemble
python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_CREATE_ENSEMBLE_MODEL"
CV_ITER_CONFIG_FP=$ENSEMBLE_MODEL_DIR/config_cv.yaml

# create ensemble model
python "$CREATE_ENSEMBLE_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --models_dir="$MODELS_DIR" --ensemble_fp="$ENSEMBLE_MODEL_FP" &> "$LOG_FP_CREATE_ENSEMBLE_MODEL"

echo "Created ensemble model in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# evaluate ensemble model
echo "Started evaluating ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_EVAL_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/eval_ensemble_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &> "$LOG_FP_EVAL_ENSEMBLE_MODEL"

echo "Evaluated ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# run inference with ensemble model
echo "Started running inference with ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_PREDICT_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/predict_ensemble_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &> "$LOG_FP_PREDICT_ENSEMBLE_MODEL"

echo "Ran inference with ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

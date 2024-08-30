# Run job for one CV iteration
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the run
# $4: Path to output directory
# $5: Number of GPUs per node
# $6: Number of CV iterations
# $7: Number of trained models per CV iteration

# External arguments
CV_DIR="$4"
CONFIG_FP="$3"
N_GPUS_PER_NODE="$5"
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
N_MODELS_PER_CV_ITER="$7"

source "$HOME"/.bashrc

#conda activate exoplnt_dl
conda activate exoplnt_dl_tf2_13

export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# Paths
SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/setup_cv_iter.py
TRAIN_MODEL_SCRIPT_FP=$PYTHONPATH/src/train_model.py
CREATE_ENSEMBLE_MODEL_SCRIPT_FP=$PYTHONPATH/models/create_ensemble_avg_model.py
EVAL_MODEL_SCRIPT_FP=$PYTHONPATH/src/evaluate_model.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict_model.py

CV_ITER=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))
# CV_ITER=$GNU_PARALLEL_INDEX
#if [ $CV_ITER -ge "$7" ]
#then
#  echo "CV iteration $CV_ITER is above total number of iterations ($7). Ending process."
#fi

CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR"/train_run_"$GNU_PARALLEL_INDEX"_jobarray_"$JOB_ARRAY_INDEX".log

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for CV iteration $CV_ITER..." > "$LOG_FP_CV_ITER"

GPU_ID=$(("$CV_ITER" % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_CV_ITER"

PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
until [ "$PROC_IN_GPU" == "" ]
do
    PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
    echo "Current process in GPU $GPU_ID: $PROC_IN_GPU"
    sleep 60
done

echo "GPU $GPU_ID is available. Resuming CV iteration." >> "$LOG_FP_CV_ITER"

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

for ((MODEL_I=0; MODEL_I<$N_MODELS_PER_CV_ITER; MODEL_I++))
do
    MODEL_DIR="$MODELS_DIR"/model$MODEL_I
    mkdir -p "$MODEL_DIR"
    LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model_"$MODEL_I".log

    # setup run
    echo "Setting up configuration for model $MODEL_I in CV iteration $CV_ITER." >> "$LOG_FP_TRAIN_MODEL"
    python "$SETUP_CV_ITER_FP" --cv_iter="$CV_ITER" --config_fp="$CONFIG_FP" --output_dir="$MODEL_DIR" --model_i="$MODEL_I" &>> "$LOG_FP_TRAIN_MODEL"
    CV_ITER_CONFIG_FP=$MODEL_DIR/config_cv.yaml

    echo "Training model $MODEL_I in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"
    python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_dir="$MODEL_DIR" &> "$LOG_FP_TRAIN_MODEL"
    echo "Finished training model $MODEL_I in CV iteration $CV_ITER" >> "$LOG_FP_CV_ITER"
done

#MODEL_IDS_FP="$CV_DIR"/missing_trained_models_cv_iter_"$CV_ITER".txt
#while read MODEL_I
#do
#    MODEL_DIR="$MODELS_DIR"/model$MODEL_I
#    mkdir -p $MODEL_DIR
#    LOG_FP_TRAIN_MODEL="$MODEL_DIR"/train_model_"$MODEL_I".log
#    echo "Training model $MODEL_I in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"
#    python "$TRAIN_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_dir="$MODEL_DIR" &>> "$LOG_FP_TRAIN_MODEL"
#    echo "Finished training model $MODEL_I in CV iteration $CV_ITER" >> "$LOG_FP_CV_ITER"
#done < "$MODEL_IDS_FP"

echo "Trained models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

echo "Creating ensemble model in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

ENSEMBLE_MODEL_DIR="$CV_ITER_DIR"/ensemble_model
mkdir -p "$ENSEMBLE_MODEL_DIR"
LOG_FP_CREATE_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/create_ensemble_model.log
ENSEMBLE_MODEL_FP="$ENSEMBLE_MODEL_DIR"/ensemble_avg_model.keras

# create ensemble model
python "$CREATE_ENSEMBLE_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --models_dir="$MODELS_DIR" --ensemble_fp="$ENSEMBLE_MODEL_FP" &>> "$LOG_FP_CREATE_ENSEMBLE_MODEL"

echo "Created ensemble model in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# evaluate ensemble model
echo "Started evaluating ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_EVAL_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/eval_ensemble_model.log
python "$EVAL_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_EVAL_ENSEMBLE_MODEL"

echo "Evaluated ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

# run inference with ensemble model
echo "Started running inference with ensemble of models in CV iteration $CV_ITER..." >> "$LOG_FP_CV_ITER"

LOG_FP_PREDICT_ENSEMBLE_MODEL="$ENSEMBLE_MODEL_DIR"/predict_ensemble_model.log
python "$PREDICT_MODEL_SCRIPT_FP" --config_fp="$CV_ITER_CONFIG_FP" --model_fp="$ENSEMBLE_MODEL_FP" --output_dir="$ENSEMBLE_MODEL_DIR" &>> "$LOG_FP_PREDICT_ENSEMBLE_MODEL"

echo "Ran inference with ensemble of models in CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"

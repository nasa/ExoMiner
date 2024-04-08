# Run job for one CV iteration
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to configuration yaml file for the run
# $4: Path to output directory
# $5: Number of GPUs per node
# $6: Root directory for model file paths

# External arguments
CV_DIR="$4"
CONFIG_FP="$3"
N_GPUS_PER_NODE="$5"
GNU_PARALLEL_INDEX="$1"
JOB_ARRAY_INDEX="$2"
MODELS_CV_ROOT_DIR="$6"

source "$HOME"/.bashrc

#conda activate exoplnt_dl
conda activate exoplnt_dl_tf2_13

export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# Paths
#SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/setup_cv_iter_predict.py
SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/setup_cv_iter.py
PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src_cv/predict_model.py

CV_ITER=$(($GNU_PARALLEL_INDEX + $JOB_ARRAY_INDEX * $N_GPUS_PER_NODE))

CV_ITER_DIR="$CV_DIR"/cv_iter_$CV_ITER
mkdir -p "$CV_ITER_DIR"

LOG_FP_CV_ITER="$CV_ITER_DIR"/cv_run_"$GNU_PARALLEL_INDEX"_jobarray_"$JOB_ARRAY_INDEX".log

echo "Starting job $GNU_PARALLEL_INDEX in job array $JOB_ARRAY_INDEX for CV iteration $CV_ITER..." > "$LOG_FP_CV_ITER"

GPU_ID=$(("$CV_ITER" % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP_CV_ITER"

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

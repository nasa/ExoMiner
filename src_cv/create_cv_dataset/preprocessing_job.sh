# Run job to preprocess data for CV experiment
# Args:
# $1: Rank of the job
# $2: Path to output directory for the preprocessing run
# $3: File path to main preprocessing Python script
# $4: File path to configuration yaml file for the preprocessing run
# $5: Total number of jobs

source "$HOME"/.bashrc

#conda activate exoplnt_dl
conda activate exoplnt_dl_tf2_13

export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

LOG_FP=$2/preprocess_cv_dataset.log

export CUDA_VISIBLE_DEVICES=""
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." > "$LOG_FP"

LOG_FP=$2/preprocessing_cv_dataset_$1.log

echo "Starting job $1..." > "$LOG_FP"

# run preprocessing pipeline
 python "$3" --rank="$1" --output_dir="$2" --config_fp="$4" &>> "$LOG_FP"

echo "Finished job $1..." >> "$LOG_FP"

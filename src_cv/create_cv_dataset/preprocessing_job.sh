# Run job to preprocess data for CV experiment
# Args:
# $1: Rank of the job
# $2: Path to output directory for the preprocessing run
# $3: File path to main preprocessing Python script
# $4: File path to configuration yaml file for the preprocessing run
# $5: Total number of jobs

source "$HOME"/.zshrc

#conda activate exoplnt_dl
conda activate exoplanet_env

export PYTHONPATH=/Users/agiri1/Desktop/ExoPlanet

LOG_DIR=$2/preprocessing_logs
mkdir -p $LOG_DIR
LOG_FP=$LOG_DIR/preprocessing_cv_dataset_$1.log
touch LOG_FP
export CUDA_VISIBLE_DEVICES=""
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_FP"

echo "Starting job $1..." >> "$LOG_FP"

# run preprocessing pipeline
python "$3" --rank="$1" --output_dir="$2" --log_dir="$LOG_DIR" --config_fp="$4" >> "$LOG_FP"

echo "Finished job $1..." >> "$LOG_FP"

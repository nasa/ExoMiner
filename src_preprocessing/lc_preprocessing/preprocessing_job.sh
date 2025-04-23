# Run job preprocessing pipeline
# Args:
# $1: Rank of the job
# $2: Path to output directory for the preprocessing run
# $3: File path to main preprocessing Python script
# $4: File path to configuration yaml file for the preprocessing run
# $5: Total number of jobs

LOG_DIR=$2/preprocessing_logs
mkdir -p "$LOG_DIR"
LOG_FP="$LOG_DIR"/preprocessing_$1.log

echo "Starting job $1 in node $HOSTNAME..." > "$LOG_FP"

# needed to be set after conda activate since that resets the environment variable
export CUDA_VISIBLE_DEVICES=''

# run preprocessing pipeline
python "$3" --rank="$1" --n_runs="$5" --output_dir="$2" --config_fp="$4" &>> "$LOG_FP"

echo "Finished job $1..." >> "$LOG_FP"

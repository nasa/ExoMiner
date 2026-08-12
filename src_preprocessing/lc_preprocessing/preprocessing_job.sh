#!/bin/bash
# Run job preprocessing pipeline
# Args:
# $1: Rank of the job
# $2: Path to output directory for the preprocessing run
# $3: File path to main preprocessing Python script
# $4: File path to configuration yaml file for the preprocessing run
# $5: Total number of jobs

source /home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/env_bootstrap.sh

LOG_DIR=$2/preprocessing_logs
mkdir -p "$LOG_DIR"
LOG_FP="$LOG_DIR"/preprocessing_$1.log

# needed to be set after conda activate since that resets the environment variable
export CUDA_VISIBLE_DEVICES=''

echo "Python: $(which python)" >> "$LOG_FP"
echo "Running on host: $HOSTNAME" >> "$LOG_FP"
echo "Job: $1 / Total: $5" >> "$LOG_FP"

# run preprocessing pipeline
python "$3" --rank="$1" --n_runs="$5" --output_dir="$2" --config_fp="$4" &>> "$LOG_FP"

echo "Finished job $1..." >> "$LOG_FP"

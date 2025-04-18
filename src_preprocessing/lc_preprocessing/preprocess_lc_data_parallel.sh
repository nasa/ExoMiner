# Conduct preprocessing run that creates a TFRecord dataset; uses GNU parallel to generate multiple processes, each one
# conducting the preprocessing of a subset of the examples.

# create output directory for preprocessing results
OUTPUT_DIR=
mkdir -p "$OUTPUT_DIR"

SCRIPT_FP=$(realpath "$0")
LC_PREPROCESSING_DIR=/codebase/src_preprocessing/lc_preprocessing

cp "$SCRIPT_FP" "$OUTPUT_DIR"/script_job.txt

# script file path
SCRIPT_FP=$LC_PREPROCESSING_DIR/generate_input_records.py
# config file path
CONFIG_FP=$LC_PREPROCESSING_DIR/config_preprocessing.yaml
# job script for running preprocessing pipeline
PREPROCESS_SH_SCRIPT=$LC_PREPROCESSING_DIR/preprocessing_job.sh

# number of total jobs
NUM_TOTAL_JOBS=384
# number of jobs run simultaneously in one node
NUM_JOBS_PARALLEL=128

# run with GNU parallel
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$PREPROCESS_SH_SCRIPT {} $OUTPUT_DIR $SCRIPT_FP $CONFIG_FP $NUM_TOTAL_JOBS"

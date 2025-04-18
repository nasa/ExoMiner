# Run train, eval experiment in parallel (multiple models run at the same time) on a single node with multiple GPUs and
# using a combination of GNU parallel and job array; option for multi-node setup.

# config file path
CONFIG_FP=/codebase/src/config_train.yaml
# job script for running the Python application
RUN_SH_SCRIPT=/codebase/src/run_train_iter.sh
# output directory
OUTPUT_DIR=
# path to codebase root directory
export PYTHONPATH=

N_MODELS=10  # total number of models to be trained

# number of GPUs to be used by this job array
N_GPUS_TOTAL=4

# number of total jobs per job in job array
NUM_TOTAL_JOBS=$((1 * 4))
# number of jobs run simultaneously
NUM_JOBS_PARALLEL=2

source "$HOME"/.bashrc
#source "$HOME"/.zshrc

# activate conda env
conda activate

mkdir -p $OUTPUT_DIR

# run with GNU parallel
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$RUN_SH_SCRIPT {} 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_MODELS"

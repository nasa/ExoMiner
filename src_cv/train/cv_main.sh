# Run CV experiment in parallel (multiple iterations at the same time) on a single GPU node with multiple GPUs and
# using a combination of GNU parallel and job array


# initialize conda and activate conda environment
source activate envname

# set path to codebase root directory
export PYTHONPATH=

# config file path
CONFIG_FP=codebase/src_cv/train/config_cv_train.yaml
# job script for running the Python application
RUN_SH_SCRIPT=codebase/src_cv/train/run_cv_iter_modular.sh
# output directory
OUTPUT_DIR=
mkdir -p $OUTPUT_DIR

N_CV_ITERS=5  # number of CV folds/iterations
N_MODELS_PER_CV_ITER=10  # number of models to train per CV iteration

# number of GPUs to be used by this job array
N_GPUS_TOTAL=4

# number of total jobs per job in job array
NUM_TOTAL_JOBS=4
# number of jobs run simultaneously
NUM_JOBS_PARALLEL=4

# run with GNU parallel
# run CV sh script
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$RUN_SH_SCRIPT {} 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_CV_ITERS $N_MODELS_PER_CV_ITER"

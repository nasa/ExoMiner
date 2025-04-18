# Run CV predict experiment in parallel (multiple iterations at the same time) on a single GPU node with multiple GPUs
# and using a combination of GNU parallel and job array


# initialize conda and activate conda environment
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate exoplnt_dl_tf2_13

# set path to codebase root directory
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# script for running inference for a single CV iteration
RUN_SH_SCRIPT=codebase/src_cv/predict/run_predict_cv_iter_modular.sh

# config yaml file used to run inference
CONFIG_FP=codebase/src_cv/predict/config_cv_predict.yaml

# config yaml file used to preprocess data for model inference
CONFIG_PREPROCESS_FP=codebase/src_cv/preprocessing/config_preprocess_cv_folds_predict_tfrecord_dataset.yaml

# output directory
OUTPUT_DIR=
mkdir -p $OUTPUT_DIR
SCRIPT_FP=$(realpath $0)
cp "$SCRIPT_FP" "$OUTPUT_DIR"/script_job.txt

# root directory for model file paths
MODELS_CV_ROOT_DIR=

# whether to preprocess data before running inference
PREPROCESS_DATA=true

# whether to delete preprocessed data after running inference
DELETE_PREPROCESSED_DATA=true

# number of GPUs to be used by this job array
N_GPUS_TOTAL=4

# number of total jobs per job in job array
NUM_TOTAL_JOBS=4
# number of jobs run simultaneously
NUM_JOBS_PARALLEL=4

# run with GNU parallel
# run CV sh script
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$RUN_SH_SCRIPT {} 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $MODELS_CV_ROOT_DIR" "$PREPROCESS_DATA" $CONFIG_PREPROCESS_FP "$DELETE_PREPROCESSED_DATA"

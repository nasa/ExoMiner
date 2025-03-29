# Run CV experiment in parallel (multiple iterations at the same time) on a single GPU node with multiple GPUs and
# using a combination of GNU parallel and job array

# initialize conda and activate conda environment
module use -a /swbuild/analytix/tools/modulefiles
# non-GH conda
module load miniconda3/v4
source activate exoplnt_dl_tf2_13
# GH conda (aarch64)
# module load miniconda3/gh2
# source activate exoplnt_dl_gh

# set path to codebase root directory
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# config files paths
CONFIG_FP=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/tess_spoc_ffi/decision_tree/config_cv_train.yaml

# job script for running the Python application
RUN_SH_SCRIPT=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/tess_spoc_ffi/decision_tree/run_cv_iter_modular.sh

# output directory
OUTPUT_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_extractedfeaturesconvbranches_ffi_vs_2min_3-28-2025_1551
mkdir -p $OUTPUT_DIR

N_CV_ITERS=10  # number of CV folds/iterations

# number of total jobs per job in job array
NUM_TOTAL_JOBS=$N_CV_ITERS
# number of jobs run simultaneously
NUM_JOBS_PARALLEL=$N_CV_ITERS

# run CV sh script
# $RUN_SH_SCRIPT 0 $CONFIG_FP $OUTPUT_DIR $N_CV_ITERS
# run CV sh script with GNU parallel
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$RUN_SH_SCRIPT {} $CONFIG_FP $OUTPUT_DIR $N_CV_ITERS"

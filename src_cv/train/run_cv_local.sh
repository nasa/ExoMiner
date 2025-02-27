# Run CV experiment locally.

# codebase root directory
export PYTHONPATH=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/

# config file path
CONFIG_FP=$PYTHONPATH/src_cv/train/config_cv_train.yaml
# job script for running the Python CV iteration script
RUN_SH_SCRIPT=$PYTHONPATH/src_cv/train/run_cv_iter_modular.sh

# output directory
OUTPUT_DIR=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/test_exominerplusplus_cv_tess-spoc-2min_s1-s67_2-25-2025_1135
mkdir -p $OUTPUT_DIR

N_CV_ITERS=1  # number of CV folds/iterations
N_MODELS_PER_CV_ITER=2  # number of models to train per CV iteration

# number of GPUs to be used by this job array
N_GPUS_TOTAL=1

echo "Started CV experiment $OUTPUT_DIR."

# run CV iterations sequentially
for ((CV_ITER=0; CV_ITER<"$N_CV_ITERS"; CV_ITER++))
do
    # run CV iteration sh script
    echo "Started CV iteration $CV_ITER."
    $RUN_SH_SCRIPT $CV_ITER 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_CV_ITERS $N_MODELS_PER_CV_ITER
    echo "Finished CV iteration $CV_ITER."
done

echo "Finished CV experiment $OUTPUT_DIR."

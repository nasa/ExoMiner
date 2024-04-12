# Run CV predict experiment sequentially.

# export path to root of codebase
export PYTHONPATH=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/
# file paths to Python scripts
export SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/setup_cv_iter.py
export PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src_cv/predict_model.py

# config file path
CONFIG_FP=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_cv/config_cv_predict.yaml
# job script for running a single CV iteration
RUN_SH_SCRIPT=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_cv/run_predict_cv_iter_modular.sh
# output directory
OUTPUT_DIR=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_paper/cv_tess_sgdetrending_fluxonly_models_predict_sgdetrendingdata_tic236887394_sg_4-10-2024_1629
# root directory for model file paths
MODELS_CV_ROOT_DIR=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_paper/cv_tess_sgdetrending_fluxonly_4-2-2024_1138
# number of CV iterations to run
N_CV_ITERS=7

mkdir -p $OUTPUT_DIR  # create experiment directory
#LOG_DIR=OUTPUT_DIR/cv_logs  # create subdirectory for logs
#mkdir -p $LOG_DIR

JOB_ARRAY_INDEX=0
N_GPUS=1
for ((CV_ITER=6; CV_ITER<"$N_CV_ITERS"; CV_ITER++))
do
    CV_ITER_DIR="$OUTPUT_DIR"/cv_iter_$CV_ITER
    mkdir -p "$CV_ITER_DIR"
    LOG_FP_CV_ITER=$OUTPUT_DIR/cv_iter_$CV_ITER/cv_run_0_jobarray_0.log
    echo "Started running CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"
    $RUN_SH_SCRIPT "$CV_ITER" $JOB_ARRAY_INDEX $CONFIG_FP $OUTPUT_DIR $N_GPUS $MODELS_CV_ROOT_DIR
    echo "Finished running CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
done

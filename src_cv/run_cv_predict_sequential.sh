# Run CV predict experiment sequentially.

# activate conda environment
source "$HOME"/.bashrc
conda activate exoplnt_dl_tf2_13

# export path to root of codebase
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

# file paths to Python scripts
export SETUP_CV_ITER_FP=$PYTHONPATH/src_cv/setup_cv_iter.py
export PREDICT_MODEL_SCRIPT_FP=$PYTHONPATH/src/predict_model.py

# config file path
CONFIG_FP=$PYTHONPATH/src_cv/config_cv_predict.yaml
# job script for running a single CV iteration
RUN_SH_SCRIPT=$PYTHONPATH/src_cv/run_predict_cv_iter_modular.sh

# script to preprocess data for model inference
PREPROCESS_SCRIPT=$PYTHONPATH/src_cv/create_cv_dataset/preprocess_cv_folds_predict_trecord_dataset.py
CONFIG_PREPROCESS_FP=$PYTHONPATH/src_cv/create_cv_dataset/config_preprocess_cv_folds_predict_tfrecord_dataset.yaml

# script to create dataset filepaths for each CV iteration
CREATE_CV_FOLDS_SCRIPT=$PYTHONPATH/src_cv/create_cv_folds_yaml_from_dir.py

# output directory
OUTPUT_DIR=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/cv_tess-spoc-2min_s1-s67_9-26-2024_1623_predict_unks/
# directory used to store preprocessed data
export DATA_DIR="$OUTPUT_DIR"data
mkdir -p "$DATA_DIR"

# root directory for model file paths
MODELS_CV_ROOT_DIR=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/cv_tess-spoc-2min_s1-s67_9-26-2024_1623/

# number of CV iterations to run
N_CV_ITERS=10

mkdir -p $OUTPUT_DIR  # create experiment directory

JOB_ARRAY_INDEX=0
N_GPUS=4
for ((CV_ITER=1; CV_ITER<"$N_CV_ITERS"; CV_ITER++))
do
    CV_ITER_DIR="$OUTPUT_DIR"/cv_iter_$CV_ITER
    mkdir -p "$CV_ITER_DIR"
    LOG_FP_CV_ITER=$OUTPUT_DIR/cv_iter_$CV_ITER/cv_run_0_jobarray_0.log

    echo "Started running CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"

    echo "Started preprocessing data for CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"
    python "$PREPROCESS_SCRIPT" --rank=$CV_ITER --config_fp=$CONFIG_PREPROCESS_FP --output_dir="$DATA_DIR"
    python "$CREATE_CV_FOLDS_SCRIPT" --config_fp=$CONFIG_FP --data_dir="$DATA_DIR"/cv_iter_$CV_ITER --cv_iter=$CV_ITER &> "$LOG_FP_CV_ITER"
    CONFIG_CV_FP=$DATA_DIR/cv_iter_$CV_ITER/config_cv.yaml
    echo "Finished preprocessing data for CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"

    echo "Started running inference on data using models from CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"
    $RUN_SH_SCRIPT "$CV_ITER" $JOB_ARRAY_INDEX $CONFIG_CV_FP $OUTPUT_DIR $N_GPUS $MODELS_CV_ROOT_DIR
    echo "Finished running inference on data using models from CV iteration $CV_ITER." > "$LOG_FP_CV_ITER"

    # delete processed data
    rm -r "$DATA_DIR"/cv_iter_$CV_ITER

    echo "Finished running CV iteration $CV_ITER." >> "$LOG_FP_CV_ITER"
done

# Conduct preprocessing run that creates a TFRecord dataset for CV.

export PYTHONPATH=codebase

# create output directory for preprocessing results
OUTPUT_DIR=
mkdir -p $OUTPUT_DIR

# script file path
SCRIPT_FP=$PYTHONPATH/src_cv/create_cv_dataset/preprocess_cv_folds_trecord_dataset.py
# config file path
CONFIG_FP=$PYTHONPATH/src_cv/create_cv_dataset/config_preprocess_cv_folds_tfrecord_dataset.yaml
# job script for running preprocessing pipeline
PREPROCESS_SH_SCRIPT=$PYTHONPATH/src_cv/create_cv_dataset/preprocessing_job.sh

# number of folds in the CV experiment
N_CV_FOLDS=5

for ((CV_FOLD_I=0; CV_FOLD_I<"$N_CV_FOLDS"; CV_FOLD_I++))
do
    $PREPROCESS_SH_SCRIPT "$CV_FOLD_I" $OUTPUT_DIR $CONFIG_FP $N_CV_FOLDS
done

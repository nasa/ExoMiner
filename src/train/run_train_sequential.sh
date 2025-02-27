# Run one or more train model iterations sequentially.

# config file path
CONFIG_FP=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src/config_train.yaml
# job script for running the Python application
RUN_SH_SCRIPT=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src/run_train_iter.sh
# output directory
OUTPUT_DIR=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/test_train_eval_test_bds_vs_planets_7-2-2024_1540
# path to codebase root directory
#export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/
export PYTHONPATH=/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/

#source "$HOME"/.bashrc
source "$HOME"/.zshrc

conda activate exoplnt_dl
#conda activate exoplnt_dl_tf2_13

mkdir -p $OUTPUT_DIR

N_GPUS_TOTAL=1  # number of GPUs available on the node/system

N_MODELS=2  # number of models

for ((MODEL_I=0; MODEL_I<"$N_MODELS"; MODEL_I++))
do
    $RUN_SH_SCRIPT "$MODEL_I" 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_MODELS
done

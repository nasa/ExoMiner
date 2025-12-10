# Run one or more train model iterations sequentially.

# config file path
CONFIG_FP=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src/train/config_train.yaml
# job script for running the Python application
RUN_SH_SCRIPT=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src/train/run_train_iter.sh
# output directory
OUTPUT_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/test_exominer_architectures/test_tess-spoc-2min-s1-s88_10-22-2025_1616

# path to codebase root directory
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

source "$HOME"/.bashrc
# source "$HOME"/.zshrc

# conda activate exoplnt_dl
conda activate exoplnt_dl_tf2_13

mkdir -p $OUTPUT_DIR

N_GPUS_TOTAL=1  # number of GPUs available on the node/system

N_MODELS=1  # number of models

for ((MODEL_I=0; MODEL_I<"$N_MODELS"; MODEL_I++))
do
    $RUN_SH_SCRIPT "$MODEL_I" 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_MODELS
done

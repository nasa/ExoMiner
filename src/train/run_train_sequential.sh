# Run one or more train model iterations sequentially.

# config file path
CONFIG_FP=codebase/src/config_train.yaml
# job script for running the Python application
RUN_SH_SCRIPT=codebase/src/run_train_iter.sh
# output directory
OUTPUT_DIR=
# path to codebase root directory
export PYTHONPATH=codebase/

conda activate exoplnt_dl

mkdir -p $OUTPUT_DIR

N_GPUS_TOTAL=1  # number of GPUs available on the node/system

N_MODELS=2  # number of models

for ((MODEL_I=0; MODEL_I<"$N_MODELS"; MODEL_I++))
do
    $RUN_SH_SCRIPT "$MODEL_I" 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_MODELS
done

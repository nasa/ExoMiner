CONFIG_FP=/users/agiri1/Desktop/ExoPlanet/src/config_train.yaml
SCRIPT=/users/agiri1/Desktop/ExoPlanet/src/run_train_iter.sh
OUTPUT_DIR=/users/agiri1/Desktop/ExoPlanet/output_dir
export PYTHONPATH=/Users/agiri1/Desktop/ExoPlanet

source "$HOME"/.zshrc

conda activate exolanet_env

mkdir -p OUTPUT_DIR

N_GPUS=1
N_MODELS=1

$SCRIPT 0 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS $N_MODELS
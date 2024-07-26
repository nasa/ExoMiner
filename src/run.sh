CONFIG_FP=/users/agiri1/Desktop/ExoPlanet/src/config_train.yaml
SCRIPT=/users/agiri1/Desktop/ExoPlanet/src/run_train_iter.sh
CHANGE_SCRIPT=/Users/agiri1/Desktop/ExoPlanet/RandomScripts/ChangeYAML.py
OUTPUT_DIR=/users/agiri1/Desktop/ExoPlanet/output_dir
export PYTHONPATH=/Users/agiri1/Desktop/ExoPlanet

source "$HOME"/.zshrc

conda activate exoplanet_env

mkdir -p OUTPUT_DIR
# changed to pass correct yaml file to each cv_iteration
N_GPUS=1
N_MODELS=5
MODEL_I=0
for ((MODEL_I=0; MODEL_I<"$N_MODELS"; MODEL_I++))
do
    echo $MODEL_I
    python "$CHANGE_SCRIPT" --current_dataset_fp="/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_$MODEL_I/dataset_fps.yaml"
    $SCRIPT $MODEL_I 0 $CONFIG_FP $OUTPUT_DIR $N_GPUS $N_MODELS
done

# Run worker for HPO run
# Args:
# $1: GNU parallel index
# $2: File path to configuration yaml file for the run
# $3: Path to output directory
# $4: Number of GPUs per node

# External arguments
OUTPUT_DIR="$3"
CONFIG_FP="$2"
N_GPUS_PER_NODE="$4"
GNU_PARALLEL_INDEX="$1"

# initialize conda
source "$HOME"/.bashrc

# activate conda environment
conda activate exoplnt_dl_tf2_13

# set path to codebase root directory
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase

HPO_PY_SCRIPT=$PYTHONPATH/src_hpo/run_hpo.py

# set CV iteration ID
#WORKER_ID=$(($GNU_PARALLEL_INDEX + $N_GPUS_PER_NODE))
WORKER_ID=$GNU_PARALLEL_INDEX

LOG_WORKER="$OUTPUT_DIR"/worker_"$WORKER_ID"_stdout.log

echo "Starting HPO run on worker $WORKER_ID..." > "$LOG_WORKER"

if [ "$WORKER_ID" != 0 ]
then
  echo "Waiting for worker on master node to start HPO run."
  sleep 30
fi

GPU_ID=$(("$WORKER_ID" % $N_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set visible GPUs to $CUDA_VISIBLE_DEVICES." >> "$LOG_WORKER"

PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
until [ "$PROC_IN_GPU" == "" ]
do
    PROC_IN_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader)
    echo "Current process in GPU $GPU_ID: $PROC_IN_GPU"
    sleep 60
done

echo "GPU $GPU_ID is available. Resuming HPO worker iteration." >> "$LOG_WORKER"

python $HPO_PY_SCRIPT --config_file="$CONFIG_FP" --output_dir="$OUTPUT_DIR" --worker_id="$WORKER_ID" &>> "$LOG_WORKER"

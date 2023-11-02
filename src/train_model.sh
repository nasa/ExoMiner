# Run job for training a model
# Args:
# $1: GNU parallel index
# $2: Job array index
# $3: File path to Python script
# $4: File path to configuration yaml file for the run
# $5: Path to output directory

source "$HOME"/.bashrc

conda activate exoplnt_dl

export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

MODEL_RUN=$(($1 + $2))

echo "Starting job $1 in job array $2 for model $MODEL_RUN..."

# run Python script for training a single model
python "$3" --rank="$1" --config_fp="$4" --output_dir="$5" &> "$5"/train_run_"$1"_jobarray_"$2".log

echo "Finished job $1 in job array $2 for model $MODEL_RUN..."

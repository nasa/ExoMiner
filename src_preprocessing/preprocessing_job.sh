# Run job preprocessing pipeline
# Args:
# $1: Rank of the job
# $2: Path to output directory for the preprocessing run
# $3: File path to main preprocessing Python script
# $4: File path to configuration yaml file for the preprocessing run
# $5: Total number of jobs

source "$HOME"/.bashrc

conda activate exoplnt_dl

export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/

echo "Starting job $1..."

# run preprocessing pipeline
python $3 --rank=$1 --n_runs=$5 --output_dir=$2 --config_fp=$4 &> $2/preprocessing_$1.log

echo "Finished job $1..."

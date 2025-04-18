# Conduct ephemeris matching run.

# activte conda environment
source activate envname

# codebase root directory
export PYTHONPATH=

# create output directory for preprocessing results
OUTPUT_DIR=
mkdir -p "$OUTPUT_DIR"

# script file path
SCRIPT_FP=$PYTHONPATH/src_preprocessing/ephemeris_matching/run_ephemeris_match_multiproc.py
# config file path
CONFIG_FP=$PYTHONPATH/src_preprocessing/ephemeris_matching/config_ephem_matching.yaml

LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR
# log file
LOG_FP=$LOG_DIR/ephem_matching_stdout.log

python $SCRIPT_FP --config_fp=$CONFIG_FP --output_dir="$OUTPUT_DIR" &> "$LOG_FP"

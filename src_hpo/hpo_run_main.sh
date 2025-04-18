# Conduct HPO run multi-GPU and multi-node; using MPIEXEC to evaluate multiple configurations (one per GPU) at the same
# time.

# config file path
CONFIG_FP=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_hpo/config_hpo.yaml
# job script for running the Python application
RUN_SH_SCRIPT=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_hpo/run_hpo_worker.sh
# output directory
OUTPUT_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_paper/hpo_runs/hpo_run_keplerq1q17dr25_11-22-2024_1703
mkdir -p $OUTPUT_DIR

# number of GPUs per node
N_GPUS_TOTAL=4

# number of total jobs per job in job array
NUM_TOTAL_JOBS=4
# number of jobs run simultaneously
NUM_JOBS_PARALLEL=2

# run with GNU parallel
seq 0 $((NUM_TOTAL_JOBS - 1)) | parallel -j $NUM_JOBS_PARALLEL "$RUN_SH_SCRIPT {} $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL"

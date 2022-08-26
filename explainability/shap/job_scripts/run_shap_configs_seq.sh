# Conduct training/predict runs for multiple configurations.

# txt file that contains in each line the name of the config file to be used in each run
CONFIG_FILE_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/shap_tessvkepler/shap_08-25-2022_1619/
CONFIGS_FILE="$CONFIG_FILE_DIR"list_config_runs_onlyonesthatfailed_8-26-2022_1010.txt
# CONFIGS_FILE=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/shap_tessvkepler/shap_08-25-2022_1619/list_config_runs.txt
#SUBMIT_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single.pbs
#SUBMIT_TRAIN_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_train.pbs
#SUBMIT_PRED_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs
# TRAIN_SCRIPT=/home/msaragoc/Projects/exoplnt_dl/codebase/src/train_keras.py
# PRED_SCRIPT=/home/msaragoc/Projects/exoplnt_dl/codebase/src/predict_ensemble_keras.py
# CONFIG_FILE_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/shap_tessvkepler/shap_08-25-2022_1619/runs_configs/
# PBS script to use to launch a job for each run
SUBMIT_SCRIPT=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/explainability/shap/job_scripts/Submitjob_shap_single_train.pbs
# PYOUT_DIR=/data5/tess_project/experiments/shap_tessvkepler/shap_08-25-2022_1255/log_py/

# # sequential
# cat "$CONFIGS_FILE" | while read line
# do
#     echo Running config "$line"
#     python "$TRAIN_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_train.yaml
#     python "$PRED_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_predict.yaml
# done


# training runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_TRAIN_SCRIPT"; done

# predict runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_PRED_SCRIPT"; done

## HECC
N_JOBS_ALLOWED_SIMULT=2  # number of jobs allowed in queue/running

cat "$CONFIGS_FILE" | while read line
do
    N_JOBS_IN_Q=$(($(qstat @pbspl4 -u msaragoc | wc -l)-3))  # check number of jobs running/in queue
    # wait if number of jobs running/in queue is equal to number of jobs allowed
    while [ $N_JOBS_IN_Q -eq $N_JOBS_ALLOWED_SIMULT ]
    do
        sleep 8m
        N_JOBS_IN_Q=$(($(qstat @pbspl4 -u msaragoc | wc -l)-3))
    done

    echo Running config "$line"
    qsub -v SHAP_CONFIG_RUN="$line",CONFIG_FILE_DIR="$CONFIG_FILE_DIR" "$SUBMIT_SCRIPT"
    sleep 10s
done

# # local system with more than one GPU
# N_JOBS_ALLOWED_SIMULT=1
# cat "$CONFIGS_FILE" | while read line
# do
#     # N_PYTHON_PROCS=$(top -n 1 -p `pgrep "python"` | wc -l)
#     N_PYTHON_PROCS=$(pgrep -lf python -u msaragoc | wc -l)
#     # echo ahaha $N_PYTHON_PROCS
#     # wait if number of jobs running/in queue is equal to number of jobs allowed
#     while [ $N_PYTHON_PROCS -eq $N_JOBS_ALLOWED_SIMULT ]
#     do
#         sleep 8m
#         # N_PYTHON_PROCS=$(top -n 1 -p `pgrep "python"` | wc -l)
#         N_PYTHON_PROCS=$(pgrep -lf python -u msaragoc | wc -l)
#     done
# 
#     echo Running config "$line"
#     # GPU_ID=$(($N_PYTHON_PROCS - 1))
#     GPU_ID=$N_PYTHON_PROCS
#     (echo Training models for $line; python "$TRAIN_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_train.yaml --job_idx="$GPU_ID" &> "$PYOUT_DIR$line"_train_log_py.txt; echo Running inference using ensemble for $line; python "$PRED_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_predict.yaml &> "$PYOUT_DIR$line"_predict_log_py.txt; echo Finished run for config "$line") &
#     sleep 10s
# done

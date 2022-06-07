# Conduct training/predict runs for multiple configurations.

# txt file that contains in each line the name of the config file to be used in each run
CONFIGS_FILE=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/shap/shap_1-18-2022/list_config_runs.txt
#SUBMIT_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single.pbs
#SUBMIT_TRAIN_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_train.pbs
#SUBMIT_PRED_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs
# PBS script to use to launch a job for each run
SUBMIT_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs

# training runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_TRAIN_SCRIPT"; done

# predict runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_PRED_SCRIPT"; done

N_JOBS_ALLOWED_SIMULT=1  # number of jobs allowed in queue/running

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
    qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_SCRIPT"
    sleep 10s
done


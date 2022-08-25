# Conduct training/predict runs for multiple configurations.

# txt file that contains in each line the name of the config file to be used in each run
CONFIGS_FILE=/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/shap_tessvkepler/shap_08-24-2022_1457/list_config_runs.txt
#SUBMIT_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single.pbs
#SUBMIT_TRAIN_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_train.pbs
#SUBMIT_PRED_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs
TRAIN_SCRIPT=/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/src/train_keras.py
PRED_SCRIPT=/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/src/predict_ensemble_keras.py
CONFIG_FILE_DIR=/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/shap_tessvkepler/shap_08-24-2022_1457/runs_configs/
# PBS script to use to launch a job for each run
SUBMIT_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs

# sequential
cat "$CONFIGS_FILE" | while read line
do
    echo Running config "$line"
    python "$TRAIN_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_train.yaml
    python "$PRED_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_predict.yaml
done


# training runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_TRAIN_SCRIPT"; done

# predict runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_PRED_SCRIPT"; done

#N_JOBS_ALLOWED_SIMULT=2  # number of jobs allowed in queue/running
#
#cat "$CONFIGS_FILE" | while read line
#do
#    N_JOBS_IN_Q=$(($(qstat @pbspl4 -u msaragoc | wc -l)-3))  # check number of jobs running/in queue
#    # wait if number of jobs running/in queue is equal to number of jobs allowed
#    while [ $N_JOBS_IN_Q -eq $N_JOBS_ALLOWED_SIMULT ]
#    do
#        sleep 8m
#        N_JOBS_IN_Q=$(($(qstat @pbspl4 -u msaragoc | wc -l)-3))
#    done
#
#    echo Running config "$line"
#    qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_SCRIPT"
#    sleep 10s
#done

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

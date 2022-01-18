# Conduct training/predict runs for multiple configurations 

CONFIGS_FILE=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/shap/shap_1-11-2022/list_config_runs.txt
SUBMIT_TRAIN_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_train.pbs
SUBMIT_PRED_SCRIPT=/home6/msaragoc/job_scripts/Kepler-TESS_exoplanet/Submitjob_shap_single_pred.pbs

# training runs
# cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_TRAIN_SCRIPT"; done

# predict runs
cat  "$CONFIGS_FILE" | while read line; do echo Running config "$line"; qsub -v SHAP_CONFIG_RUN="$line" "$SUBMIT_PRED_SCRIPT"; done

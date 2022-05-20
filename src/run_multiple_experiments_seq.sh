# Run multiple prediction runs sequentially based on  different YAML configuration files

CONFIG_FILE_DIR=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/experiments/run_configs/explainability_shap_1-11-2022/runs/
CONFIGS_FILE=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/experiments/run_configs/explainability_shap_1-11-2022/list_config_runs.txt
TRAIN_SCRIPT=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/codebase/src/train_keras.py
PREDICT_ENSEMBLE_SCRIPT=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/codebase/src/predict_ensemble_keras.py

#conda activate exoplnt_dl

echo "$CONFIG_FILE_DIR"
echo "$CONFIGS_FILE"
echo "$TRAIN_SCRIPT"
echo "$PREDICT_ENSEMBLE_SCRIPT"

#for config_file in "$CONFIG_FILE_DIR"*; do echo "$config_file"; done
#for config_file in "$CONFIG_FILE_DIR"*; do echo "$config_file"; python "$TRAIN_SCRIPT" --config_file="$config_file" --job_idx=0; python "$PREDICT_ENSEMBLE_SCRIPT" --config_file="$config_file"; done
#cat "$CONFIGS_FILE" | while read line; do echo "$CONFIG_FILE_DIR$line"_pred.yaml;done
cat "$CONFIGS_FILE" | while read line; do echo Running "$line"; python "$TRAIN_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_train.yaml --job_idx=0; python "$PREDICT_ENSEMBLE_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_predict.yaml; done
#cat "$CONFIGS_FILE" | while read line; do echo Running "$line"; python "$PREDICT_ENSEMBLE_SCRIPT" --config_file="$CONFIG_FILE_DIR$line"_predict.yaml; done


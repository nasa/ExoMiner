# Run multiple prediction runs sequentially based on  different YAML configuration files

CONFIG_FILE_DIR=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/experiments/run_configs/explainability_shap_1-10-2022/
PREDICT_ENSEMBLE_SCRIPT=/Users/msaragoc/OneDrive\ -\ NASA/Projects/exoplanet_transit_classification/codebase/src/predict_ensemble_keras.py

#conda activate exoplnt_dl

echo "$CONFIG_FILE_DIR"
echo "$PREDICT_ENSEMBLE_SCRIPT"

#for config_file in "$CONFIG_FILE_DIR"*; do echo "$config_file"; done
for config_file in "$CONFIG_FILE_DIR"*; do echo "$config_file"; python "$PREDICT_ENSEMBLE_SCRIPT" --config_file="$config_file"; done

# Run multiple label noise runs sequentially based on different YAML configuration files

CONFIG_FILE_DIR=/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-11-2022_1144/configs/
CONFIGS_FILE=/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-11-2022_1144/config_runs.txt
SCRIPT=/home/msaragoc/Projects//Kepler-TESS_exoplanet/codebase/aum/train_keras.py
#SCRIPT=/home/msaragoc/Projects//Kepler-TESS_exoplanet/codebase/aum/predict_keras.py

#conda activate exoplnt_dl

cat "$CONFIGS_FILE" | while read line
do
  echo Running "$line"
  python "$SCRIPT" --config_file="$CONFIG_FILE_DIR$line" --job_idx=0
done

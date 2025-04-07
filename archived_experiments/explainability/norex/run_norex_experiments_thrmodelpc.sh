# Run several NOREX experiments for different threshold values for selecting model PCs.

source ~/.bashrc
conda activate exoplnt_dl

export PYTHONPATH=~/Projects/exoplnt_dl/codebase/

SCRIPT_FP=~/Projects/exoplnt_dl/codebase/explainability/run_exp_replacing_pc.py
CONFIG_FP=~/Projects/exoplnt_dl/codebase/explainability/config_replacing_pc.yaml

for THR_MODELPCS_I in {0..13}
do
  echo "Starting run for model PC threshold $THR_MODELPCS_I..."
  python $SCRIPT_FP --config_file=$CONFIG_FP --thr_modelpcs=$THR_MODELPCS_I &> ~/Projects/exoplnt_dl/experiments/explainability/model_pc_replacement/pylog_thr_$THR_MODELPCS_I.txt &
done

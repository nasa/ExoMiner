# Run several NOREX experiments for different numbers of model PCs.

source ~/.bashrc
conda activate exoplnt_dl

export PYTHONPATH=~/Projects/exoplnt_dl/codebase/

SCRIPT_FP=~/Projects/exoplnt_dl/codebase/explainability/run_exp_replacing_pc.py
CONFIG_FP=~/Projects/exoplnt_dl/codebase/explainability/config_replacing_pc.yaml

for NUM_PCS in {1..15}
do
  echo "Starting run for $NUM_PCS..."
  python $SCRIPT_FP --config_file=$CONFIG_FP --num_pcs=$NUM_PCS &> ~/Projects/exoplnt_dl/experiments/explainability/model_pc_replacement/pylog_$NUM_PCS.txt &
done

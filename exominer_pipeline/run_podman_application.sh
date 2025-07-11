
# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/Users/miguelmartinho/Projects/test_exominer_devel/inputs"
# file path to the TICs table
tics_tbl_fp=$inputs_dir/tics_tbl.csv
# file path to the configuration file for the ExoMiner Pipeline run
config_fp=$inputs_dir/pipeline_run_config.yaml
# name of the run
exominer_pipeline_run=exominer_pipeline_run_7-10-2025_1925
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir="/Users/miguelmartinho/Projects/test_exominer_devel/runs/$exominer_pipeline_run"

mkdir -p $exominer_pipeline_run_dir

echo "Started ExoMiner Pipeline run $exominer_pipeline_run..."
echo "Running ExoMiner Pipeline with the following parameters:"
echo "Inputs directory: $inputs_dir"
echo "TICs table file: $tics_tbl_fp"
echo "Configuration file: $config_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

podman run \
  -v $inputs_dir:/inputs:Z \
  -v $exominer_pipeline_run_dir:/outputs:Z \
  exominer_pipeline \
  --tic_ids_fp=/inputs/tics_tbl.csv \
  --output_dir=/outputs/$exominer_pipeline_run \
  --config_fp=/inputs/pipeline_run_config.yaml

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run."

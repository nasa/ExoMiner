
# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs"
# file path to the TICs table
tics_tbl_fp=$inputs_dir/tics_tbl.csv
# name of the run
exominer_pipeline_run=exominer_pipeline_run_7-16-2025_1357
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/$exominer_pipeline_run"
# data collection mode: either 2min or ffi
data_collection_mode=2min
# number of processes
num_processes=1
# number of jobs to split the TIC IDs
num_jobs=1
# set to "true" or "false". If "true", it will create a CSV file with URLs to the SPOC DV reports for each TCE in the
# queried TICs
download_spoc_data_products=true
# path to a directory containing the light curve FITS files and DV XML files for the TIC IDs and sector runs that you
# want to query; set to "null" otherwise
external_data_repository=null

mkdir -p $exominer_pipeline_run_dir

echo "Started ExoMiner Pipeline run $exominer_pipeline_run..."
echo "Running ExoMiner Pipeline with the following parameters:"
echo "Inputs directory: $inputs_dir"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

podman run \
  -v $inputs_dir:/inputs:Z \
  -v $exominer_pipeline_run_dir:/outputs:Z \
   d8c467fe1966 \
  --tic_ids_fp=/inputs/tics_tbl.csv \
  --output_dir=/outputs \
  --data_collection_mode=$data_collection_mode \
  --num_processes=$num_processes \
  --num_jobs=$num_jobs \
  --download_spoc_data_products=$download_spoc_data_products \
  --external_data_repository=$external_data_repository \

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run."

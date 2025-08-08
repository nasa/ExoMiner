### Run ExoMiner Pipeline by running a container of a Podman image

# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs"
# file path to the TICs table
tics_tbl_fn=tics_tbl_356473034_S60.csv
tics_tbl_fp=$inputs_dir/$tics_tbl_fn
# name of the run
exominer_pipeline_run=exominer_pipeline_run_7-19-2025_1304
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
# define source of stellar parameters for TICs. If set to 'ticv8', TIC-8 is queried; if set to 'tess-spoc', it uses the
# parameters stored in the TICs DV XML files; if set to a filepath that points to an external catalog of stellar
# parameters, it will use those values.
stellar_parameters_source=ticv8
# define source of Gaia RUWE for TICs. If set to 'gaiadr2', Gaia DR2 is queried; if set to 'unavailable', it assumes the
# values are missing; if set to a filepath that points to an external catalog of RUWE parameters, it will use those
# values.
ruwe_source=gaiadr2

mkdir -p $exominer_pipeline_run_dir

# set up volume mounts
volume_mounts="-v $inputs_dir:/inputs:Z -v $exominer_pipeline_run_dir:/outputs:Z"

# conditionally add external_data_repository mount
if [ "$external_data_repository" != "null" ]; then
  volume_mounts="$volume_mounts -v $external_data_repository:/external_data_repository:Z"
  external_data_repository_arg="--external_data_repository=/external_data_repository"
else
  external_data_repository_arg=""
fi

# add mount to external TICs stellar parameters catalog if filepath provided
if [ -f "$stellar_parameters_source" ]; then
    volume_mounts="$volume_mounts -v $stellar_parameters_source:/tics_stellar_parameters.csv:Z"
    stellar_parameters_source_arg=/tics_stellar_parameters.csv
else
    stellar_parameters_source_arg=$stellar_parameters_source
fi

# add mount to external TICs RUWE catalog if filepath provided
if [ -f "$ruwe_source" ]; then
    volume_mounts="$volume_mounts -v $ruwe_source:/tics_ruwe.csv:Z"
    ruwe_source_arg=/tics_ruwe.csv
else
    ruwe_source_arg=$ruwe_source
fi

echo "Started ExoMiner Pipeline run $exominer_pipeline_run..."
echo "Running ExoMiner Pipeline with the following parameters:"
echo "Inputs directory: $inputs_dir"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

podman run \
  ${volume_mounts} \
   ghcr.io/nasa/exominer  \
  --tic_ids_fp=/inputs/$tics_tbl_fn \
  --output_dir=/outputs \
  --data_collection_mode=$data_collection_mode \
  --num_processes=$num_processes \
  --num_jobs=$num_jobs \
  --download_spoc_data_products=$download_spoc_data_products \
  --stellar_parameters_source=$stellar_parameters_source_arg \
  --ruwe_source=$ruwe_source_arg \
  $external_data_repository_arg \

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run."
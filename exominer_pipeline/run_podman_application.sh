#!/bin/bash

### Run ExoMiner Pipeline by running a container of a Podman image

### default values ## 

# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs/"
# file path to the TICs table
tics_tbl_fn="tics_tbl_356473034_S60.csv"
tics_tbl_fp=$inputs_dir/$tics_tbl_fn
# name of the run
exominer_pipeline_run=test_tic2356473034-S60_exominer-single_model-external_10-9-2025_0952
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/$exominer_pipeline_run
# data collection mode: either 2min or ffi
data_collection_mode="2min"
# number of processes
num_processes=1
# number of jobs to split the TIC IDs
num_jobs=1
# set to "true" or "false". If "true", it will create a CSV file with URLs to the SPOC DV reports for each TCE in the
# queried TICs
download_spoc_data_products="false"
# path to a directory containing the light curve FITS files and DV XML files for the TIC IDs and sector runs that you
# want to query; set to "null" otherwise
external_data_repository="null"  # "/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/test_tic2356473034-S60_exominer-single_model-external_10-8-2025_2143/job_0/mastDownload"  # null  # /Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/test_tic235678745-s14s86_exominer-single_10-8-2025_1328/job_0/mastDownload/
# define source of stellar parameters for TICs. If set to 'ticv8', TIC-8 is queried; if set to 'tess-spoc', it uses the
# parameters stored in the TICs DV XML files; if set to a filepath that points to an external catalog of stellar
# parameters, it will use those values.
stellar_parameters_source=ticv8
# define source of Gaia RUWE for TICs. If set to 'gaiadr2', Gaia DR2 is queried; if set to 'unavailable', it assumes the
# values are missing; if set to a filepath that points to an external catalog of RUWE parameters, it will use those
# values.
ruwe_source=gaiadr2
# which ExoMiner model to use for inference. Choose between "exominer++_single", "exominer++_cviter-mean-ensemble", and "exominer++_cv-super-mean-ensemble".
# or provide a filepath to a custom ExoMiner model (in Keras .keras format)
exominer_model="/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/data/exominer-plusplus_cv-iter0-model0_tess-spoc-2min-s1s67_tess-kepler.keras"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --tics_tbl_fp FILE                   TIC IDs table filepath"
    echo "  --exominer_pipeline_run_dir DIR      Directory to store pipeline run output"
    echo "  --data_collection_mode MODE          Data collection mode (2min or ffi)"
    echo "  --num_processes N                    Number of processes"
    echo "  --num_jobs N                         Number of jobs"
    echo "  --download_spoc_data_products BOOL   Whether to download TESS SPOC DV reports for the detected TCEs: true or false"
    echo "  --external_data_repository DIR       Path to external data repository containing light curve FITS files and DV XML files for the TIC IDs in the TICs table"
    echo "  --stellar_parameters_source SOURCE   Source for TICs stellar parameters"
    echo "  --ruwe_source SOURCE                 Source for TICs Gaia RUWE parameters"
    echo "  --exominer_model MODEL               ExoMiner model to use for inference"
    echo "  --help                               Show ExoMiner Pipeline help"
    echo ""
    exit 
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tics_tbl_fp) tics_tbl_fp="$2"; shift 2 ;;
        --exominer_pipeline_run_dir) exominer_pipeline_run_dir="$2"; shift 2 ;;
        --data_collection_mode) data_collection_mode="$2"; shift 2 ;;
        --num_processes) num_processes="$2"; shift 2 ;;
        --num_jobs) num_jobs="$2"; shift 2 ;;
        --download_spoc_data_products) download_spoc_data_products="$2"; shift 2 ;;
        --external_data_repository) external_data_repository="$2"; shift 2 ;;
        --stellar_parameters_source) stellar_parameters_source="$2"; shift 2 ;;
        --ruwe_source) ruwe_source="$2"; shift 2 ;;
        --exominer_model) exominer_model="$2"; shift 2 ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

mkdir -p $exominer_pipeline_run_dir

# Save parameters to a file inside the run directory
echo "Saving run parameters to $exominer_pipeline_run_dir/run_parameters.txt"

params_file="$exominer_pipeline_run_dir/run_parameters.txt"

cat <<EOF > "$params_file"
TICs table file: $tics_tbl_fp
ExoMiner Pipeline run directory: $exominer_pipeline_run_dir
Data collection mode: $data_collection_mode
Number of processes: $num_processes
Number of jobs: $num_jobs
Download SPOC data products: $download_spoc_data_products
External data repository: $external_data_repository
Stellar parameters source: $stellar_parameters_source
RUWE source: $ruwe_source
ExoMiner model: $exominer_model
Image revision: $(podman inspect ghcr.io/nasa/exominer:latest --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')
EOF

# set up volume mounts
volume_mounts="-v $tics_tbl_fp:/tics_tbl.csv:Z -v $exominer_pipeline_run_dir:/outputs:Z"

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

# handle custom model path
if [[ "$exominer_model" != "exominer++_single" && "$exominer_model" != "exominer++_cviter-mean-ensemble" && "$exominer_model" != "exominer++_cv-super-mean-ensemble" ]]; then
    if [[ -f "$exominer_model" ]]; then
        volume_mounts="$volume_mounts -v $exominer_model:/custom_model.keras:Z"
        exominer_model_arg="/custom_model.keras"
    else
        echo "Error: Provided exominer_model path '$exominer_model' does not exist or is not a file."
        exit 1
    fi
else
    exominer_model_arg="$exominer_model"
fi

echo "Running ExoMiner Pipeline with the following parameters:"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

podman run \
  ${volume_mounts} \
  ghcr.io/nasa/exominer \
  --tic_ids_fp=/tics_tbl.csv \
  --output_dir=/outputs \
  --data_collection_mode=$data_collection_mode \
  --num_processes=$num_processes \
  --num_jobs=$num_jobs \
  --download_spoc_data_products=$download_spoc_data_products \
  --stellar_parameters_source=$stellar_parameters_source_arg \
  --ruwe_source=$ruwe_source_arg \
  --exominer_model=$exominer_model_arg \
  $external_data_repository_arg \

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run_dir."

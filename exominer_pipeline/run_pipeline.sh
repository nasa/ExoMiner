#!/bin/bash

### Run ExoMiner Pipeline by running the Python application

### default values ## 

# python pipeline python script
pipeline_python_script="/data3/exoplnt_dl/codebase/exominer_pipeline/exominer_pipeline.py"
# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/data3/exoplnt_dl/experiments/exominer_pipeline/inputs"
# file path to the TICs table
tics_tbl_fn="tics_S72-72.csv"
tics_tbl_fp=$inputs_dir/$tics_tbl_fn
# name of the run
exominer_pipeline_run=test_exominer_pipeline_run_8-19-2025_1753
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir=/data3/exoplnt_dl/experiments/exominer_pipeline/runs/$exominer_pipeline_run
# data collection mode: either 2min or ffi
data_collection_mode="2min"
# number of processes
num_processes=1
# number of jobs to split the TIC IDs
num_jobs=1
# set to "true" or "false". If "true", it will create a CSV file with URLs to the SPOC DV reports for each TCE in the
# queried TICs
download_spoc_data_products=false
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
# which ExoMiner model to use for inference. Choose between "exominer++_single", "exominer++_cviter-mean-ensemble", and "exominer++_cv-super-mean-ensemble".
exominer_model="exominer++_single"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --pipeline_python_script_fp FILE     Pipeline Python main script filepath"
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
        --pipeline_python_script_fp) pipeline_python_script="$2"; shift 2 ;;
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

mkdir -p "$exominer_pipeline_run_dir"

echo "Running ExoMiner Pipeline with the following parameters:"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

# conditionally add external_data_repository mount
if [ "$external_data_repository" != "null" ]; then
  external_data_repository_arg="--external_data_repository=/external_data_repository"
else
  external_data_repository_arg=""
fi

python "$pipeline_python_script" \
  --tic_ids_fp="$tics_tbl_fp" \
  --output_dir="$exominer_pipeline_run_dir" \
  --data_collection_mode="$data_collection_mode" \
  --num_processes="$num_processes" \
  --num_jobs="$num_jobs" \
  --download_spoc_data_products="$download_spoc_data_products" \
  --stellar_parameters_source="$stellar_parameters_source" \
  --ruwe_source="$ruwe_source" \
  --exominer_model=$exominer_model \
  $external_data_repository_arg \

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run_dir."

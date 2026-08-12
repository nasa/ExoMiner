#!/bin/bash

### Run ExoMiner Pipeline either via the Python application or Podman

### default values ##

# execution runner ('python' for Python or 'podman' for Podman application)
runner="podman"
podman_img="localhost/exominer:latest"  # "ghcr.io/nasa/exominer"

# python pipeline python script (used if runner=python)
pipeline_python_script=/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/run_pipeline.py
# path to TIC IDs input table
tics_tbl_fp=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs/test_tics_spoc-ffi_6-19-2026.csv
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir=/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/test_exominer-pipeline_planet-validation_ffi_7-2-2026_1501
# data collection mode: either 2min or ffi
data_collection_mode="ffi"
# number of processes used for preprocessing parallelization
num_processes=3
# number of jobs to split the TIC IDs for preprocessing
num_jobs=3
# set to "true" or "false". If "true", it will create a CSV file with URLs to the SPOC DV reports for each TCE in the
# queried TICs
get_mast_urls_dv_reports="false"
# path to a directory containing the light curve FITS files and DV XML files for the TIC IDs and sector runs that you
# want to query; set to "null" otherwise
dv_xml_data_repository="null" #"/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/data/lc_dv_data" 
lc_data_repository="null" # "/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/data/lc_dv_data"
# define source of stellar parameters for TICs. If set to 'ticv8', TIC-8 is queried; if set to 'tess-spoc', it uses the
# parameters stored in the TICs DV XML files; if set to a filepath that points to an external catalog of stellar
# parameters, it will use those values.
stellar_parameters_source="ticv8" #"/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/data/source_catalogs/tic8_results.csv"
# define source of Gaia RUWE for TICs. If set to 'gaiadr2', 'gaiadr3', or 'gaiaedr3', Gaia DR2, DR3, or EDR3, respectively is queried; if set to 'unavailable', it assumes the
# values are missing; if set to a filepath that points to an external catalog of RUWE parameters, it will use those
# values.
ruwe_source="gaiadr2" # "/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/data/source_catalogs/gaiadr2.csv_with_ticid.csv"
# Whether to plot model input figures for all SPOC TCEs found
plot_inputs_to_model="false"
# choose classification task between "phot-vetting" (PC vs AFP vs NTP) and "planet-validation" (planet vs not-planet).
task="planet-validation"
# choose type of ExoMiner model among: single, cv_ensemble (avg 10 models), or full_cv_ensemble (avg 10 ensemble CV models)
exominer_model="cv_ensemble"
# max number of workers for inference parallelization
max_model_workers=4

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --runner RUNNER                      Runner to use: 'local' (default) or 'podman'"
    echo "  --pipeline_python_script_fp FILE     Pipeline Python main script filepath (used for 'local' runner)"
    echo "  --tics_tbl_fp FILE                   TIC IDs table filepath"
    echo "  --exominer_pipeline_run_dir DIR      Directory to store pipeline run output"
    echo "  --data_collection_mode MODE          Data collection mode (2min or ffi)"
    echo "  --num_processes N                    Number of processes"
    echo "  --num_jobs N                         Number of jobs"
    echo "  --get_mast_urls_dv_reports BOOL      Whether to create CSV file with MAST URLs for the TESS SPOC DV reports for the detected TCEs: true or false"
    echo "  --dv_xml_data_repository DIR         Path to data repository containing TESS SPOC DV XML files for the TIC IDs/sector runs in the TICs table"
    echo "  --lc_data_repository DIR             Path to data repository containing light curve FITS files for the TIC IDs/sector runsin the TICs table"
    echo "  --stellar_parameters_source SOURCE   Source for TICs stellar parameters (ticv8, tess-spoc, or path to local file)"
    echo "  --ruwe_source SOURCE                 Source for TICs Gaia RUWE parameters (gaiadr2, gaiadr3, gaiaedr3, or path to local file)"
    echo "  --plot_inputs_to_model BOOL          Whether to plot model input figures for all SPOC TCEs found for the TIC IDs requested in the run"
    echo "  --task TASK                          Classification task to perform between photometric vetting (PC vs AFP vs NTP) and planet vs not-planet"
    echo "  --exominer_model MODEL               ExoMiner model to use for inference (exominer_phot-vetting or exominer_planet-validation or path to custom model)"
    echo "  --max_model_workers N                Maximum number of workers created to run inference in parallel"
    echo "  --help                               Show ExoMiner Pipeline help"
    echo ""
    exit
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner) runner="$2"; shift 2 ;;
        --pipeline_python_script_fp) pipeline_python_script="$2"; shift 2 ;;
        --tics_tbl_fp) tics_tbl_fp="$2"; shift 2 ;;
        --exominer_pipeline_run_dir) exominer_pipeline_run_dir="$2"; shift 2 ;;
        --data_collection_mode) data_collection_mode="$2"; shift 2 ;;
        --num_processes) num_processes="$2"; shift 2 ;;
        --num_jobs) num_jobs="$2"; shift 2 ;;
        --get_mast_urls_dv_reports) get_mast_urls_dv_reports="$2"; shift 2 ;;
        --dv_xml_data_repository) dv_xml_data_repository="$2"; shift 2 ;;
        --lc_data_repository) lc_data_repository="$2"; shift 2 ;;
        --stellar_parameters_source) stellar_parameters_source="$2"; shift 2 ;;
        --ruwe_source) ruwe_source="$2"; shift 2 ;;
        --plot_inputs_to_model) plot_inputs_to_model="$2"; shift 2 ;;
        --task) task="$2"; shift 2 ;;
        --exominer_model) exominer_model="$2"; shift 2 ;;
        --max_model_workers) exominer_model="$2"; shift 2 ;;
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

# Determine the runner display string based on the runner type
if [[ "$runner" == "podman" ]]; then
    runner_display="Runner: $runner (Image: $podman_img)"
else
    runner_display="Runner: $runner"
fi

echo "Running ExoMiner Pipeline with the following parameters:"
echo "$runner_display"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

# Save parameters to a file inside the run directory
echo "Saving run parameters to $exominer_pipeline_run_dir/run_parameters.txt"
params_file="$exominer_pipeline_run_dir/run_parameters.txt"

cat <<EOF > "$params_file"
runner_display
TICs table file: $tics_tbl_fp
ExoMiner Pipeline run directory: $exominer_pipeline_run_dir
Data collection mode: $data_collection_mode
Number of processes: $num_processes
Number of jobs: $num_jobs
Get MAST URLS DV reports: $get_mast_urls_dv_reports
DV XML data repository: $dv_xml_data_repository
Light curve data repository: $lc_data_repository
Stellar parameters source: $stellar_parameters_source
RUWE source: $ruwe_source
Plot inputs to model: $plot_inputs_to_model
Task: $task
ExoMiner model: $exominer_model
Max number of processes for inference: $max_model_workers
EOF

if [ "$runner" = "podman" ]; then

    # set up volume mounts
    volume_mounts="-v $tics_tbl_fp:/tics_tbl.csv:Z -v $exominer_pipeline_run_dir:/outputs:Z"

    # conditionally add external_data_repository mount
    if [ "$dv_xml_data_repository" != "null" ]; then
      volume_mounts="$volume_mounts -v $dv_xml_data_repository:/dv_xml_data_repository:Z"
      dv_xml_data_repository_arg="--dv_xml_data_repository=/dv_xml_data_repository"
    else
      dv_xml_data_repository_arg=""
    fi

        # conditionally add external_data_repository mount
    if [ "$lc_data_repository" != "null" ]; then
      volume_mounts="$volume_mounts -v $lc_data_repository:/lc_data_repository:Z"
      lc_data_repository_arg="--lc_data_repository=/lc_data_repository"
    else
      lc_data_repository_arg=""
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
    if [[ "$exominer_model" != "single" && "$exominer_model" != "cv_ensemble" && "$exominer_model" != "full_cv_ensemble" ]]; then
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

    # conditionally add plot figures
    if [ "$plot_inputs_to_model" = "true" ]; then
      plot_inputs_to_model_arg="--plot_inputs_to_model"
    else
      plot_inputs_to_model_arg=""
    fi

    podman run \
      --pids-limit=-1 \
      --shm-size=16g \
      -e OPENBLAS_NUM_THREADS=1 \
      -e OMP_NUM_THREADS=1 \
      -e MKL_NUM_THREADS=1 \
      -e VECLIB_MAXIMUM_THREADS=1 \
      -e NUMEXPR_NUM_THREADS=1 \
      -e TF_NUM_INTRAOP_THREADS=1 \
      -e TF_NUM_INTEROP_THREADS=1 \
      ${volume_mounts} \
      $podman_img \
      --tic_ids_fp=/tics_tbl.csv \
      --output_dir=/outputs \
      --data_collection_mode=$data_collection_mode \
      --num_processes=$num_processes \
      --num_jobs=$num_jobs \
      --get_mast_urls_dv_reports=$get_mast_urls_dv_reports \
      --stellar_parameters_source=$stellar_parameters_source_arg \
      --ruwe_source=$ruwe_source_arg \
      --task="$task" \
      --exominer_model=$exominer_model_arg \
      --max_model_workers=$max_model_workers \
      $dv_xml_data_repository_arg \
      $lc_data_repository_arg \
      $plot_inputs_to_model_arg > "$exominer_pipeline_run_dir/podman_output.log" 2>&1

elif [ "$runner" = "python" ]; then
    echo "Python script: $pipeline_python_script" >> "$params_file"

    # conditionally add data repositories
    if [ "$dv_xml_data_repository" != "null" ]; then
      dv_xml_data_repository_arg="--dv_xml_data_repository=$dv_xml_data_repository"
    else
      dv_xml_data_repository_arg=""
    fi

    if [ "$lc_data_repository" != "null" ]; then
      lc_data_repository_arg="--lc_data_repository=$lc_data_repository"
    else
      lc_data_repository_arg=""
    fi

    # conditionally add plot figures
    if [ "$plot_inputs_to_model" = "true" ]; then
      plot_inputs_to_model_arg="--plot_inputs_to_model"
    else
      plot_inputs_to_model_arg=""
    fi

    python "$pipeline_python_script" \
      --tic_ids_fp="$tics_tbl_fp" \
      --output_dir="$exominer_pipeline_run_dir" \
      --data_collection_mode="$data_collection_mode" \
      --num_processes="$num_processes" \
      --num_jobs="$num_jobs" \
      --get_mast_urls_dv_reports="$get_mast_urls_dv_reports" \
      --stellar_parameters_source="$stellar_parameters_source" \
      --ruwe_source="$ruwe_source" \
      --task="$task" \
      --exominer_model="$exominer_model" \
      --max_model_workers=$max_model_workers \
      $dv_xml_data_repository_arg \
      $lc_data_repository_arg \
      $plot_inputs_to_model_arg > "$exominer_pipeline_run_dir/python_output.log" 2>&1
else
    echo "Error: Unknown runner '$runner'. Please choose 'python' or 'podman'."
    exit 1
fi

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run_dir."

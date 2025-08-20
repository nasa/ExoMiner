#!/bin/bash

# Setup
runs_root_dir="/data3/exoplnt_dl/experiments/exominer_pipeline/runs"
inputs_dir="/data3/exoplnt_dl/experiments/exominer_pipeline/inputs"
run_podman_sh_script_fp="/data3/exoplnt_dl/codebase/exominer_pipeline/run_podman_application.sh"

data_collection_mode="2min"
num_processes=1
num_jobs=1
download_spoc_data_products="true"
external_data_repository="null"
stellar_parameters_source="ticv8"
ruwe_source="gaiadr2"
delete_data_after_run=false

# Logging setup
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${runs_root_dir}/exominer_pipeline_runs_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Searching for TICs tables in $inputs_dir..."
tics_tbls_fps=($inputs_dir/test*.csv)
echo "Found ${#tics_tbls_fps[@]} TICs tables."

# Iterate over each TICs table
for fp in "${tics_tbls_fps[@]}"; do
    echo "Processing $(basename "$fp")..."

    timestamp=$(date +"%m-%d-%Y_%H%M%S")
    run_name="exominer_pipeline_run_$(basename "$fp" .csv)_${timestamp}"
    run_dir="${runs_root_dir}/${run_name}"

    bash "$run_podman_sh_script_fp" \
        --tics_tbl_fp "$fp" \
        --exominer_pipeline_run_dir "$run_dir" \
        --data_collection_mode "$data_collection_mode" \
        --num_processes "$num_processes" \
        --num_jobs "$num_jobs" \
        --download_spoc_data_products "$download_spoc_data_products" \
        --external_data_repository "$external_data_repository" \
        --stellar_parameters_source "$stellar_parameters_source" \
        --ruwe_source "$ruwe_source"

    echo "Run completed for $(basename "$fp"). Results saved to $run_dir."

    if [ "$delete_data_after_run" = true ]; then
        echo "Cleaning up job directories and data in $run_dir..."
        rm -f "$run_dir/$(basename "$fp")"
        rm -rf "$run_dir"/job_*
        echo "Cleanup complete for $run_dir."
    fi
done

# Running ExoMiner Pipeline

After pulling the Podman image, you are ready to run the pipeline by running the script 
[run_pipeline.sh](/exominer_pipeline/run_pipeline.sh) in your terminal! The pipeline is fully 
parallelizable and can make use of multi-CPU 
cores machines. The TIC IDs are split evenly across a set of jobs defined by the user, meaning that all TCEs for a given
TIC ID and sector run are processed in the same job. 

## Command-line arguments
Before you run the pipeline, you have to set the filepaths to the command-line arguments. Run 
```podman run ghcr.io/nasa/exominer --help``` for a detailed description on these inputs. 
You can modify the command ```podman run``` to suit your use case (e.g., give TIC IDs as a comma-separated list instead 
of a CSV file).

## Running the Podman container application

```bash
🖥️ [ Main Process ]
                                           |
                              🔀 (Multiprocessing: Splits into N Jobs)
                                           |
   +------------------------------------------------------------------------------------------+
   |                                                                                          |
   |    🤖 [ Worker 1 ]                🤖 [ Worker 2 ]                🤖 [ Worker N ]         |
   |  📡 Query MAST/S3 OR            📡 Query MAST/S3 OR            📡 Query MAST/S3 OR       |
   |     local DV XML/LC FITS           local DV XML/LC FITS           local DV XML/LC FITS   |
   |  📋 Create DV TCE table         📋 Create DV TCE table         📋 Create DV TCE table    |
   |  ⭐ Query/read stellar &        ⭐ Query/read stellar &        ⭐ Query/read stellar     |
   |     RUWE sources                   RUWE sources                   & RUWE sources         |
   |  📉 Preprocess light curves     📉 Preprocess light curves     📉 Preprocess lightcurves |
   |  📸 Extract/preprocess          📸 Extract/preprocess          📸 Extract/preprocess     |
   |     DV diff img                    DV diff img                    DV diff img            |
   |  ⚖️ Normalize features          ⚖️ Normalize features          ⚖️ Normalize features     |
   |  📄 Create TCE plot             📄 Create TCE plot             📄 Create TCE plot        |
   |     summaries PDFs [opt]           summaries PDFs [opt]           summaries [opt]        |
   |                                                                                          |
   +------------------------------------------------------------------------------------------+
                                           |
                                 🤝 (Process Pool Joins)
                                           |
                                  🖥️ [ Main Process ]
                                           |
                              🔀 (Multiprocessing: Splits into M Processes)
                                           |
   +------------------------------------------------------------------------------------------+
   |                                                                                          |
   |    🧵 [ Worker 1 ]                🧵 [ Worker 2 ]                🧵 [ Worker M ]         |
   |  🧠 Load ExoMiner Model 1       🧠 Load ExoMiner Model 2       🧠 Load ExoMiner Model M  |
   |  📊 Predict on dataset          📊 Predict on dataset          📊 Predict on dataset     |
   |                                                                                          |
   +------------------------------------------------------------------------------------------+
                                           |
                                 🤝 (Process Pool Joins)
                                           |
                                  🖥️ [ Main Process ]
                                📉 - Average ensemble scores
                                💾 - Save prediction CSVs
                                📑 - Query MAST for DV reports [opt]
```
The pipeline is structured in two separate stages: 
- **Stage 1 [DATA PREPROCESSING]**: the TIC IDs defined in the input table `tics_tbl_fp` are split into `num_jobs` jobs, each one responsible for downloading/reading the light curves and SPOC DV XML files for those targets, and prepocessing the inputs for the SPOC TCEs to be fed to the ExoMiner model. The number of processes `num_processes` determines how many workers to spawn in parallel to run those jobs. Set `num_processes` according to your system resources. For example, if running on an 8-core CPU machine, you can have up to 8 parallel workers processing up to 8 jobs at any given time. Keep in mind that each process takes up its own memory and you might need some resources for your other tasks. **If you are running into memory issues, decrease the number of processes used to parallelize data preprocessing.** It is recommendeded to set the number of jobs to be larger than the number of processes being used.
- **Stage 2 [MODEL INFERENCE]**: after creating the dataset in Stage 1, the model inference is performed. This stage involves loading the model and dataset into memory, and running inference to produce the final set of predictions for the SPOC TCEs found in those targets and sector runs whose data were successfully preprocessed. The `max_model_workers` defines the maximum number of processes running in parallel for the model inference in the case of choosing ensemble models. Again, the number of processes should be adjusted based on your system resources (start small and increase if you think you have enough wiggle room to increase parallelization). **More workers means faster inference when using ensemble models since more models are running inference in parallel. However, more workers also means higher memory usage (more processes and threads running for the data ingestion pipeline, loading weights, activations, and library binaries). Start with a small number of workers. If you are running into memory issues, lower the number of workers**.

A **detailed explanation of the ExoMiner Pipeline outputs** can be found in [here](../docs/pipeline_outputs/exominer-pipeline-outputs.pdf).

### Template Shell Script for Running ExoMiner Pipeline

```bash
#!/bin/bash

### Run ExoMiner Pipeline either via the Python application or Podman

### default values ##

# execution runner ('python' for Python or 'podman' for Podman application)
runner="podman"
podman_img="ghcr.io/nasa/exominer" 

# python pipeline python script (used if runner=python)
pipeline_python_script="./run_pipeline.py"

# file path to the TICs table
tics_tbl_fp="./tics_table.csv"

# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir="./exominer_pipeline_run_outputs"

# data collection mode: either 2min or ffi
data_collection_mode="2min"

# number of processes
num_processes=8

# number of jobs to split the TIC IDs
num_jobs=9

# set to "true" or "false". If "true", it will create a CSV file with URLs to the SPOC DV reports for each TCE in the
# queried TICs
get_mast_urls_dv_reports="true"

# path to directories containing the light curve FITS files and DV XML files for the TIC IDs and sector runs that you want to query; set to "null" otherwise
dv_xml_data_repository="null"
lc_data_repository="null"

# define source of stellar parameters for TICs. If set to 'ticv8', TIC-8 is queried; if set to 'tess-spoc', it uses the
# parameters stored in the TICs DV XML files; if set to a filepath that points to an external catalog of stellar
# parameters, it will use those values.
stellar_parameters_source="tess-spoc"

# define source of Gaia RUWE for TICs. If set to 'gaiadr2', 'gaiadr3', or 'gaiaedr3', Gaia DR2, DR3, or EDR3, respectively is queried; if set to 'unavailable', it assumes the
# values are missing; if set to a filepath that points to an external catalog of RUWE parameters, it will use those
# values.
ruwe_source="gaiadr2"

# whether to plot model input figures for all SPOC TCEs found
plot_inputs_to_model="true"

# choose classification task between "phot-vetting" (PC vs AFP vs NTP) and "planet-validation" (planet vs not-planet).
task="planet-validation"

# choose type of ExoMiner model among: single, cv_ensemble (avg 10 models), or full_cv_ensemble (avg 10 ensemble CV models)
exominer_model="single"

# max number of workers for inference parallelization
max_model_workers=8

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

```

### Shell Script Arguments

Information on arguments used in the shell script are provided below:

- `runner`: Defines the execution environment. Set to `"podman"` to run the containerized application (using the image defined in `podman_img`) or `"python"` to run the local Python application. **Using `"python"` requires creating a Python environment in your machine (e.g., using Conda) and downloading the ExoMiner repository from GitHub.**
- `pipeline_python_script_fp`: The filepath pointing to the main pipeline Python script (e.g., `run_pipeline.py`). This argument is only used if the `runner` is set to `"python"`.
- `tics_tbl_fp`: Filepath to the input CSV table containing the TIC IDs and sector runs you want to process. (For information on the structure of the input CSV file, see section [TIC IDs input](#tic-ids-input)).
- `exominer_pipeline_run_dir`: The directory filepath where all pipeline outputs, logs, and results will be saved. The script will automatically create this directory if it doesn't exist.
- `data_collection_mode`: Specifies the TESS data type to use. Options are `"2min"` (2-minute cadence) or `"ffi"` (Full Frame Images).
- `num_processes`: The number of processes used for parallelizing the preprocessing steps (set to `1` for no parallelization).
- `num_jobs`: The number of chunks/jobs to split the queried TIC IDs into for preprocessing. **For example, if you want to process 100 TIC IDs using 4 processes and 10 jobs, then each job will handle 10 TIC IDs, and at any time there will be 4 jobs running in parallel (each processing 10 TIC IDs, so a total of 40 TIC IDs simultaneously).**
- `get_mast_urls_dv_reports`: Set to `"true"` or `"false"`. If `"true"`, the pipeline creates a CSV file containing MAST URLs for the TESS SPOC DV reports corresponding to the queried TCEs. **This requires Internet access and that the MAST/S3 servers are accessible and running.**
- `dv_xml_data_repository`: Filepath to a local directory containing TESS SPOC DV XML files. If set to `"null"`, the pipeline will automatically download these files from MAST. **This requires Internet access and that the MAST/S3 servers are accessible and running.**
- `lc_data_repository`: Filepath to a local directory containing light curve TESS FITS files. If set to `"null"`, the pipeline will automatically download them from MAST. **This requires Internet access and that the MAST/S3 servers are accessible and running.**
- `stellar_parameters_source`: Defines where to get stellar parameters for the TICs. Set to `"ticv8"` to query TIC-8, `"tess-spoc"` to use parameters stored in the DV XML files, or provide a filepath to a local CSV catalog to use your own values. **By setting the source as TIC-8, it requires Internet access and that the MAST servers are accessible and running.** 
- `ruwe_source`: Defines the source for the Gaia RUWE values. Options include `"gaiadr2"`, `"gaiadr3"`, `"gaiaedr3"` to query the respective Gaia catalogs; `"unavailable"` to assume values are missing; or a filepath to a local CSV catalog containing the intended parameters. **By specifying a Gaia source for RUWE, it requires Internet access and that the Gaia servers are accessible and running.**
- `plot_inputs_to_model`: Set to `"true"` or `"false"`. If `"true"`, the pipeline will generate summary PDF files plotting the input features to the model for each SPOC TCE found. (See [ExoMiner Pipeline Plots](#exominer-pipeline-plots) for examples).
- `task`: The classification task to perform. Choose `"phot-vetting"` to classify TCEs into planet candidates (PCs), astrophysical false positives (AFPs), or non-transiting phenomena (NTPs); or choose `"planet-validation"` to classify them strictly as planets vs. not-planets.
- `exominer_model`: The specific ExoMiner model architecture to use for inference. Built-in options include `"single"`, `"cv_ensemble"` (average of 10 models), or `"full_cv_ensemble"` (average of 10 ensemble CV models). You can also provide a filepath to a custom `.keras` model file. (See [ExoMiner Models](#exominer-models)).
- `max_model_workers`: The maximum number of workers spun up to run model inference in parallel. This only matters when choosing ensemble models like `"cv_ensemble"` and `"full_cv_ensemble"`. For example, if `"max_model_workers"` is set to 2, then at any time two model inference processes will be running simultaneously, each one running the full dataset through a model. That means that the inference time will be cut down by approximately half. **Start small with one or two workers. Even though your machine might have a large number of scores much greater than this, each process takes a decent chunk of memory, especially when running inference with large datasets (i.e., at least hundreds of examples). If you find that the run failed to generate predictions, adjust this number to 1 and run again.**

## TIC IDs input

You can provide a set of TIC IDs by creating a CSV file with the columns "tic_id" and "sector_run" and set the variable 
[tics_tbl_fp](#running-the-podman-container-application) to its path.
The following example showcases a CSV file that can be used with the pipeline to generate results for the TCEs of TIC 167526485 in single-sector run S6 and multi-sector run S1-39.

Example: CSV file

```csv
tic_id, sector_run
167526485, 6-6
167526485, 1-39
```

## Outputs

![Pipeline Demo - Show results of run](/others/media/3_check_exominer_pipeline_run_results_edited.gif)

The following diagram represents the hierarchy of the data output for a run of the ExoMiner Pipeline. In this example, 
the pipeline was run using one single job and the TESS SPOC TCEs were queried for the 2-min data. The structure was 
ordered from most recent file/folder to the oldest.

```code
exominer_pipeline_run_name
├── run_main.log
├── pipeline_run_config.yaml
├── master_tic_tracking_summary.csv
├── master_tce_tracking_summary.csv
├── podman_output.log (or python_output.log)
├── plot_inputs_to_model [optional]
├── dv_reports_all_jobs.csv [optional]
├── predictions_predictset.csv
├── pipeline_run_config.yaml
├── tics_tbl.csv
├── run_parameters.txt
├── config_files
└── job_0
    ├── run_0.log
    ├── mast_urls_tables [optional]
    ├── tfrecord_data_diffimg_normalized
    ├── tfrecord_data_diffimg
    ├── diff_img_preprocessed
    ├── diff_img_extracted
    ├── tfrecord_data
    ├── tce_table
    ├── manifest_requested_products_2min.csv [when downloading products from MAST]
    ├── requested_products_2min.csv [when downloading products from MAST]
    └── mastDownload [when downloading products from MAST]
```

**Content description**

- `run_main.log`: main log file for the run.
- `pipeline_run_config.yaml`: main configuration file for the pipeline.
- `master_tic_tracking_summary.csv`: tracks processed TICs over jobs.
- `master_tce_tracking_summary.csv`: tracks processed TCEs over jobs. 
- `podman_output.log (or python_output.log)`: stdout/stderr output of the pipeline.
- `plot_inputs_to_model`: this directory will contain summary PDF files for each TCE, showing the inputs provided to the model after preprocessing in the ExoMiner Pipeline. This summary includes plots such as odd/even comparison, weak secondary, periodogram, and difference images.
- `dv_reports_all_jobs.csv` (optional): if the flag `--download_spoc_data_products` is set to `"true"`, then a CSV file will be created that contains, for each TCE in all the queried TICs, the URLs for the TESS SPOC DV data reports found at the MAST.
- `predictions_predictset.csv`: if the run is completed, a CSV file is generated containing the predictions scores produced by the ExoMiner model for the set of TCEs associated with the TIC IDs and sector runs defined in `tics_tbl.csv`. If multiple jobs are completed, it aggregates the predictions generated across them.
- `pipeline_run_config.yaml`: YAML file that stores some of the run parameters internally used in the Podman container.
- `tics_tbl.csv`: CSV file containing the queried TIC IDs and sector runs.
- `run_parameters.txt`: text file containing information about the parameters used for the run, including the version of the pipeline.
- `config_files`: this directory contains copies of some of the configuration YAML files used in different parts of the pipeline.
- `job_{job_id}`: directory containing the results for the TIC IDs and sector runs assigned to the job.
    - `run_{job_id}`: log file for the job.
    - `tce_tracker_job_{job_id}.csv`: CSV file tracking processed TCEs in the job.
    - `mast_urls_tables` (optional): if the flag `--download_spoc_data_products` is set to `"true"`, then a CSV file `spoc-dv_mast-urls_job{job_id}` will be created under this directory containing, for each TCE in the queried TICs for this job, the URLs for the TESS SPOC DV data reports found at the MAST.
    <!-- - `predictions`: contains the CSV file, `predictions_predictset.csv`, with the predictions generated for the assigned TIC IDs and sector runs. -->
    - `tfrecord_data_diffimg_normalized`: TFRecord dataset with light curve and difference image data for the TCEs. Features have been normalized. It should include a TFRecord file name `shard-tess_diffimg_spoc_data_0` that contains the normalized data. It also includes a `normalization_run.yaml` with the settings used to normalize the difference image data.
    - `tfrecord_data_diffimg`: TFRecord dataset with light curve and difference image data for the TCEs. It should include a TFRecord file name `shard-tess_diffimg_spoc_data_0` that contains the preprocessed light curve data and the difference image data. It also includes a `config_add_diff_img_tfrecords.yaml` with the settings used to add the difference image data to the TFRecord dataset, as well as a `logs` directory and a `examples_failed.csv` in case any TCE had its difference image data failed to be added to the dataset.
    - `diff_img_preprocessed`: includes the preprocessed difference image data for the queried TCEs. It should include a file, `tess_diffimg_spoc_data/diffimg_preprocess.npy`, that contains the preprocessed difference image data for the TCEs.
    - `diff_img_extracted`: includes the difference image data for the TCEs extracted from the DV XML files for the assigned TIC IDs and sector runs. It should include a file named `tess_diffimg_spoc_data.npy` after the difference image has been extracted from the DV XML files.
    - `tfrecord_data`: TFRecord dataset with light curve data for the TCEs. The directory should contain one TFRecord file with the filename `shard-00000-of-00001-node-{node_id}`, and a `shards_tbl.csv` that provided information on the content of the file. The directory `exclusion_logs` will show one log file per TCE for those TCEs whose light curve data preprocessing found an error or warning. The directory `preprocessing_logs` contains a log file with information related to the preprocessing of the data.
    - `tce_table`: contains data used to create a table of TESS SPOC TCEs detected for the assigned TIC IDs and sector runs. The final preprocessed table is `tess-spoc-dv_tces_0_processed.csv`. Additional results include querying Gaia DR2 for RUWE values, and TICv8 for updated stellar parameters.
    - `manifest_requested_products_{data_collection_mode}.csv`: CSV file that includes information on the location of the downloaded files from the MAST and whether the download was successful.
    - `requested_products_{data_collection_mode}.csv`: CSV file that shows all data products that are requested for download (light curves FITS files and DV XML files) from the MAST.
    - `mastDownload`: includes the light curve FITS files and DV XML files downloaded from the MAST for the assigned TIC IDs and sector runs. If the download is successful, each target should have a directory with a DV XML file related to the corresponding sector run, and a set of one or more folders related to the sectors the target was observed, each one containing the corresponding light curve FITS file.

## Local source catalogs

Instead of querying external repositories such as TIC-8 and Gaia DR2 (or Gaia DR3 and EDR3) to get the set of stellar parameters and Gaia RUWE 
for each queried TIC, the user can also provide their own source catalogs as CSV files. These catalogs should have the 
following format:

- TICs stellar parameters catalog
```csv
target_id, tic_steff, tic_steff_err, tic_smass, tic_smass_err, tic_smet, tic_smet_err, tic_sradius, tic_sradius_err, tic_sdens, tic_sdens_err, tic_slogg, tic_slogg_err, tic_ra, tic_dec, kic_id, gaia_id, tic_tmag, tic_tmag_err

167526485, 5778, 0, 1, 0, 0, 0, 1, 1, ... 
```

with the following mapping: `tic_steff` stellar effective temperature (K), `tic_smass` stellar mass (Solar mass), 
`tic_smet` stellar metallicity (dex), `tic_sradius` stellar radius (Solar Radii), tic_sdens is stellar density (g/cm3), 
`tic_slogg` stellar gravity (log10(cm/s)), `tic_ra` right ascension (deg), `tic_dec` declination (deg),`kic_id` KIC ID, 
`gaia_id` Gaia ID, `tic_tmag` TESS magnitude. 

KIC ID, Gaia ID, and parameter uncertainties (i.e., "_err") are not used and can be missing while `tic_ra` and `tic_dec` should 
be available. All the remaining parameters can be missing, but it is encouraged that they are present - otherwise the 
pipeline will replace those values automatically.

- RUWE catalog
```csv
target_id, ruwe

167526485, 1 
```

## ExoMiner Pipeline Plots

In here, we showcase an example of what the ExoMiner Pipeline TCE Summary looks like. The TCE Summary is a collection of plots that show the preprocessed versions of the inputs provided to the ExoMiner models for each TCE. You can [click here](../others/images/tess-spoc-tce_tic66818296-1-S1-92_summary_exominer-pipeline.pdf) to view a summary example for TIC 66818296.1 in sector run S1-S92. For a more detailed explanation of these plots, we refer to the [ExoMiner Pipeline Outputs report](pipeline_outputs/exominer-pipeline-outputs.pdf). Below you can find a brief description of each plot:

- **Phase-folded and binned flux and centroid views** [Page 1]: this first page shows the phase-folded and binned views of full-orbit and transit-view of the flux and flux-weighted centroid motion, as well as other transit-views for the odd and even fluxes and weak secondary. The red dashed lines represent the +-1-sigma standard error of the mean (SEM) envelope which gives an idea of the uncertainty/noise in the data used to create these binned views. For some of the views, the number of phases used to create the binned views is shown. At the top, the TCE ID and ephemerides of the TCE are shown, followed by an assortment of TCE and TIC stellar parameters are shown.
- **Weak Secondary** [Page 2]: this page shows the views for the detected weak secondary. The top plot shows the full-orbit (i.e., one full period) weak secondary view centered on the detected secondary transit. The transit midpoint of the primary transit is indicated by a vertical red dashed line. The bottom plots show a transit-view of the weak secondary view (on the left, not normalized, on the right the normalized version using the absolute minimum that is provided as input to the model).
- **Odd/Even** [Page 3]: this page shows the separate views for the odd and even transit-view fluxes at the top. The bottom plots shows the two binned views overlapped so it is easier to compare them.
- **Periodogram** [Page 4]: this page shows the full flux time series at the top plot. The middle and bottom plots show the same periodogram but with no normalization (middle) and with normalization by the max amplitude (bottom). The transit pulse model (TPM) periodogram is obtained by computing the periodogram on a transit pulse model based on the detected ephemerides and transit depth, and it is used to give information to the model about the expected periodogram features for the transit signal when compared to the peridogram of the data. The red dashed vertical lines point to the harmonics.
- **Difference Images** [Page 5+]: Each sector with available difference images that were successfully sampled and preprocessed by the ExoMiner Pipeline gets a page in this summary report. The difference images are shown in a grid format with the preprocessed difference, out-of-transit, and SNR flux images at the top. At the bottom, a "valid pixels" image is shown that describes which pixels in the image are valid (e.g., not missing; negative out-of-flux values). The header shows the quality metric of the difference image along with the target star's magnitude. The position of the target in pixel coordinates is shown as a red cross.

<!-- <iframe src="../others/images/tess-spoc-tce_tic66818296-1-S1-92_summary_exominer-pipeline.pdf" width="100%" height="600px">
    This browser does not support PDFs. Please download the PDF to view it: <a href="../others/images/tess-spoc-tce_tic66818296-1-S1-92_summary_exominer-pipeline.pdf">Download PDF</a>.
</iframe> -->


## ExoMiner Models

For a detailed description of the models shipped with the ExoMiner Pipeline we refer to [models_specs.md](models_specs.md). This model card describes the training setup for both classification tasks, input features (name, data type, dimensionality), and neural network topology. Currently, the ExoMiner Pipeline supports the use of these three models:
- `single` (~1.5M parameters): single model from a single cross-validation (CV) iteration. This is the most lightweight option, it will load in most consummer-grade computers memory and produce inference results in a reasonable amount of time for most systems.
- `cv_ensemble` (~15M parameters): ensemble average of 10 models from a single CV iteration. Requires more memory and computational power than the `single` model. These models were all trained using the same dataset split, but with different initializations.
- `full_cv_ensemble` (~150M parameters): ensemble average of 100 model from the full 10-fold CV run. Requires significantly more memory and computational power than the other two models. These models were trained using different dataset splits, as well as with different initializations.

<!-- - `exominer_phot-vetting` (~1.5M parameters): this model was trained for the multiclass classification task of classifying TCEs as planet candidates (PCs), astrophysical false positives (AFPs), or non-transiting phenomena (NTP). This model is more suitable for vetting efforts.
- `exominer_planet-validation` (~1.5M parameters): this model was trained for the binary classification task of classifying TCEs as planets vs not-planets. This model was developed for validation efforts. -->

These models are based in the ExoMiner++ architecture described in the [ExoMiner on TESS 2-min paper](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4).

These models reflect a tradeoff between generalization/robustness and computational cost. A single model from one cross-validation iteration (`single`) produces the fastest inference. However, we recommend the `full_cv_ensemble` (100 models) for users who want robust predictions that smooth out biases from individual CV iterations and training initializations. The `cv_ensemble` provides an intermediate option.

For the ensembles, the models are loaded separately and their scores are averaged. A standard deviation score is also computed, which provides valuable information about ExoMiner's epistemic uncertainty. 

For standard vetting purposes, the **mean score** serves as your primary classification metric, while the **uncertainty score (standard deviation)** serves as a triage tool to help SMEs identify which candidates require manual review. 

### Example scenarios regarding score uncertainty with model ensembles:
*(Note: A similar rationale applies to the 3-class photometric vetting task, but the baseline "ambiguous" mean score is 0.33 instead of 0.50).*

* **High Mean (0.99), Low Std (0.01):** The ensemble is highly confident. The data is clear, and models agree (most likely a planet). *The same logic applies to a near-zero mean for a clear false positive.*
* **Mid Mean (0.50), Low Std (0.01):** The ensemble unanimously agrees that the data is inherently ambiguous or noisy.
* **Mid Mean (0.50), High Std (0.40):** The models disagree significantly. This often indicates a complex edge case or out-of-distribution event that the model architecture struggles with. **These are prime candidates for manual SME review** and future model improvements.
* **High Mean (0.80), High Std (0.25):** The ensemble leans toward planet classification, but a minority of models strongly disagree. This potentially indicates a tricky false positive that warrants SME attention.

Given how fast inference is and the low memory footprint of an individual model, **we highly suggest users run the pipeline with the full CV ensemble of 100 models** to take advantage of these uncertainty insights.

### Using Custom Models
Alternatively, you can provide the filepath to a custom TensorFlow Keras model. To ensure compatibility with the Podman application, your custom model must:

1. Be compatible with **TensorFlow 2.13**.
2. Expect the same input features (or a subset of them) as the pre-packaged models. Input features must match the exact names, dimensions, and data types used by the default models. 

For a complete description of the expected model input features, please refer to [models_specs.md](models_specs.md).

## Running the pipeline without Podman

If you do not want to use the Podman image, you can also run the ExoMiner Pipeline as a Python application. This method 
requires an initial setup step:
1. Clone the GitHub repository. 
2. Install the required package dependencies. 

You can get the required package dependencies by installing a package manager and environment management system such as Mamba/micromamba, and then use it to build an environment with all the packages and Python modules required to run the pipeline. Depending on the architecture of your system, to replicate the environment used to run the ExoMiner Pipeline, you can use [environment_amd64.yml](../exominer_pipeline/environment_amd64.yml) along with [requirements_amd64.txt](../exominer_pipeline/requirements_amd64.txt) for x86_64 architectures or, similarly, [environment_amd64.yml](../exominer_pipeline/environment_amd64.yml) along with [requirements_amd64.txt](../exominer_pipeline/requirements_amd64.txt) for ARM64 architectures. Run command `micromamba create -n exoplnt_dl -f /path/to/micromamba_env.yml` to create the environment, and then `micromamba activate exoplnt_dl` to activate it. After going through the setup, you can run the pipeline using the shell script [run_pipeline.sh](../exominer_pipeline/run_pipeline.sh) with variable `runner='local'`, or simply run the [run_pipeline.py](../exominer_pipeline/run_pipeline.py) Python script with the appropriate arguments.

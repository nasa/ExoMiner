# Running ExoMiner Pipeline

After pulling the Podman image, you are ready to run the pipeline by running the script 
[run_podman_application.sh](/exominer_pipeline/run_podman_application.sh) in your terminal! The pipeline is fully 
parallelizable and can make use of multi-CPU 
cores machines. The TIC IDs are split evenly across a set of jobs defined by the user, meaning that all TCEs for a given
TIC ID and sector run are processed in the same job. 

## Command-line arguments
Before you run the pipeline, you have to set the filepaths to the command-line arguments. Run 
```podman run ghcr.io/nasa/exominer --help``` for a detailed description on these inputs. 
You can modify the command ```podman run``` to suit your use case (e.g., give TIC IDs as a comma-separated list instead 
of a CSV file).

## Running the Podman container application

The contents of [run_podman_application.sh](/exominer_pipeline/run_podman_application.sh) are displayed below. To run 
the podman image for the ExoMiner Pipeline, simply set the arguments for your use case in the shell script file and run 
in your terminal `/path/to/run_podman_application.sh`. In this example, the ExoMiner Pipeline will be run for the TIC 
IDs and sector runs found in the CSV file that the variable `tics_tbl_fp` points to in your system. The pipeline will 
use TESS SPOC `2-min` data (see `data_collection_mode` variable) and `1` process will be used (no parallelization). The 
TIC IDs are split across `2` jobs. The results will be saved into the filepath that `exominer_pipeline_run_dir` points 
to. Furthermore, because `download_spoc_data_products` was set to `true`, a CSV file with the MAST URLs for the TESS 
SPOC DV reports generated for the TCEs of the queried TIC IDs will be generated. Since `external_data_repository` was 
set to `null`, the pipeline will download the light curve FITS and DV XML files for the queried TICs from the MAST. 
Similarly, the source for the stellar parameters and Gaia RUWE value - `stellar_parameters_source` and `ruwe_source`-  
were set to `ticv8` and `gaiadr2`, respectively. This means that the pipeline will query TIC-8 and Gaia DR2 to extract 
these parameters for the queried, but the user can also provide their own source catalogs by setting these variables to 
the path of the intended CSV tables. More information on using local source catalogs in [here](#local-source-catalogs). 
For information on the structure of the input CSV file, see section [TIC IDs input](#tic-ids-input).

```bash
# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs"
# file path to the TICs table
tics_tbl_fn=tics_tbl_filename.csv
tics_tbl_fp=$inputs_dir/$tics_tbl_fn
# name of the run
exominer_pipeline_run=exominer_pipeline_run_7-21-2025_1432
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir="/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/$exominer_pipeline_run"
# data collection mode: either 2min or ffi
data_collection_mode=ffi
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
```

## TIC IDs input

You can provide a set of TIC IDs using two methods: 
- create a CSV file with the columns "tic_id" and "sector_run" and set the variable 
[tics_tbl_fp](#running-the-podman-container-application) to its path.
-  set the variable [tic_ids](#running-the-podman-container-application) to a string in which the TIC IDs are 
separated by a comma. The following examples showcase the use of these two methods to generate results for the TCEs of 
TIC 167526485 in single-sector run S6 and multi-sector run S1-39.

Example: CSV file

```csv
tic_id, sector_run
167526485, 6-6
167526485, 1-39
```

Example: comma-separated list
```bash
167526485_6-6,167526485_1-39
```

## Outputs

The following diagram represents the hierarchy of the data output for a run of the ExoMiner Pipeline. In this example, 
the pipeline was run using one single job and the TESS SPOC TCEs were queried for the 2-min data. The structure was 
ordered from most recent file/folder to the oldest.

```code
exominer_pipeline_run_name
├── run_main.log
├── dv_reports_all_jobs.csv [optional]
├── predictions_{exominer_pipeline_run_name}.csv
├── pipeline_run_config.yaml
├── tics_tbl.csv
└── job_0
    ├── run_0.log
    ├── dv_reports.csv [optional]
    ├── predictions
    ├── tfrecord_data_diffimg_normalized
    ├── tfrecord_data_diffimg
    ├── diff_img_preprocessed
    ├── diff_img_extracted
    ├── tfrecord_data
    ├── tce_table
    ├── manifest_requested_products_2min.csv
    ├── requested_products_2min.csv
    └── mastDownload
```

**Content description**

- `run_main.log`: main log file for the run.
- `dv_reports_all_jobs.csv` (optional): if the flag `--download_spoc_data_products` is set to `"true"`, then a CSV file will be created that contained, for each 
       TCE in all the queried TICs, the URLs for the TESS SPOC DV data reports found at the MAST.
- `predictions_output.csv`: if the run is completed, a CSV file is generated containing the predictions scores produced by the ExoMiner model for the set of TCEs associated with the TIC IDs and sector runs defined in `tics_tbl.csv`. If multiple jobs are completed, it aggregates the predictions generated across them.
- `pipeline_run_config.yaml`: YAML file that stores the run parameters.
- `tics_tbl.csv`: CSV file containing the queried TIC IDs and sector runs.
- `job_{job_id}`: directory containing the results for the TIC IDs and sector runs assigned to the job.
    - `run_{job_id}`: log file for the job.
    - `dv_reports.csv` (optional): if the flag `--download_spoc_data_products` is set to `"true"`, then a CSV file will be created that contained, for each 
       TCE in the queried TICs for this job, the URLs for the TESS SPOC DV data reports found at the MAST.
    - `predictions`: contains the CSV file, `ranked_predictions_predictset.csv`, with the predictions generated for the assigned TIC IDs and sector runs.
    - `tfrecord_data_diffimg_normalized`: TFRecord dataset with light curve and difference image data for the TCEs. Features have been normalized. It should include a TFRecord file name `shard-tess_diffimg_TESS_0` that contains the normalized data.
    - `tfrecord_data_diffimg`: TFRecord dataset with light curve and difference image data for the TCEs. It should include a TFRecord file name `shard-tess_diffimg_TESS_0` that contains the preprocessed light curve data and the difference image data.
    - `diff_img_preprocessed`: includes the preprocessed difference image data for the queried TCEs. It should include a file, `tess_diffimg_TESS/diffimg_preprocess.npy`, that contains the preprocessed difference image data for the TCEs.
    - `diff_img_extracted`: includes the difference image data for the TCEs extracted from the DV XML files for the assigned TIC IDs and sector runs. It should include a file named `tess_diffimg_TESS.npy` after the difference image has been extracted from the DV XML files.
    - `tfrecord_data`: TFRecord dataset with light curve data for the TCEs. The directory should contain one TFRecord file with the filename `shard-00000-of-00001-node-{node_id}`, and a `shards_tbl.csv` that provided information on the content of the file. The directory `exclusion_logs` will show one log file per TCE for those TCEs whose light curve data preprocessing found an error or warning. The directory `preprocessing_logs` contains a log file with information related to the preprocessing of the data.
    - `tce_table`: contains data used to create a table of TESS SPOC TCEs detected for the assigned TIC IDs and sector runs. The final preprocessed table is `tess-spoc-dv_tces_0_processed.csv`. Additional results include querying Gaia DR2 for RUWE values, and TICv8 for updated stellar parameters.
    - `manifest_requested_products_2min.csv`: CSV file that includes information on the location of the downloaded files from the MAST and whether the download was successful. The "2min" suffix means that 2-min data was downloaded.
    - `requested_products_2min.csv`: CSV file that shows all data products that are requested for download (light curves FITS files and DV XML files) from the MAST.
    - `mastDownload`: includes the light curve FITS files and DV XML files downloaded from the MAST for the assigned TIC IDs and sector runs. If the download is successful, each target should have a directory with a DV XML file related to the corresponding sector run, and a set of one or more folders related to the sectors the target was observed, each one containing the corresponding light curve FITS file.

## Local source catalogs

Instead of querying external repositories such as TIC-8 and Gaia DR2 to get the set of stellar parameters and Gaia RUWE 
for each queried TIC, the user can also provide their own source catalogs as CSV files. These catalogs should have the 
following format:

- TICs stellar parameters catalog
```csv
target_id, tic_steff, tic_steff_err, tic_smass, tic_smass_err, tic_smet, tic_smet_err, tic_sradius, tic_sradius_err, tic_sdens, tic_sdens_err, tic_slogg, tic_slogg_err, tic_ra, tic_dec, kic_id, gaia_id, tic_tmag, tic_tmag_err

167526485, 5778, 0, 1, 0, 0, 0, 1, 1, ... 
```

with the following mapping: `tic_steff` stellar effective temperature (K), `tic_smass` stellar mass (Solar mass), 
`tic_smet` stellar metallicity (dex), `tic_sradius` stellar radius (Solar Radii), tic_sdens is stellar density (g/cm3), 
`tic_slogg` stellar gravity (log10(cm/s)), `ra` right ascension (deg), `deg` declination (deg),`kic_id` KIC ID, 
`gaia_id` Gaia ID, `tic_tmag` TESS magnitude. 

KIC ID, Gaia ID, and parameter uncertainties (i.e., "_err") are not used and can be missing while `ra` and `dec` should 
be available. All the remaining parameters can be missing, but it is encouraged that they are present - otherwise the 
pipeline will replace those values automatically.

- RUWE catalog
```csv
target_id, ruwe

167526485, 1 
```

## Running the pipeline without Podman

If you do not want to use the Podman image, you can also run the ExoMiner Pipeline as a Python application. This method 
requires an initial setup step:
1. Clone the GitHub repository. 
2. Install the required package dependencies. 

You can get the required package dependencies by installing a package manager and environment 
management system such as Conda, and then use it to build an environment with all the packages and Python modules 
required to run the pipeline. Depending on the architecture of your system, you can use 
[conda_env_exoplnt_dl_amd64.yml](../exominer_pipeline/conda_env_exoplnt_dl_amd64.yml) or 
[conda_env_exoplnt_dl_arm64.yml](../exominer_pipeline/conda_env_exoplnt_dl_arm64.yml) to replicate a Conda environment 
that can be used to run this pipeline. Run command `conda env create -f /path/to/conda_env.yml` to create the 
environment, and then `conda activate exoplnt_dl` to activate it.

After going through the setup, you can run the pipeline using the shell script 
[run_pipeline.sh](../exominer_pipeline/run_pipeline.sh) that wraps around the Python script [run_pipeline.py](../exominer_pipeline/run_pipeline.py), or 
simply run this Python script.
- Running shell script (set variables accordingly):
```bash
/path/to/run_pipeline.sh -i $INPUTS_DIR -t $TICS_TBL_FN -r $RUN_DIR -m $DATA_COLLECTION_MODE -p $NUM_PROCESSES 
-j $NUM_JOBS -d $DOWNLOAD_SPOC_DATA_PRODUCTS -e $EXTERNAL_DATA_REPOSITORY -s $PIPELINE_PYTHON_FP
```

- Running Python script (set variables accordingly):
```bash
python /path/to/run_pipeline.py --output_dir $INPUTS_DIR --tic_ids_fp $TICS_TBL_FP --data_collection_mode $DATA_COLLECTION_MODE --num_processes $NUM_PROCESSES 
--num_jobs $NUM_JOBS --download_spoc_data_products $DOWNLOAD_SPOC_DATA_PRODUCTS -external_data_repository $EXTERNAL_DATA_REPOSITORY
```

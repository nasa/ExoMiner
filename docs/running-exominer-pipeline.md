# Running ExoMiner Pipeline

After pulling the Podman image, you are ready to run the pipeline by running the script [run_podman_application.sh](/exominer_pipeline/run_podman_application.sh) in your terminal! The pipeline is fully parallelizable and can make use of multi-CPU cores machines. The TIC IDs are split evenly across a set of jobs defined by the user, meaning that all TCEs for a given TIC ID and sector run are processed in the same job. 

## Command-line arguments
Before you run the pipeline, you have to set the filepaths to the command-line arguments. Run ```podman run exominer_pipeline --help``` for a detailed description on these inputs. The command-line arguments that have a corresponding argument in [pipeline_run_config.yaml](/exominer_pipeline/pipeline_run_config.yaml), if set, will overwrite the values in the YAML file. You can modify the command ```podman run``` to suit your use case (e.g., give TIC IDs as a comma-separated list instead of a CSV file).

## YAML run configuration file

The YAML file [pipeline_run_config.yaml](/exominer_pipeline/pipeline_run_config.yaml) is split into two parts: user-defined parameters at the top. Some of those parameters can be overwritten by setting the corresponding command-line argument; fixed run parameters that are internal to the ExoMiner pipeline and can be modified by more advanced users interested in accessing the application inside the Podman container and modifying the underlying code of the pipeline. 

```yaml
# path to output directory used to save results
output_dir: null
# process results either from TESS SPOC 2-min "2min" or FFI "ffi" data
data_collection_mode: null
# path to CSV file with TIC IDs to generate results for
tic_ids_fp: null
# number of processes to run application in parallel
num_processes: 1
# number of jobs to split the TIC IDs through
num_jobs: 1

### FIXED RUN PARAMETERS ###
# path to configuration file used to preprocess light curve data into a TFRecord dataset
lc_preprocessing_config_fp: exominer_pipeline/config_lc_preprocessing.yaml
# attributes to show for each example in the TFRecord dataset table
data_fields_tfrec_tbl:
  uid: str
  target_id: int
  tce_plnt_num: int
  sector_run: str
# path to configuration file used to preprocess difference image data
diff_img_preprocessing_config_fp: exominer_pipeline/config_diff_img_preprocessing.yaml
# path to configuration file used to add difference image data to TFRecord dataset
diff_img_add_tfrecord_dataset_config_fp: exominer_pipeline/config_add_diff_img_tfrecords.yaml
# path to configuration file used to normalize features in TFRecord dataset
normalize_tfrec_data_config_fp: exominer_pipeline/config_normalize_data.yaml
# path to ExoMiner TensorFlow Keras model used to generate predictions
model_fp: exominer_pipeline/data/model.keras
# model_fp: /model/model.keras
# path to configuration file used for running inference
predict_config_fp: exominer_pipeline/config_predict_model.yaml
```

## Running the Podman container application

The contents of `run_podman_application.sh` are displayed below.

```bash
# directory where the inputs for the ExoMiner Pipeline are stored
inputs_dir="path/to/directory/with/saved/inputs"
# file path to the TICs table
tics_tbl_fp=$inputs_dir/tics_tbl.csv
# file path to the configuration file for the ExoMiner Pipeline run
config_fp=$inputs_dir/pipeline_run_config.yaml
# name of the run
exominer_pipeline_run=exominer_pipeline_run_{date, e.g. 7-10-2025_1925}
# directory where the ExoMiner Pipeline run is saved
exominer_pipeline_run_dir="/path/to/stored_runs/$exominer_pipeline_run"

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

```

## TIC IDs input

You can provide a set of TIC IDs using two methods: 
- create a CSV file with the columns "tic_id" and "sector_run" and set the variable [tics_tbl_fp](#running-the-podman-container-application) to its path.
-  set the variable [tic_ids](#running-the-podman-container-application) to a string in which the TIC IDs are separated by a comma. The following examples showcase the use of these two methods to generate results for the TCEs of TIC 167526485 in single-sector run S6 and multi-sector run S1-39.

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

The following diagram represents the hierarchy of the data output for a run of the ExoMiner Pipeline. In this example, the pipeline was run using one single job and the TESS SPOC TCEs were queried for the 2-min data. The structure was ordered from most recent file/folder to the oldest.

```code
exominer_pipeline_run_{date}
├── run_main.log
├── predictions_outputs.csv
├── pipeline_run_config.yaml
├── tics_tbl.csv
└── job_0
    ├── run_0.log
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
- `predictions_output.csv`: if the run is completed, a CSV file is generated containing the predictions scores produced by the ExoMiner model for the set of TCEs associated with the TIC IDs and sector runs defined in `tics_tbl.csv`. If multiple jobs are completed, it aggregates the predictions generated across them.
- `pipeline_run_config.yaml`: YAML file that stores the run parameters.
- `tics_tbl.csv`: CSV file containing the queried TIC IDs and sector runs.
- `job_{job_id}`: directory containing the results for the TIC IDs and sector runs assigned to the job.
    - `run_{job_id}`: log file for the job.
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
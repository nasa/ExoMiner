# Preprocessing Modules

The [src_preprocessing](/src_preprocessing/) module contains several modules related to the preprocessing of light curve data and difference image data, and the split and normalization of the preprocessed data into a dataset that can be used by ExoMiner models for training, evaluation, hyperparameter optimization, and inference.

**Note**: some of these modules might not be compatible or designed with Kepler in mind since we have moved to work mostly with TESS data.

## Get TESS Target Lightcurve Data

Note that this step involves excluding those targets that were observed but no DV results were generated, and hence no 
TCEs were detected for those targets.

1. Download LC and DV SH files for sector runs of interest. These SH files can be downloaded from the MAST, for example.
2. Use script `src_preprocessing/filter_lc_targets_using_target_table.py` to filter lightcurves for targets of interest.
3. Download LC FITS files for targets of interest using script 
`src_preprocessing/tess_spoc_ffi/download_tess_spoc_ffi_lc_files.sh`.

## Get TESS SPOC Target DV XML data

1. Download DV SH files for sector runs of interest. For 2-min data, those can be obtained from the [MAST](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html). For FFI data, see [TESS SPOC HLSP FFI](https://archive.stsci.edu/hlsp/tess-spoc).
2. Filter DV products in DV SH files to get sh files only with the curl statements for the DV XML files using script [get_curl_tess_spoc_dv_xml_only.sh](../src_preprocessing/tess_spoc_dv_xml/get_curl_tess_spoc_dv_xml_only.sh).
3. Download DV XML files using script [download_tess_spoc_dv_xml_files.sh](../src_preprocessing/tess_spoc_dv_xml/download_tess_spoc_dv_xml_files.sh).

## Lightcurve Preprocessing

The module [lc_preprocessing](/src_preprocessing/lc_preprocessing/) contains scripts related to the pipeline that preprocesses data contained in the targets lightcurve FITS files to generate data for a set of TCEs provided as an input table. See [README.md](/src_preprocessing/lc_preprocessing/README.md) for more details.

## TCE Tables

The module [tce_tables](/src_preprocessing/tce_tables) contains scripts related to the preprocessing of TCE tables, which includes:
- Ephemeris matching
- Adding RUWE values by querying Gaia
- Adding stellar parameters by querying TIC-8
- Extracting TCE data from TESS SPOC DV XML files

See [README.md](/src_preprocessing/tce_tables/README.md) for more details.

## Differece Image

The module [diff_img](/src_preprocessing/diff_img/) contains scripts related to the preprocessing of difference image data contained in the SPOC DV XML files.

## Dataset Split

The module [split_tfrecord_train-test](/src_preprocessing/split_tfrecord_train-test/) contains scripts designed to split the data into training, validation, test, and predict sets. See [README.md](/src_preprocessing/split_tfrecord_train-test/README.md) for more details.

## Data Normalization

The module [normalize_tfrecord_dataset](/src_preprocessing/normalize_tfrecord_dataset/) contains scripts that deal with the computation of normalization statistics as well as applying those to normalize the input features to ExoMiner. See [README.md](/src_preprocessing/normalize_tfrecord_dataset/README.md) for more details.

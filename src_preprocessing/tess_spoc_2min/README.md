# TESS-SPOC 2-min Preprocessing: LC and DV xml data, and TCE tables

This file contains step-by-step instructions on how to get data from the lightcurve (LC) FITS files and Data Validation 
(DV) xml files for TESS SPOC 2-min sector runs, and the corresponding DV TCE tables. All the referred scripts are part 
of the module `src_preprocessing/tess_spoc_2min`.

## Get LC data

Note that this step involves excluding those targets that were observed but no DV results were generated, and hence no 
TCEs were detected for those targets.

1. Download lc and dv sh files for sector runs of interest.
2. Use dv sh files to filter targets in the lc sh files using script 
`src_preprocessing/tess_spoc_2min/filter_dv_targets_lc_sh_tess_2min_data.py`.
3. Download lc fits files for DV targets using script 
`src_preprocessing/tess_spoc_2min/download_tess_spoc_2min_lc_files.sh`.

## Get DV xml data

1. Download dv sh files for sector runs of interest.
2. Filter DV products in dv sh files to get sh files only with the curl statements for the DV xml files using script 
`src_preprocessing/tess_spoc_2min/get_curl_tess_spoc_dv_xml_only.sh`.
3. Download dv xml files using script `src_preprocessing/tess_spoc_2min/download_tess_spoc_2min_dv_xml_files.sh`.


## Get TCE data into csv files

1. Get DV mat files for the sector runs of interest (from Doug Caldwell), and convert them to csv files using script 
  `src_preprocessing/tess_spoc_2min/create_dv_tce_csv_from_mat.py`.
2. Prepare csv tables using script `src_preprocessing/tess_spoc_ffi/create_tess_tce_tbl_from_dv.py`.

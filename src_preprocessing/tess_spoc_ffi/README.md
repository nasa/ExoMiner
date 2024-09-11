# TESS-SPOC FFI Preprocessing: LC and DV xml data, and TCE tables

This file contains step-by-step instructions on how to get data from the lightcurve (LC) FITS files and Data Validation 
(DV) xml files for TESS SPOC FFI sector runs, and the corresponding DV TCE tables. All the referred scripts are part of 
the module `src_preprocessing/tess_spoc_ffi`.

## Get DV xml data

1. Download dv sh files for sector runs of interest.
2. Filter DV products in dv sh files to get sh files only with the curl statements for the DV xml files using script 
`src_preprocessing/tess_spoc_ffi/get_curl_tess_spoc_dv_xml_only.sh`.
3. Download dv xml files using script `src_preprocessing/tess_spoc_ffi/download_tess_spoc_ffi_dv_xml_files.sh`.

## Get LC data

Note that this step involves excluding those targets that were observed but no DV results were generated, and hence no 
TCEs were detected for those targets.

1. Download lc and dv sh files for sector runs of interest.
2. Use dv sh file to filter targets in the lc sh file using script 
`src_preprocessing/tess_spoc_ffi/filter_dv_targets_lc_sh.py`.
3. Download lc fits files for DV targets using script 
`src_preprocessing/tess_spoc_ffi/download_tess_spoc_ffi_lc_files.sh`.



## Get TCE data into csv files

- Option 1 (from DV mat files)
    1. Get DV mat files for the sector runs of interest (from Doug Caldwell), and convert them to csv files using script 
  `src_preprocessing/tess_spoc_ffi/create_dv_tce_csv_from_mat.py`.
    2. Prepare csv tables using script `src_preprocessing/tess_spoc_ffi/create_tess_tce_tbl_from_dv.py`.
- Option 2 (from DV xml files - obtained using the steps in [Get lc and dv xml data](#get-lc-and-dv-xml-data)) 
    1. Get TCE data from DV xml files for sector runs of interest using script 
  `src_preprocessing/tess_spoc_ffi/extract_tce_data_from_dv_xml.py` to create the csv files for each sector run.
    2. Prepare csv tables using script `src_preprocessing/tess_spoc_ffi/xml_tbls_rename_cols_add_uid.py`.

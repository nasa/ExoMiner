# TESS-SPOC FFI Preprocessing: LC and DV XML data

This file contains step-by-step instructions on how to get data from the lightcurve (LC) FITS files and Data Validation 
(DV) XML files for TESS SPOC FFI sector runs. All the referred scripts are part  of the module 
`src_preprocessing/tess_spoc_2min`.

## Get LC data

Note that this step involves excluding those targets that were observed but no DV results were generated, and hence no 
TCEs were detected for those targets.

1. Download LC and DV SH files for sector runs of interest. These SH files can be downloaded from the MAST, for example.
2. Use script `src_preprocessing/filter_lc_targets_using_target_table.py` to filter lightcurves for targets of interest.
3. Download LC FITS files for targets of interest using script 
`src_preprocessing/tess_spoc_ffi/download_tess_spoc_ffi_lc_files.sh`.

## Get DV XML data

1. Download DV SH files for sector runs of interest using, for example, the SH files that can be downloaded from the 
MAST.
2. Filter DV products in DV SH files to get sh files only with the curl statements for the DV XML files using script 
`src_preprocessing/tess_spoc_ffi/get_curl_tess_spoc_dv_xml_only.sh`.
3. Download dv xml files using script `src_preprocessing/tess_spoc_ffi/download_tess_spoc_ffo_dv_xml_files.sh`.

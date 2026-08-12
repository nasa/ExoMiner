# TESS SPOC Preprocessing: LC and DV xml data

This file contains step-by-step instructions on how to get data from the lightcurve (LC) FITS files and Data Validation 
(DV) XML files for TESS SPOC sector runs.

## Get DV XML data

1. Download DV SH files for sector runs of interest using, for example, the SH files that can be downloaded from the 
MAST.
2. Filter DV products in DV SH files to get sh files only with the curl statements for the DV XML files using script 
[get_curl_tess_spoc_dv_xml_only.sh](../tess_spoc_dv_xml/get_curl_tess_spoc_dv_xml_only.sh).
3. Download DV XML files using script [download_tess_spoc_dv_xml_files.sh](../tess_spoc_dv_xml/download_tess_spoc_dv_xml_files.sh).

## Get LC data

Note that this step involves excluding those targets that were observed but no SPOC DV results were generated, and hence no 
TCEs were detected for those targets.

1. Download LC FITS files for sectors of interest. These SH files can be downloaded from the MAST, for example.
2. Use script [filter_lc_targets_using_target_table.py](../filter_lc_targets_using_target_table.py) to filter light curves for targets of interest. This requires a CSV file with columns 'target_id' and 'sector' to filter target SH files.
3. Download LC FITS files for targets of interest using script [download_tess_spoc_dv_xml_files.sh](../download_tess_spoc_dv_xml_files.sh.sh).



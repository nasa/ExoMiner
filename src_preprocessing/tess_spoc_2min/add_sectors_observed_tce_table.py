"""
Add observed sectors from the targets tables with observed sectors to the TCE table.
"""

# 3rd party
from pathlib import Path
import pandas as pd

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
tics_sectors_observed_tbl_fp = Path('/home/msaragoc/Projects/exoplnt_dl/experiments/extract_sectors_observed_from_tess_dv_xml/tess_spoc_2min_tics_sectors_observed.csv')

tce_tbl = pd.read_csv(tce_tbl_fp)
tics_sectors_observed_tbl = pd.read_csv(tics_sectors_observed_tbl_fp)

tce_tbl = tce_tbl.merge(tics_sectors_observed_tbl, on=['target_id', 'sector_run'], how='left', validate='many_to_one')
tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_sectorsobs.csv', index=False)

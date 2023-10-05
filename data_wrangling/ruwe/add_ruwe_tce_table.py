""" Add RUWE values for TICs to TCE table. """

# 3rd party
import pandas as pd
from pathlib import Path

tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_09-25-2023_1608.csv')
ruwe_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/gaiadr2_tics_ruwe/gaiadr2_with_ticid.csv')

tce_tbl = pd.read_csv(tce_tbl_fp)  # load TCE table
ruwe_tbl = pd.read_csv(ruwe_tbl_fp)

tce_tbl = tce_tbl.merge(ruwe_tbl[['target_id', 'ruwe']], on='target_id', how='left', validate='many_to_one')
print(f'Number of TCEs without RUWE value: {tce_tbl["ruwe"].isna().sum()} ouf of {len(tce_tbl)}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ruwe.csv', index=False)

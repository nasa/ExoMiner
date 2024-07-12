""" Add RUWE values for TICs to TCE table. """

# 3rd party
import pandas as pd
from pathlib import Path

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/preprocessed_tce_tables/tess_spoc_ffi_s36-s69_tces_7-8-2024_1647/tess_spoc_ffi_s36-s69_tces_7-8-2024_1647_features_adjusted.csv')
ruwe_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/preprocessed_tce_tables/tess_spoc_ffi_s36-s69_tces_7-8-2024_1647/gaiadr2_tics_ruwe/gaiadr2_with_ticid.csv')

tce_tbl = pd.read_csv(tce_tbl_fp)  # load TCE table
ruwe_tbl = pd.read_csv(ruwe_tbl_fp)

tce_tbl = tce_tbl.merge(ruwe_tbl[['target_id', 'ruwe']], on='target_id', how='left', validate='many_to_one')
print(f'Number of TCEs without RUWE value: {tce_tbl["ruwe"].isna().sum()} ouf of {len(tce_tbl)}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ruwe.csv', index=False)

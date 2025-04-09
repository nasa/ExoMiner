"""
Prepare targets table for neighbors' search using as source data a TCE table.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

# destination file path
target_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/target_sector_pairs_tess_ffi_tces_dv_s36-s72_4-7-2025_0931.csv')
# source table
tce_tbl_fp = '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_fromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942_renamed_cols_added_uid_ruwe_ticstellar_label_features_adjusted.csv'
# load tce table
tce_tbl_cols = [
    'target_id',
    'sectors_observed',
    # 'mag',  # not needed now
    # 'uid'
    # 'sector_run',  # not needed
]
tce_tbl = pd.read_csv(tce_tbl_fp)  # , usecols=tce_tbl_cols)


def _convert_sectors_observed_format(x):

    MAX_NUM_SECTORS = 150

    if '_' in x['sectors_observed'] or len(x['sectors_observed']) != MAX_NUM_SECTORS:
        sectors_lst = [int(sector_str) for sector_str in x['sectors_observed'].split('_')]
        sectors_observed = ''.join(['0' if sector_i not in sectors_lst else '1' for sector_i in
                                    range(MAX_NUM_SECTORS)])
        x['sectors_observed'] = sectors_observed

    return x

# if needed
tce_tbl = tce_tbl.apply(_convert_sectors_observed_format, axis=1)

# get sectors observed for each target in the table
targets_dict = {'target_id': [], 'sector': []}  # , 'mag': []}
targets = tce_tbl.groupby('target_id')
# get observed sectors for each target based on its TCEs
for target_id, target_data in targets:
    target_data = target_data.reset_index(drop=True)
    obs_sectors_target = np.nonzero(np.sum(
        target_data['sectors_observed'].apply(lambda x: np.array([int(el) for el in x])).values, axis=0))[0]
    for obs_sector in obs_sectors_target:
        targets_dict['target_id'].append(target_id)
        targets_dict['sector'].append(obs_sector)
        # targets_dict['mag'].append(target_data.loc[0, 'mag'])

targets_df = pd.DataFrame(targets_dict)
targets_df.to_csv(target_tbl_fp, index=False)

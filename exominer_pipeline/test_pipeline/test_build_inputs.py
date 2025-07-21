"""
Build input CSV file with TIC IDs and sector run.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%%

tics_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs/tics_tbl.csv')
# create CSV file with TICs
twomin = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')
# twomin = twomin.loc[twomin['uid'] == '273574141-1-S14']
twomin = twomin.loc[twomin['sector_run'] == '36']

tics_tbl = twomin[['target_id', 'sector_run']]
tics_tbl['sector_run'] = tics_tbl['sector_run'].apply(lambda x: f'{x}-{x}' if '-' not in x else x)
tics_tbl = tics_tbl.drop_duplicates(subset=['target_id', 'sector_run'])
tics_tbl = tics_tbl.rename(columns={'target_id': 'tic_id'})

tics_tbl = tics_tbl.iloc[:30]

# tics_tbl = pd.DataFrame(
#     data = {
#         'tic_id': [
#             # 167526485,
#             # 167526485,
#             # 167526485,
#             # 184240683,  # non-existing ffi
#             356473034,  # ffi
#         ],
#         'sector_run': [
#             # '6-6',
#             # '7-7',
#             # '1-39',
#             # '29-29',  # non-existing ffi
#             '60-60',
#         ]
#     }
# )

tics_tbl.to_csv(tics_tbl_fp, index=False)
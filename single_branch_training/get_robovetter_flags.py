"""
Get Robovetter flags for Q1-Q17 DR25 TCE catalog.
Notes on Robovetter minor flags for Q1-Q17 DR25 TCEs:
- https://exoplanetarchive.ipac.caltech.edu/docs/koi_comment_flags.html
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path

#%% Read Robovetter Q1-Q17 DR25 TCE table

robovetter_cols = [
    'TCE',
    'Robovetter_Score',
    'Disposition',
    'Not_Transit-Like_Flag',
    'Stellar_Eclipse_Flag',
    'Centroid Offset_Flag',
    'Ephemeris_Match_Flag',
    'Minor_Descriptive_Flags'
]

robovetter_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/robovetter_tables/kplr_dr25_obs_robovetter_output.txt')
robovetter_tbl = pd.read_csv(robovetter_tbl_fp, skiprows=1, names=robovetter_cols, sep=' ', skipinitialspace=False)
robovetter_tbl['uid'] = robovetter_tbl.apply(lambda x: f"{int(x['TCE'].split('-')[0])}-{int(x['TCE'].split('-')[1])}",
                                             axis=1)

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_3-6-2023_1734.csv')

# add label, KOI id from TCE table to Robovetter table
robovetter_tbl = robovetter_tbl.merge(tce_tbl[['uid', 'label', 'kepoi_name']], on='uid', how='left', validate='one_to_one')

robovetter_minor_flags = []
for tce_i, tce in robovetter_tbl.iterrows():

    if isinstance(tce['Minor_Descriptive_Flags'], str):
    # if np.isnan(tce['Minor_Descriptive_Flags']):
        robovetter_minor_flags.extend(tce['Minor_Descriptive_Flags'].split('---'))

robovetter_minor_flags = np.unique(np.array((robovetter_minor_flags)))

#%% Set branch flags based on Robovetter major and minor flags for Q1-Q17 DR25 TCE catalog

branch_flag_cols = [
    'odd_even_flag',
    'sec_flag',
    'centroid_flag',
    'not_transit_like_flag'
]
for col in branch_flag_cols:
    robovetter_tbl[col] = 0

robovetter_tbl.loc[((robovetter_tbl['Stellar_Eclipse_Flag'] == 1) &
                    (robovetter_tbl['Minor_Descriptive_Flags'].str.contains('|'.join(['HAS_SEC_TCE',
                                                                                      'IS_SEC_TCE'])))),
'sec_flag'] = 1

robovetter_tbl.loc[((robovetter_tbl['Stellar_Eclipse_Flag'] == 1) &
                    (robovetter_tbl['Minor_Descriptive_Flags'].str.contains('|'.join(['DEPTH_ODDEVEN_ALT',
                                                                                      'DEPTH_ODDEVEN_DV'])))),
'odd_even_flag'] = 1

robovetter_tbl.loc[((robovetter_tbl['Centroid Offset_Flag'] == 1) &
                    (robovetter_tbl['Minor_Descriptive_Flags'].str.contains('|'.join(['CENT_RESOLVED_OFFSET',
                                                                                      'CENT_UNRESOLVED_OFFSET'])))),
'centroid_flag'] = 1

robovetter_tbl.loc[(robovetter_tbl['Not_Transit-Like_Flag'] == 1), 'not_transit_like_flag'] = 1

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/')
robovetter_tbl.to_csv(save_dir / 'kplr_dr25_obs_robovetter_output.csv', index=False)

for col in branch_flag_cols:
    print(robovetter_tbl[col].value_counts())

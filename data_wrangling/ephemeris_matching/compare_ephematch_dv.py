"""
Comparing ephemeris matching performed using our implementation versus the results from the implementation used in DV.
"""

# 3rd party
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

#%% Check TOIs matched

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/09-28-2022_1139/')
save_dir = exp_dir / 'sector_run_tic_tbls'

match_tbl = pd.read_csv(exp_dir / 'matched_signals.csv')
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi-tce_matching_dv/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_final.csv')
match_tbl = match_tbl.rename(columns={'signal_a': 'uid'})
match_tbl = match_tbl.merge(tce_tbl[['uid', 'toi_dv', 'toi_dv_corr']], on='uid', how='left', validate='one_to_one')

# check if TCEs were matched to different TOIs
diff_match = match_tbl.loc[match_tbl['signal_b'] != match_tbl['toi_dv']]
print(f'Number of TCEs matched with our method that have a different match between us and DV: {len(diff_match)}')

match_tbl_same = match_tbl.loc[match_tbl['signal_b'] == match_tbl['toi_dv']]
f, ax = plt.subplots()
ax.scatter(match_tbl_same['match_corr_coef'], match_tbl_same['toi_dv_corr'], s=8)
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('DV Correlation Coefficient')
f.savefig(save_dir.parent / 'scatter_matched_tois.png')

#%% Check TOIs not matched

tois_in_s14_dv = tce_tbl.loc[(tce_tbl['sector_run'] == '14') & (~tce_tbl['toi_dv'].isna()),
                             ['uid', 'target_id', 'toi_dv', 'toi_dv_corr']]
tois_in_s14_notmatched = tois_in_s14_dv.loc[~tois_in_s14_dv['uid'].isin(match_tbl['uid'])]
print(f'Number of TCEs matched with DV that were not matched with our method: {len(tois_in_s14_notmatched)}')

tois_in_s14_notmatched['match_corr_coef'] = np.nan
dir_no_matches = save_dir.parent / 'tois_not_matched'
dir_no_matches.mkdir(exist_ok=True)
for tce_i, tce in tois_in_s14_notmatched.iterrows():

    match_tbl_tic_fp = save_dir / f'match_tbl_s14_tic_{tce["target_id"]}.csv'

    if not match_tbl_tic_fp.exists():
        print(f'Matching table not found for TIC {tce["target_id"]}')
        continue

    match_tbl_tic = pd.read_csv(match_tbl_tic_fp, index_col=0)
    tois_in_s14_notmatched.loc[tce_i, 'match_corr_coef'] = match_tbl_tic.loc[tce['uid'], str(tce['toi_dv'])]

tois_in_s14_notmatched.to_csv(dir_no_matches / 'not_matches_s14.csv', index=False)

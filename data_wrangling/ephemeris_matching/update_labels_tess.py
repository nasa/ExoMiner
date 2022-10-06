""" Update labels for TESS TCEs using the TOI-TCE matching results and other sources of dispositions. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

#%%

tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

#%% update based on TCE-TOI matchings using our algorithm

match_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/10-05-2022_1621/matched_signals.csv')
match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_toi_our'})

tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

toi_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/10-3-2022/exofop_tess_tois-4.csv')

toi_cols = [
    'TFOPWG Disposition',
    'TESS Disposition',
    'Period (days)',
    'Duration (hours)',
    'Depth (ppm)',
    'Epoch (BJD)',
    'Planet Radius (R_Earth)',
    'Planet SNR',
    'Spectroscopy Observations',
    'Imaging Observations',
    'Time Series Observations',
    'Comments',
]
tce_tbl[toi_cols] = np.nan

tce_tbl['label_source'] = np.nan
# cnt_updates = 0
# cnt_unks_updated = 0
for toi_i, toi in toi_tbl.iterrows():

    tces_found = tce_tbl['matched_toi_our'] == toi['TOI']
    if tces_found.sum() == 0 or not isinstance(toi['TFOPWG Disposition'], str):
        continue

    # cnt_updates += (tce_tbl.loc[tces_found, 'label'] != toi['TFOPWG Disposition']).sum()
    # cnt_unks_updated += ((tce_tbl.loc[tces_found, 'label'] != toi['TFOPWG Disposition']) &
    #                      (tce_tbl.loc[tces_found, 'label'] == 'UNK')).sum()

    tce_tbl.loc[tces_found, ['label', 'TFOPWG Disposition']] = toi['TFOPWG Disposition']
    tce_tbl.loc[tces_found, 'label_source'] = 'TFOPWG Disposition'

# print(f'Number of unlabeled TCEs that changed their disposition: {cnt_unks_updated}')
# print(f'Number of TCEs that changed their disposition: {cnt_updates}')

tce_tbl.loc[tce_tbl['label_source'].isna(), ['label', 'label_source', 'TFOPWG Disposition']] = 'UNK', 'N/A', np.nan
print(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')

# tce_tbl.to_csv(save_dir / f'{tce_tbl_fp.stem}_toidv.csv', index=False)

#%%
# 2) match TCEs to TSO EBs
# tce_tbl.loc[(tce_tbl['tso_eb'] == 'yes') & (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO EB'
tce_tbl.loc[(tce_tbl['in_jon_spoc_ebs'] == 'yes') &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO EB'

# # 3) match TCEs to Jon's EBs
# tce_tbl.loc[(tce_tbl['eb_match_dist'] < matching_thr) &
#             (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'Jon\'s EBs'

# 4) create NTPs based on TEC flux triage
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (~tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)) &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'NTP', 'TEC flux triage'
# set to UNK those TCEs that did not pass the TEC flux triage because they failed AltDet and their period is less or
# equal to 0.3 days
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (tce_tbl['tec_fluxtriage_comment'] == 'AltDetFail') &
            (tce_tbl['label_source'] == 'TEC flux triage') & (tce_tbl['tce_period'] <= 0.3),
            ['label', 'label_source']] = 'UNK', 'N/A'

print(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')
print(f'TCE label source counts after update:\n {tce_tbl["label_source"].value_counts()}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ourmatch.csv', index=False)

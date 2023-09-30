""" Update labels for TESS TCEs using the TOI-TCE matching results and other sources of dispositions. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

#%% Load TCE table

tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_09-25-2023_1608.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

#%% Initialize label field to unknown

tce_tbl['label'] = 'UNK'
tce_tbl['label_source'] = 'N/A'

print(f'TCE disposition counts before any update:\n{tce_tbl["label"].value_counts()}')
print(f'TCE label source counts before any update:\n {tce_tbl["label_source"].value_counts()}')

#%% Add labels from ExoFOP TOI catalog using our TCE-TOI matchings

# load TCE-TOI matching table
match_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/tces_spoc_dv_2mindata_s1-s68_09-25-2023_2028/matched_signals_thr0.75.csv')
match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_toi_our'})

# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

# load TOI table used in matching TCEs with TOIs
toi_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/9-19-2023/exofop_tess_tois.csv', header=1)

# define columns that want to be added from the TOI catalog
toi_cols = [
    'TFOPWG Disposition',
    'TESS Disposition',
    'Period (days)',
    'Duration (hours)',
    'Depth (ppm)',
    'Transit Epoch (BJD)',
    'Planet Radius (R_Earth)',
    'Planet SNR',
    'Spectroscopy Observations',
    'Imaging Observations',
    'Time Series Observations',
    'Comments',
]
tce_tbl[toi_cols] = np.nan  # initialize them to nan

# iterate through TOIs and check which TCEs matched to a given TOI
for toi_i, toi in toi_tbl.iterrows():

    tces_found = tce_tbl['matched_toi_our'] == toi['TOI']
    if tces_found.sum() == 0 or not isinstance(toi['TFOPWG Disposition'], str):
        continue  # no TCEs matched or TOI does not have TFOPWG disposition

    tce_tbl.loc[tces_found, ['label', 'TFOPWG Disposition']] = toi['TFOPWG Disposition']
    tce_tbl.loc[tces_found, 'label_source'] = 'TFOPWG Disposition'

    tce_tbl.loc[tces_found, toi_cols] = toi[toi_cols].values

print(f'TCE disposition counts after ExoFOP TOI update:\n{tce_tbl["label"].value_counts()}')
print(f'TCE label source counts after ExoFOP TOI update:\n {tce_tbl["label_source"].value_counts()}')

# tce_tbl.to_csv(save_dir / f'{tce_tbl_fp.stem}_toidv.csv', index=False)

#%% Add labels from other sources besides TOI catalog

# 2) match TCEs to TSO SPOC EBs
tso_spoc_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/spocEBs_processed.csv')
tce_tbl['in_tso_spoc_ebs'] = 'no'
tce_tbl.loc[tce_tbl['uid'].isin(tso_spoc_tbl['uid']), 'in_tso_spoc_ebs'] = 'yes'
tce_tbl.loc[(tce_tbl['in_tso_spoc_ebs'] == 'yes') &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO SPOC EB'

print(f'TCE disposition counts after TSO SPOC EB update:\n{tce_tbl["label"].value_counts()}')
print(f'TCE label source counts after TSO SPOC EB update:\n {tce_tbl["label_source"].value_counts()}')

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

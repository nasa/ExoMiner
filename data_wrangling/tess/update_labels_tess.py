"""
Update labels for TESS SPOC TCEs using ephemeris matching results against a set of catalogs of objects with
dispositions.

Priority rules are followed to come up with the most comprehensive and clean set of labels possible.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%% Load TCE table

tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# tce_tbl.drop(
#     columns=[
#     # 'TOI',
#     'TFOPWG Disposition',
#     'TESS Disposition',
#     'Period (days)',
#     'Duration (hours)',
#     'Depth (ppm)',
#     'Transit Epoch (BJD)',
#     'Planet Radius (R_Earth)',
#     'Planet SNR',
#     'Spectroscopy Observations',
#     'Imaging Observations',
#     'Time Series Observations',
#     'Comments',
# ] + ['matched_toiexofop', 'match_corr_coef_toiexofop', 'matched_astronet-qlp_tce', 'match_corr_coef_astronet-qlp_tce', 'matched_villanova_ebs', 'match_corr_coef_villanova_ebs', 'tec_fluxtriage_pass', 'tec_fluxtriage_comment', 'label_astronet-qlp']
#              , inplace=True)

# #%% Initialize label field to unknown
#
# print(f'TCE disposition counts before any update:\n{tce_tbl["label"].value_counts()}')
# print(f'TCE label source counts before any update:\n {tce_tbl["label_source"].value_counts()}')

#%% Add dispositions from ExoFOP TOI catalog based on ephemeris matching

# load TCE-TOI matching table
match_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching/tces_spoc_dv_2mindata_s1-s67_exofoptois_1-23-2024/matched_signals_thr0.75_relaxedmatching.csv')
# define columns that want to be added from the TOI catalog
toi_cols = [
    'TOI',
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
# load TOI table used in matching TCEs with TOIs
toi_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/1-23-2024/exofop_toilists.csv', header=0)

match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_toiexofop', 'match_corr_coef': 'match_corr_coef_toiexofop'})
# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

toi_tbl = toi_tbl[toi_cols].rename(columns={'TOI': 'matched_toiexofop'})

# merge tce table with toi table based on matching
tce_tbl = tce_tbl.merge(toi_tbl, on='matched_toiexofop', how='left', validate='many_to_one')

print(f'TCE TFOPWG disposition counts after ExoFOP TOI matching:\n{tce_tbl["TFOPWG Disposition"].value_counts()}')
print(f'TCE TESS disposition counts after ExoFOP TOI matching:\n{tce_tbl["TESS Disposition"].value_counts()}')

#%% Add dispositions from Astronet QLP TCEs based on ephemeris matching

# load TCE-Astronet QLP matching table
match_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching/tces_spoc_dv_2mindata_s1-s67_astronetqlptces_1-24-2024_0949/matched_signals_thr0.75_relaxedmatching.csv')
# define columns that want to be added to the TCE table
toi_cols = [
    'uid',
    'label',
]
# load Astronet QLP TCE table used in matching with TCEs
toi_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/astronet/astronet_training_processed.csv')

match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_astronet-qlp_tce', 'match_corr_coef': 'match_corr_coef_astronet-qlp_tce'})
# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

toi_tbl = toi_tbl[toi_cols].rename(columns={'uid': 'matched_astronet-qlp_tce', 'label': 'label_astronet-qlp'})

# merge tce table with toi table based on matching
tce_tbl = tce_tbl.merge(toi_tbl, on='matched_astronet-qlp_tce', how='left', validate='many_to_one')

print(f'TCE disposition counts after Astronet QLP matching:\n{tce_tbl["label_astronet-qlp"].value_counts()}')

#%% Add dispositions from Villanova's EB based on ephemeris matching

# load TCE-Villanova's EBs matching table
match_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching/tces_spoc_dv_2mindata_s1-s67_villanovaebs_1-25-2024_1235/matched_signals_thr0.75_relaxedmatching.csv')
# define columns that want to be added to the TCE table
toi_cols = [
    'uid',
]
# load Villanova's EBs table used in matching with TCEs
toi_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/ebs_villanova/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat_processed.csv')

match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_villanova_ebs', 'match_corr_coef': 'match_corr_coef_villanova_ebs'})
# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

toi_tbl = toi_tbl[toi_cols].rename(columns={'uid': 'matched_villanova_ebs'})

# merge tce table with toi table based on matching
tce_tbl = tce_tbl.merge(toi_tbl, on='matched_villanova_ebs', how='left', validate='many_to_one')

print(f'TCE disposition counts after Villanova\'s EBs matching:\n{(~tce_tbl["matched_villanova_ebs"].isna()).sum()}')

#%% Add dispositions from TSO-SPOC EBs

tso_spoc_ebs = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_tso/spocEBs_processed.csv')

tce_tbl['in_tso_spoc_ebs'] = 'no'
tce_tbl.loc[tce_tbl['uid'].isin(tso_spoc_ebs['uid']), 'in_tso_spoc_ebs'] = 'yes'

#%% Add dispositions from TEC flux triage

tec_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/TEC_SPOC/tec_tbl_fluxtriage_s1-s41_10-4-2023.csv')
tec_cols = [
    'uid',
    'tec_fluxtriage_pass',
    'tec_fluxtriage_comment',
]

# merge matching results to TCE table
tce_tbl = tce_tbl.merge(tec_tbl[tec_cols], on='uid', how='left', validate='one_to_one')

#%% Set labels of TCEs according to priority rules

# initialize labels as UNK and no label source
tce_tbl['label'] = 'UNK'
tce_tbl['label_source'] = 'None'
tce_tbl['matched_object'] = 'None'

# 1) TFOPWG dispositions from ExoFOP TOI catalog; only those dispositioned as 'KP', 'CP', 'FP'
idxs_matched_exofop = (~tce_tbl['matched_toiexofop'].isna()) & (tce_tbl['label'] == 'UNK') & tce_tbl['TFOPWG Disposition'].isin(['KP', 'CP', 'FP'])
tce_tbl.loc[idxs_matched_exofop, ['label_source']] = 'TFOPWG'
tce_tbl.loc[idxs_matched_exofop, 'label'] = tce_tbl.loc[idxs_matched_exofop, 'TFOPWG Disposition']
tce_tbl.loc[idxs_matched_exofop, 'matched_object'] = tce_tbl.loc[idxs_matched_exofop, 'matched_toiexofop']

# 2) Villanova's EBs
idxs_matched_villanovaebs = (~tce_tbl['matched_villanova_ebs'].isna()) & (tce_tbl['label'] == 'UNK')
tce_tbl.loc[idxs_matched_villanovaebs, ['label_source']] = 'Villanova'
tce_tbl.loc[idxs_matched_villanovaebs, 'label'] = 'EB'
tce_tbl.loc[idxs_matched_villanovaebs, 'matched_object'] = tce_tbl.loc[idxs_matched_villanovaebs, 'matched_villanova_ebs']

# # 3) TSO SPOC EBs
# tce_tbl.loc[(tce_tbl['in_tso_spoc_ebs'] == 'yes') & (tce_tbl['label'] == 'UNK'), ['label_source']] = 'TSO SPOC EBs'
# tce_tbl.loc[tce_tbl['label_source'] == 'TSO SPOC EBs', 'label'] = 'EB'

# # 4) Astronet QLP TCEs; only those dispositioned as 'J' and 'B'
# idxs_matched_astronetqlp = (~tce_tbl['matched_astronet-qlp_tce'].isna()) & (tce_tbl['label'] == 'UNK') & (tce_tbl['label_astronet-qlp'].isin(['J', 'B']))
# tce_tbl.loc[idxs_matched_astronetqlp, ['label_source']] = 'Astronet QLP'
# tce_tbl.loc[idxs_matched_astronetqlp, 'label'] = tce_tbl.loc[idxs_matched_astronetqlp, 'label_astronet-qlp']
# tce_tbl.loc[idxs_matched_astronetqlp, 'matched_object'] = tce_tbl.loc[idxs_matched_astronetqlp, 'matched_villanova_ebs']

# 5) create NTPs based on TEC flux triage; don't include TCEs detected as secondaries of other TCEs
idxs_matched_tec_ntps = (tce_tbl['tec_fluxtriage_pass'] == 0) & (~tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)) & (tce_tbl['label'] == 'UNK')
tce_tbl.loc[idxs_matched_tec_ntps, ['label', 'label_source']] = 'NTP', 'TEC flux triage'
# set to UNK those TCEs that did not pass the TEC flux triage because they failed AltDet and their period is less or
# equal to 0.3 days
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (tce_tbl['tec_fluxtriage_comment'] == 'AltDetFail') &
            (tce_tbl['label_source'] == 'TEC flux triage') & (tce_tbl['tce_period'] <= 0.3),
            ['label', 'label_source']] = 'UNK', 'None'

print(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')
print(f'TCE label source counts after update:\n {tce_tbl["label_source"].value_counts()}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_label.csv', index=False)

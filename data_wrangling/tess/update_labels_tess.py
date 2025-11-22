"""
Update labels for TESS SPOC TCEs using ephemeris matching and other results against a set of catalogs of objects with
dispositions.

Priority rules are followed to come up with the most comprehensive and clean set of labels possible.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%% Load TCE table

tce_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_fromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942_renamed_cols_added_uid_ruwe_ticstellar_label_features_adjusted.csv')

# load TCE table
tce_tbl = pd.read_csv(tce_tbl_fp)

# drop columns in TCE table that were there from previous labeling process
drop_columns = [
    'in_tso_spoc_ebs',
    'matched_toiexofop',
    'match_corr_coef_toiexofop',
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
    'matched_astronet-qlp_tce',
    'match_corr_coef_astronet-qlp_tce',
    'label_astronet-qlp',
    'matched_villanova_ebs',
    'match_corr_coef_villanova_ebs',
    'tec_fluxtriage_pass',
    'tec_fluxtriage_comment',
    'matched_object',
    'sg1_master_disp',
    'exofop_toi',
    'exofop_toi_match_score',
    'Master Disposition',
    'Phot Disposition',
    'Spec Disposition',
    'sg_gaia_ruwe',
    'sg_comments',
]
tce_tbl = tce_tbl.drop(columns=drop_columns, errors='ignore')

#%% Add dispositions from ExoFOP TOI catalog based on ephemeris matching

toi_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/exofop_toilists_4-17-2025_1515_processed_ephem_matching.csv')
match_toi_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/tces_spoc_ffi_s36-s72_exofoptois_4-17-2025_1601/matched_signals_thr0.75.csv')

# load TCE-ExoFOP TOI matching table
match_tbl = pd.read_csv(match_toi_tbl_fp)
# rename columns in match table
match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_toiexofop',
                                      'match_corr_coef': 'match_corr_coef_toiexofop'})
# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

# define columns that want to be added from the TOI catalog
toi_cols = [
    'uid',
    'TFOPWG Disposition',
    'TESS Disposition',
    'period',  # 'Period (days)',
    'duration',  # 'Duration (hours)',
    'Depth (ppm)',
    'epoch',  # 'Epoch (BJD)',
    'Planet Radius (R_Earth)',
    'Planet SNR',
    'Spectroscopy Observations',
    'Imaging Observations',
    'Time Series Observations',
    'Comments',
]
# load ExoFOP TOI table used in matching TCEs with TOIs
toi_tbl = pd.read_csv(toi_tbl_fp, header=0, usecols=toi_cols)
# rename columns in TOI table
toi_tbl = toi_tbl.rename(columns={'uid': 'matched_toiexofop', 'epoch': 'epoch_exofop',
                                  'duration': 'duration_exofop', 'period': 'period_exofop'})
# merge tce table with toi table based on matching
tce_tbl = tce_tbl.merge(toi_tbl, on='matched_toiexofop', how='left', validate='many_to_one')

print(f'TCE TFOP WG disposition counts after ExoFOP TOI matching:\n{tce_tbl["TFOPWG Disposition"].value_counts()}')
print(f'TCE TESS disposition counts after ExoFOP TOI matching:\n{tce_tbl["TESS Disposition"].value_counts()}')

#%% Add dispositions from Prsa's EB based on ephemeris matching

prsa_ebs_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat_processed.csv')
match_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/tces_spoc_ffi_s36-s72_prsaebs_4-17-2025_2110/matched_signals_thr0.75.csv')

# load TCE-Prsa's EBs matching table
match_tbl = pd.read_csv(match_tbl_fp)
match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_villanova_ebs',
                                      'match_corr_coef': 'match_corr_coef_villanova_ebs'})
# merge matching results to TCE table
tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

# define columns that want to be added to the TCE table
prsa_eb_tbl_cols = [
    'uid',
]
# load Villanova's EBs table used in matching with TCEs
prsa_ebs_tbl = pd.read_csv(prsa_ebs_tbl_fp, usecols=prsa_eb_tbl_cols)
prsa_ebs_tbl = prsa_ebs_tbl.rename(columns={'uid': 'matched_villanova_ebs'})
# merge tce table with toi table based on matching
tce_tbl = tce_tbl.merge(prsa_ebs_tbl, on='matched_villanova_ebs', how='left', validate='many_to_one')

print(f'TCEs matched to Prsa\'s EBs:\n{(~tce_tbl["matched_villanova_ebs"].isna()).sum()}')

#%% Add dispositions from TEC flux triage

tec_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tec_fluxtriage_4-15-2025_1330.csv')

tec_cols = [
    'uid',
    'tec_fluxtriage_pass',
    'tec_fluxtriage_comment',
]
tec_tbl = pd.read_csv(tec_tbl_fp, usecols=tec_cols)

# merge matching results to TCE table
tce_tbl = tce_tbl.merge(tec_tbl[tec_cols], on='uid', how='left', validate='one_to_one')

print(f'Number of TCEs found in TEC flux triage: {(~tce_tbl["tec_fluxtriage_pass"]).isna().sum()}')

#%% Set labels of TCEs according to priority rules

# initialize labels as UNK and no label source
tce_tbl['label'] = 'UNK'
tce_tbl['label_source'] = 'None'
tce_tbl['matched_object'] = 'None'

# 1) TFOPWG dispositions from ExoFOP TOI catalog; only those dispositioned as 'KP', 'CP', 'FP'
idxs_matched_exofop = ((~tce_tbl['matched_toiexofop'].isna()) & (tce_tbl['label'] == 'UNK') &
                       tce_tbl['TFOPWG Disposition'].isin(['KP', 'CP', 'FP']))
tce_tbl.loc[idxs_matched_exofop, ['label_source']] = 'TFOPWG'
tce_tbl.loc[idxs_matched_exofop, 'label'] = tce_tbl.loc[idxs_matched_exofop, 'TFOPWG Disposition']
tce_tbl.loc[idxs_matched_exofop, 'matched_object'] = tce_tbl.loc[idxs_matched_exofop, 'matched_toiexofop']

# 2) Prsa's EBs
idxs_matched_villanovaebs = ((~tce_tbl['matched_villanova_ebs'].isna()) &
                             (tce_tbl['label'] == 'UNK') &
                            #  ~tce_tbl['TFOPWG Disposition'].isin(['KP', 'CP', 'FP', 'PC', 'APC', 'FA'])
                             ~tce_tbl['matched_toiexofop'].isna()
                             )
tce_tbl.loc[idxs_matched_villanovaebs, ['label_source']] = 'Villanova'
tce_tbl.loc[idxs_matched_villanovaebs, 'label'] = 'EB'
tce_tbl.loc[idxs_matched_villanovaebs, 'matched_object'] = (
    tce_tbl.loc)[idxs_matched_villanovaebs, 'matched_villanova_ebs']

# 3) create NTPs based on TEC flux triage; don't include TCEs detected as secondaries of other TCEs

# for TESS SPOC 2-min TCEs
idxs_matched_tec_ntps = ((tce_tbl['tec_fluxtriage_pass'] == 0) &
                         (~tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)) &
                         ~tce_tbl['matched_toiexofop'].isna() &
                         (tce_tbl['label'] == 'UNK')
                         )
tce_tbl.loc[idxs_matched_tec_ntps, ['label', 'label_source']] = 'NTP', 'TEC flux triage'
# set to UNK those TCEs that did not pass the TEC flux triage because they failed AltDet and their period is less or
# equal to 0.3 days
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (tce_tbl['tec_fluxtriage_comment'] == 'AltDetFail') &
            (tce_tbl['label_source'] == 'TEC flux triage') & (tce_tbl['tce_period'] <= 0.3),
            ['label', 'label_source']] = 'UNK', 'None'

# # for TESS SPOC FFI TCEs, match to 2-min NTP TCEs
# idxs_matched_tec_ntps = ((~tce_tbl['matched_tecntps'].isna()) &
#                          (~tce_tbl['matched_toiexofop'].isna()))
# tce_tbl.loc[idxs_matched_tec_ntps, ['label', 'label_source']] = 'NTP', 'TEC flux triage'

#%% add SG1 dispositions

sg1_toi_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/sg1/sg1_tois_4-21-2025_1118.csv')

sg1_toi_cols = [
    'TOI',
    'Master Disposition',
    'Phot Disposition',
    'Spec Disposition',
    'Comments',
    'SG2 Notes',
]
sg1_toi_tbl = pd.read_csv(sg1_toi_tbl_fp, usecols=sg1_toi_cols)
sg1_toi_tbl.rename(columns={'Master Disposition': 'sg1_master_disp', 'TOI': 'matched_object',
                            'Phot Disposition': 'sg1_phot_disp', 'Spec Disposition': 'sg1_spec_disp',
                            'Comments': 'sg1_comments', 'SG2 Notes': 'sg2_notes'}, inplace=True,
                   errors='raise')
sg1_toi_tbl['matched_object'] = sg1_toi_tbl['matched_object'].astype('str')
sg1_toi_tbl.drop_duplicates('matched_object', inplace=True)

tce_tbl['matched_object'] = tce_tbl['matched_object'].astype('str')

tce_tbl = tce_tbl.merge(sg1_toi_tbl[['matched_object', 'sg1_master_disp']], how='left', on='matched_object',
                        validate='many_to_one')

# # update labels to NEB, NPC and BD
# idxs_tces = tce_tbl['sg1_master_disp'].isin(['BD', 'NEB', 'NPC'])
# tce_tbl.loc[idxs_tces, 'label'] = tce_tbl.loc[idxs_tces, 'sg1_master_disp']
# tce_tbl.loc[idxs_tces, 'label_source'] = 'SG1'

#%% save TCE table with new matching and labels

print(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')
print(f'TCE label source counts after update:\n {tce_tbl["label_source"].value_counts()}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_label.csv', index=False)

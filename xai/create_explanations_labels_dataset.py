""" Generate CSV file with explanations labels for the TCE dataset. """

# 3rd party
import pandas as pd
from pathlib import Path

#%% set directories

exp_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai')

#%% read CSVs

# TCE table
tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label_dv_mast_urls.csv', usecols=['uid', 'target_id', 'tce_plnt_num', 'sector_run', 'label', 'label_source', 'matched_object', 'sg1_master_disp', 'TFOPWG Disposition', 'TESS Disposition', 'DV mini-report'])
# TEC results
tec_results = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/tec_results_9-2-2025_1440.csv', dtype={'fail_flags_bits': str, 'has_sec_flag': str})
tec_results = tec_results[[tec_col for tec_col in tec_results.columns if tec_col not in ['target_id', 'tce_plnt_num', 'sector_run']]]

#%% merge tables

tce_tec = tce_tbl.merge(tec_results, on='uid', how='left', validate='one_to_one')

#%% generate explanations labels

branches_label_cols = [
    'transitview_flux_branch_label',
    'fullorbit_flux_branch_label',
    'weak_secondary_branch_label',
    'odd_even_branch_label',
    'diff_img_branch_label',
    'centroid_motion_branch_label',
    'periodogram_branch_label',
    'flux_trend_branch_label',
    'unfolded_flux_branch_label',
    'dv_branch_label',
    'stellar_params_branch_label',
]

tce_tec = tce_tec.assign(**{col: 0 for col in branches_label_cols})

def set_explanation_labels_tce(tce):
    """Generate explanations labels for a TCE based on its TEC results.

    :param pandas Series tce: TCE row
    :return pandas Series: TCE row with updated explanations labels
    """

    # use tier 2 results
    tce_tier2_fail_bits = str(tce.get('fail_flags_bits', '') or '')    
    tce_tier2_flag_idxs = [i for i, bit in enumerate(tce_tier2_fail_bits) if bit == '1']

    if 0 in tce_tier2_flag_idxs:  # centroid offset
        tce['diff_img_branch_label'] = 1
    if 1 in tce_tier2_flag_idxs:
        tce[['diff_img_branch_label', 'centroid_motion_branch_label']] = 1
        
    for tier2_sec_flag_idx in [4, 5, 12, 13]:  # weak secondary
        if tier2_sec_flag_idx in tce_tier2_flag_idxs:
            tce['weak_secondary_branch_label'] = 1
            break
    
    for tier2_odd_even_flag_idx in [6, 7]:  # odd-even
        if tier2_odd_even_flag_idx in tce_tier2_flag_idxs:
            tce['odd_even_branch_label'] = 1
            break
    
    # use tier 3 results
    if tce['has_sec_flag'] == 'TRUE':  # secondary
        tce['weak_secondary_branch_label'] = 1
    
    return tce

tce_tec = tce_tec.apply(set_explanation_labels_tce, axis=1)

# use flux triage secondary flag

# get TCEs flagged as secondary of other detected TCEs by TEC
tec_flagged_secondary_of_pn = tce_tec.loc[tce_tec['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)]

def get_primary_uid(tce):
    """Gets primary TCE 'uid' from a TCE flagged as secondary of that primary in TEC flux triage.

    :param pandas Series tce: TCE row
    :return str: primary TCE 'uid'
    """
    
    tce_plnt_num_primary = int(tce['tec_fluxtriage_comment'].split('_')[1])
    primary_uid = f'{tce["target_id"]}-{tce_plnt_num_primary}-S{tce["sector_run"]}'
    
    return primary_uid

# get uid of primary TCEs
tec_flagged_secondary_of_pn['uid_primary'] = tec_flagged_secondary_of_pn.apply(get_primary_uid, axis=1)

# set weak secondary label of primary TCEs
primary_tces_idxs = tce_tec['uid'].isin(tec_flagged_secondary_of_pn['uid_primary'])
tce_tec.loc[primary_tces_idxs, 'weak_secondary_branch_label'] = 1
tce_tec.to_csv(exp_dir / 'tess-spoc-2min_tces_dv_s1-s88_3-27-2025_1316_explanations_labels_9-2-2025_1658.csv', index=False)

# create table with information on primary TCEs of those flagged as secondary TCEs
primary_tces = tce_tec.loc[primary_tces_idxs, ['uid', 'label', 'matched_object']]
primary_tces = primary_tces.rename(columns={col: f'{col}_primary' for col in primary_tces})
tec_flagged_secondary_of_pn = tec_flagged_secondary_of_pn.merge(primary_tces, on='uid_primary', how='left', validate='many_to_one')
tec_flagged_secondary_of_pn.to_csv(exp_dir / 'tess-spoc-2min_tec_secondaries_9-2-2025_1658.csv', index=False)

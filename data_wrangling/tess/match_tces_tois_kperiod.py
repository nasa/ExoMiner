"""

"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% load tables

tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_09-25-2023_1608_ruwe_ticstellar_features_adjusted_label.csv')
obj_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/9-19-2023/exofop_tess_tois.csv', header=1)
match_tbl_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching/ephemeris_matching_tess_spoc_2min_dv_tces/matching_runs/tces_spoc_dv_2mindata_s1-s68_09-25-2023_2028/sector_run_tic_tbls')

#%%

tce_tbl.loc[tce_tbl['targetToiId'] != -1, ['uid',
    'targetToiId',
'planetToiId',
'planetToiCorrelation', 'matched_toiexofop',
'match_corr_coef_toiexofop', 'TFOPWG Disposition', 'TESS Disposition', 'tce_period', 'Period (days)']].to_csv('/Users/msaragoc/Downloads/tces_tess2min_dvtoi.csv', index=False)

#%%

tics_lst = tce_tbl['target_id'].unique()

for tic in tics_lst:

    # get sector matching tables for this target
    sector_match_tbls_fps = [fp for fp in match_tbl_dir.iterdir() if int(fp.stem.split('_')[4]) == tic]

    print(f'Found {len(sector_match_tbls_fps)} tables for target {tic}.')
    for sector_match_tbls_fp in sector_match_tbls_fps:

        sector_match_tbl = pd.read_csv(sector_match_tbls_fp)
        print(f'Found {len(sector_match_tbl)} TCEs matched to {len(sector_match_tbl.columns) - 1} objects.')
        for match_i, match in sector_match_tbl.iterrows():


#%%

fields_cols = ['uid', 'obj_id_closest', 'closest_obj_period', 'period_ratio_k_frac_obj_id_closest', 'all_objs_ids',
               'all_period_ratio_k_frac_objs']
tces_dict = {field: [] for field in fields_cols}
for tce_i, tce in tce_tbl.iterrows():

    if tce_i % 10000 == 0:
        print(f'TCE {tce_i}')

    tces_dict['uid'].append(tce['uid'])
    for field_col in fields_cols[1:]:
        tces_dict[field_col].append(np.nan)

    objs_in_tce_tic = obj_tbl.loc[obj_tbl['TIC ID'] == tce['target_id']]
    # print(f'Found {len(objs_in_tce_tic)} objects in the same TIC ID.')
    if len(objs_in_tce_tic) == 0:
        continue

    # per_ratio = per_tce / per_obj
    # objs_in_tce_tic['tce-toi_period_ratio'] = tce['tce_period'] / objs_in_tce_tic['Period (days)']
    for obj_i, obj in objs_in_tce_tic.iterrows():
        if tce['tce_period'] > obj['Period (days)']:
            objs_in_tce_tic.loc[obj_i, 'period_ratio'] = tce['tce_period'] / obj['Period (days)']
        else:
            objs_in_tce_tic.loc[obj_i, 'period_ratio'] = obj['Period (days)'] / tce['tce_period']
                # per_ratio_k = round(per_tce / per_obj)
    objs_in_tce_tic['period_ratio_k'] = np.round(objs_in_tce_tic['period_ratio'])
    # per_ratio_mod = per_tce / per_obj - round(per_tce / per_obj)
    objs_in_tce_tic['period_ratio_k_frac'] = np.abs(objs_in_tce_tic['period_ratio'] - objs_in_tce_tic['period_ratio_k'])
    objs_in_tce_tic = objs_in_tce_tic.sort_values('period_ratio_k_frac', ascending=True)

    tces_dict['all_objs_ids'][-1] = objs_in_tce_tic['TOI'].values
    tces_dict['all_period_ratio_k_frac_objs'][-1] = objs_in_tce_tic['period_ratio_k_frac'].values
    tces_dict['obj_id_closest'][-1] = objs_in_tce_tic['TOI'].values[0]
    tces_dict['period_ratio_k_frac_obj_id_closest'][-1] = objs_in_tce_tic['period_ratio_k_frac'].values[0]
    tces_dict['closest_obj_period'][-1] = obj_tbl.loc[obj_tbl['TOI'] == tces_dict['obj_id_closest'][-1], 'Period (days)']

tces_df = pd.DataFrame(tces_dict)
tces_df = tces_df.merge(tce_tbl[['uid', 'targetToiId', 'planetToiId','planetToiCorrelation', 'matched_toiexofop',
                           'match_corr_coef_toiexofop', 'TFOPWG Disposition', 'TESS Disposition', 'tce_period',
                           'Period (days)']], on='uid', how='left', validate='one_to_one')

#%% plot histogram of period_ratio_k_frac_obj_id_closest

bins = np.logspace(-2, 0, 100)

f, ax = plt.subplots(2, 1)
ax[0].hist(tces_dict['period_ratio_k_frac_obj_id_closest'], bins, edgecolor='k')
ax[0].set_ylabel('Counts')
ax[1].set_xlabel('period_ratio_k_frac_obj_id_closest')
ax[0].set_xscale('log')
ax[0].set_xlim(bins[[0, -1]])
ax[1].hist(tces_dict['period_ratio_k_frac_obj_id_closest'], bins, edgecolor='k', cumulative=True)
ax[1].set_ylabel('Cumulative Counts')
ax[1].set_xscale('log')
ax[1].set_xlim(bins[[0, -1]])

#%% period matching vs dv

per_match_thr_lst = [np.inf]  # [1e-2, 5e-2, 1e-1, 2e-1, 3e-1]
for per_match_thr in per_match_thr_lst:
    print(per_match_thr)
    tces_df['period_match_same_match_dv'] = tces_df.apply(lambda x: 'yes' if x['planetToiId'] == x['obj_id_closest'] else 'no', axis=1)
    tces_df.loc[tces_df['period_ratio_k_frac_obj_id_closest'] > per_match_thr, 'period_match_same_match_dv'] = 'no'
    # tces_df.loc[((tce_tbl['matched_toiexofop'].isna()) | (tce_tbl['planetToiId'] == -1)), 'same_match_dv'] = 'na'
    tces_df.loc[((tce_tbl['planetToiId'] == -1)), 'period_match_same_match_dv'] = 'na'
    print(tces_df['period_match_same_match_dv'].value_counts())

#%% ephemeris matching vs dv

tces_df['ephem_match_same_match_dv'] = tces_df.apply(lambda x: 'yes' if x['planetToiId'] == x['matched_toiexofop'] else 'no', axis=1)
# tces_df.loc[((tce_tbl['matched_toiexofop'].isna()) | (tce_tbl['planetToiId'] == -1)), 'same_match_dv'] = 'na'
tces_df.loc[((tce_tbl['planetToiId'] == -1)), 'ephem_match_same_match_dv'] = 'na'
print(tces_df['ephem_match_same_match_dv'].value_counts())

#%% period matching vs ephemeris matching

tces_df['ephem_match_same_match_period'] = tces_df.apply(lambda x: 'yes' if x['obj_id_closest'] == x['matched_toiexofop'] else 'no', axis=1)
tces_df.loc[((tces_df['obj_id_closest'].isna()) | tces_df['matched_toiexofop'].isna()), 'ephem_match_same_match_period'] = 'na'
print(tces_df['ephem_match_same_match_period'].value_counts())

#%%
tces_df.to_csv('/Users/msaragoc/Downloads/tess2min_spoctces_toisexofop_periodmatch.csv', index=False)
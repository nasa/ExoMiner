""" Use ephemeris matching results to match TCEs to objects.
"""

#%% 3rd party
import pandas as pd
from pathlib import Path

#%%

match_tbl_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/ephemeris_matchingtces_spoc_ffi_s36-s72_exofoptois_10-8-2025_1224/matched_signals_thr0.75.csv')
tce_tbl_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_fromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942_renamed_cols_added_uid_ruwe_ticstellar_features_adjusted_label.csv')

match_tbl = pd.read_csv(match_tbl_fp)
tce_tbl = pd.read_csv(tce_tbl_fp)

#%% update columns for ExoFOP TOIs

map_columns = {
    'signal_a': 'uid', 
    'signal_b': 'matched_toiexofop', 
    'match_corr_coef': 'match_corr_coef_toiexofop',
    }
match_tbl = match_tbl.rename(columns=map_columns)
match_tbl = match_tbl.set_index('uid')
tce_tbl = tce_tbl.set_index('uid')

print(f'Number of TCEs matched to ExoFOP TOIs (before): {(~tce_tbl["matched_toiexofop"].isna()).sum()}')
print(f'Number of matched ExoFOP TOIs (before): {len(tce_tbl["matched_toiexofop"].value_counts().index.unique())}')
tce_tbl.update(match_tbl)
print(f'Number of TCEs matched to ExoFOP TOIs (after): {(~tce_tbl["matched_toiexofop"].isna()).sum()}')
print(f'Number of matched ExoFOP TOIs (after): {len(tce_tbl["matched_toiexofop"].value_counts().index.unique())}')

#%% save table

tce_tbl.to_csv(tce_tbl_fp)

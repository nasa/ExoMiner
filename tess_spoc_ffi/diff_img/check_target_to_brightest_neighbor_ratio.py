"""
Investigate the distribution of TESS Mag of target to its brightest neighbor.
"""

# 3rd party
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

#%%

neighbors_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/tess_spoc_ffi_s36-s72_search_radius_arcsec_168.0_tpf_wcs_4-7-2025_1322/')
# tce_tbl = pd.read_csv('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')
tce_tbl = pd.read_csv('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_fromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942_renamed_cols_added_uid_ruwe_ticstellar_features_adjusted_label.csv')
experiment_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/')

neighbors_csv_fps = list(neighbors_dir.rglob('neighbors_S*.csv'))
print(f'Found {len(neighbors_csv_fps)} CSV files.')

#%%

neighbors_df_lst = []
for neighbors_csv_fp_i, neighbors_csv_fp in enumerate(neighbors_csv_fps):

    print(f'Iterating over neighbors CSV file {neighbors_csv_fp_i + 1}/{len(neighbors_csv_fps)}...')

    neighbors_df = pd.read_csv(neighbors_csv_fp)
    neighbors_df_min_tmag = neighbors_df.loc[neighbors_df.groupby('target_id')['Tmag'].idxmin()]

    neighbors_df_lst.append(neighbors_df_min_tmag)

neighbors_df = pd.concat(neighbors_df_lst, axis=0, ignore_index=True)
neighbors_df = neighbors_df.merge(tce_tbl[['target_id', 'mag']], on='target_id', how='left', validate='many_to_many')

neighbors_df.rename(columns={'Tmag': 'neighbor_tmag', 'mag': 'target_tmag'}, inplace=True)
neighbors_df['target_to_neighbor_mag'] = neighbors_df['target_tmag'] / neighbors_df['neighbor_tmag']
neighbors_df.to_csv(experiment_dir / 'tess_spoc_ffi_brightest_neighbors.csv')

#%%

bin_mag_ratio = np.linspace(0, 5, 51)
# bin_mag_ratio = np.logspace(-1, 1, 31)

f, ax = plt.subplots()
ax.hist(neighbors_df['target_to_neighbor_mag'], bins=bin_mag_ratio, edgecolor='black')
ax.set_xlabel('Target-to-brightest-neighbor TMag')
ax.set_ylabel('Target-Sector Count')
ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xticks(np.linspace(0, 10, 11))
ax.set_xlim([0, 5])
f.tight_layout()
f.savefig(experiment_dir / 'hist_tess_spoc_ffi_target_to_brightest_neighbor_tmag.png')
plt.show()
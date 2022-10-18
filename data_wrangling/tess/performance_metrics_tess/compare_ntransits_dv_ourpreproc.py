"""
Comparing number of observed transits for TCEs in DV to the number of transits found in our preprocessing pipeline.
"""

# 3rd party
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

#%%

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/Analysis/tess_ntransits_dv_vs_ntransits_ourpreproc')
exp_dir.mkdir(exist_ok=True)

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_smet_ourmatch.csv')
preproc_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/tess/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_data/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624/merged_shards.csv')

preproc_tbl = preproc_tbl.merge(tce_tbl[['uid', 'tce_num_transits_obs', 'label']], on='uid', how='left', validate='one_to_one')

f, ax = plt.subplots()
ax.scatter(preproc_tbl['num_transits_flux'], preproc_tbl['tce_num_transits_obs'], s=8)
ax.set_ylabel('DV TCE Number Observed Transits')
ax.set_xlabel('Our Preprocessing TCE Number Transits')
ax.set_yscale('log')
ax.set_xscale('log')
f.savefig(exp_dir / 'scatter_ntransits_dv_vs_ntransits_ourpreproc.png')
plt.close()

for cat in preproc_tbl['label'].unique():
    preproc_tbl_cat = preproc_tbl.loc[preproc_tbl['label'] == cat]
    f, ax = plt.subplots()
    ax.scatter(preproc_tbl_cat['num_transits_flux'], preproc_tbl_cat['tce_num_transits_obs'], s=8)
    ax.set_ylabel('DV TCE Number Observed Transits')
    ax.set_xlabel('Our Preprocessing TCE Number Transits')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'{cat} TCEs')
    f.savefig(exp_dir / f'scatter_ntransits_dv_vs_ntransits_ourpreproc_{cat}.png')
    plt.close()

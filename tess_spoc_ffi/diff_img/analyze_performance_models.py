"""
Study difference image ExoMiner model performance.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#%%

experiment_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/diff_img/original_diff_img_branch_8-14-2025_1541')

ranking_tbl = pd.read_csv(experiment_dir / 'ranked_predictions_trainset.csv')

#%% plot tce_dikco_msky vs model score

ranking_tbl_aux = ranking_tbl.loc[ranking_tbl['label'].isin(['CP', 'KP', 'NEB', 'NPC'])]
ranking_tbl_aux = ranking_tbl_aux.loc[ranking_tbl_aux['tce_dikco_msky_err'] < 1]

f, ax = plt.subplots()
for label in ranking_tbl_aux['label'].unique():
    ranking_tbl_aux_label = ranking_tbl_aux.loc[ranking_tbl_aux['label'] == label]
    ax.scatter(ranking_tbl_aux_label['tce_dikco_msky'], ranking_tbl_aux_label['score'], s=8, alpha=0.3, label=label)
ax.legend()
ax.set_xlabel('tce_dikco_msky [arcsec]')
ax.set_ylabel('Model Score')
ax.set_xlim([0, 33])
f.tight_layout()

#%% plot tce_dikco_msky

bins_dikco_msky = np.linspace(0, 33, 34)
for label in ranking_tbl['label'].unique():

    f, ax = plt.subplots()
    ax.hist(ranking_tbl.loc[ranking_tbl['label'] == label]['tce_dikco_msky'], bins=bins_dikco_msky, edgecolor='k')
    ax.set_xlabel('tce_dikco_msky [arcsec]')
    ax.set_ylabel('Counts')
    ax.set_title(label)
    ax.set_xlim(bins_dikco_msky[[0, -1]])
    ax.set_yscale('log')
"""
Study difference image ExoMiner model performance.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#%%

experiment_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/diff_img/original_diff_img_branch_regression_tce_dikco_msky_mae_8-14-2025_2157')

ranking_tbl = pd.read_csv(experiment_dir / 'ranked_predictions_trainset.csv')

#%% plot tce_dikco_msky vs model score

ranking_tbl_aux = ranking_tbl.copy()  # ranking_tbl.loc[ranking_tbl['label'].isin(['CP', 'KP', 'NEB', 'NPC'])]
ranking_tbl_aux = ranking_tbl_aux.loc[ranking_tbl_aux['tce_dikco_msky_err'] != -1]
ranking_tbl_aux['tce_dikco_msky_err_rel'] = ranking_tbl_aux['tce_dikco_msky_err'] / ranking_tbl_aux['tce_dikco_msky']
ranking_tbl_aux = ranking_tbl_aux.loc[ranking_tbl_aux['tce_dikco_msky_err_rel'] <= 0.5]
# ranking_tbl_aux = ranking_tbl_aux.loc[ranking_tbl_aux['tce_dikco_msky_err'] < 1]

# f, ax = plt.subplots()
# for label in ranking_tbl_aux['label'].unique():
#     ranking_tbl_aux_label = ranking_tbl_aux.loc[ranking_tbl_aux['label'] == label]
#     ax.scatter(ranking_tbl_aux_label['tce_dikco_msky'], ranking_tbl_aux_label['score'], s=8, alpha=0.3, label=label)
# ax.legend()
# ax.set_xlabel('tce_dikco_msky [arcsec]')
# ax.set_ylabel('Model Score')
# ax.set_xlim([0, 33])
# f.tight_layout()

n_pixels = 1
arcsec_range = n_pixels * 21
tce_dikco_msky_range = np.linspace(0, arcsec_range, 22)
for label in ranking_tbl_aux['label'].unique():
    f, ax = plt.subplots()
    ranking_tbl_aux_label = ranking_tbl_aux.loc[ranking_tbl_aux['label'] == label]
    ax.scatter(ranking_tbl_aux_label['tce_dikco_msky'], ranking_tbl_aux_label['score'], s=8, alpha=0.3, label=label)
    ax.plot([0, arcsec_range], [0, arcsec_range], color='k', linestyle='--', label='Perfect Model')
    ax.legend()
    ax.set_xlabel('tce_dikco_msky [arcsec]')
    ax.set_ylabel('Model Predicted tce_dikco_msky [arcsec]')
    ax.set_xlim([0, arcsec_range])
    ax.set_ylim([0, arcsec_range])
    ax.set_xticks(tce_dikco_msky_range)
    ax.set_yticks(tce_dikco_msky_range)
    ax.set_title(label)
    ax.legend()
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

#%% plot training curves

res_train = np.load(experiment_dir / 'res_train.npy', allow_pickle=True).item()

metric = 'mae'

f, ax = plt.subplots()
ax.plot(res_train[metric], label='Train')
ax.plot(res_train[f'val_{metric}'], label='Validation')
ax.set_xlabel('Epoch Number')
ax.set_ylabel(metric)
ax.set_xlim([0, len(res_train[metric])])
ax.legend()

#%% check examples of larger errors

ranking_tbl_aux['prediction_err'] = np.abs(ranking_tbl['score'] - ranking_tbl['tce_dikco_msky'])
ranking_tbl_aux = ranking_tbl_aux.sort_values(by='prediction_err', ascending=False)


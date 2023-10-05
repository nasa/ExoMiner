""" Plot precision and recall as functions of parameters such as orbital period. """

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#%% Set experiment directory and load ranking table and others

res_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/interns/charles_yates/cv_merged_fluxvar_2-4-2023_1-19-2023/')
ranking_tbl = pd.read_csv(res_dir / 'ensemble_ranked_predictions_allfolds.csv')
# tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv')

#%% precision and recall over period

clf_thr = 0.5  # classification threshold
# bins_period = np.logspace(-1, 3, 10)
bins_period = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20])  # , 25, 30, 50, 100, 200, 500, 700, 900])  # np.linspace(0.1, 27, 10)
nbins = len(bins_period) - 1
rec_bins, prec_bins = np.nan * np.ones(nbins), np.nan * np.ones(nbins)
for bin_i, bin_edges in enumerate(zip(bins_period[:-1], bins_period[1:])):

    # get TCEs in bin
    tces_in_bin = ranking_tbl.loc[((ranking_tbl['tce_period'] >= bin_edges[0]) &
                                   (ranking_tbl['tce_period'] < bin_edges[1]))]

    # compute precision and recall
    rec_bins[bin_i] = ((tces_in_bin['original_label'].isin(['T-KP', 'T-CP'])) &
                       (tces_in_bin['score'] > clf_thr)).sum() / \
                      (tces_in_bin['original_label'].isin(['T-KP', 'T-CP'])).sum()
    prec_bins[bin_i] = ((tces_in_bin['original_label'].isin(['T-KP', 'T-CP'])) &
                        (tces_in_bin['score'] > clf_thr)).sum() / \
                       (tces_in_bin['score'] > clf_thr).sum()

# plot metrics
f, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].bar(bins_period[:-1], rec_bins, edgecolor='k', align='edge', width=np.diff(bins_period))
ax[0].set_ylabel('Recall')
# ax[0].set_xscale('log')
ax[0].set_xlim(bins_period[[0, -1]])
ax[0].set_xticks(bins_period)
ax[1].bar(bins_period[:-1], prec_bins, edgecolor='k', align='edge', width=np.diff(bins_period))
ax[1].set_ylabel('Precision')
# ax[1].set_xscale('log')
ax[1].set_xticks(bins_period)
ax[1].set_xlabel('TCE Period (day)')
ax[1].set_xlim(bins_period[[0, -1]])
f.tight_layout()

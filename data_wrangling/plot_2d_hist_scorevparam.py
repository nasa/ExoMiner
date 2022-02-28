"""
Plot histograms of score vs some parameter.
"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path


res_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/tess_kps_score_2-17-2022')
# res_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_pcs_score_2-17-2022')
res_dir.mkdir(exist_ok=True)

# run = 'PC v non-PC, full model_score'
# run = 'PC v non-PC, no weak secondry_score'
# run = 'PC v non-PC, no transit depth_score'
run = 'PC v non-PC, no transit depth, no weak secondry_score'
# run = 'cv_score'

parameter = 'transit_depth'

# rtbl = pd.read_csv('/Users/msaragoc/Downloads/tess_results_pc-nonpc.csv')
rtbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/interns/hongbo/kepler_to_tess/merged_2-14-2022.csv')
# rtbl = pd.read_csv('/Users/msaragoc/Downloads/ranking_comparison_with_paper_12-18-2020_merged_ra_dec_prad_CV_v15_targetcnts_ruwe_1-4-2022.csv')

tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet.csv')
# tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')

tce_tbl_cols = [
    'target_id',
    'tce_plnt_num',
    'sector_run',
    # 'transit_depth',
    'tce_depth',
    'tce_prad',
    'wst_depth',
    'tce_dikco_msky'
]
rtbl = rtbl.merge(tce_tbl[tce_tbl_cols], on=['target_id', 'tce_plnt_num', 'sector_run'], validate='one_to_one')
# rtbl = rtbl.merge(tce_tbl[tce_tbl_cols], on=['target_id', 'tce_plnt_num'], validate='one_to_one')

rtbl_kps = rtbl.loc[rtbl['original_label'] == 'KP']


# bins = [0, 1, 5, 10, 15, 20]
# hist, bin_edges = np.histogram(rtbl_kps['score'], bins)

# f, ax = plt.subplots()
# ax.bar(bin_edges, hist)
# ax.set_ylabel('Relative Count per bin')

bins_x = np.logspace(2, 5, 4, dtype='int')  # transit depth
# bins_x = [7.1, 10, 15, 20, 50, 100, 250, 500, 750, 1000]  # mes
# bins_x = [100, 250, 500, 750, 1000, 2500, 5000, 10000]  # wst_depth
# bins_x = [0, 100, 250, 500, 750, 1000, 2500, 5000, 10000]  # wst_depth
# bins_x = [1, 5, 10, 15, 20, 30]  # tce prad
# bins_x = [0, 0.5, 1.0, 2.5, 5, 7.5, 10, 15, 20]
bins_y = np.linspace(0, 1, 11)

hist, bin_edges_x, bin_edges_y = np.histogram2d(
    rtbl_kps[parameter],
    rtbl_kps[run],
    bins=[bins_x, bins_y]
)
hist_norm = (hist.T / hist.sum(axis=1)).T * 100

f, ax = plt.subplots(3, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 2, 2]})
ax[0, 0].hist(rtbl_kps[run], bins_y, edgecolor='k')

ax[0, 0].set_ylabel('Counts')
ax[0, 0].set_xticks(bins_y)
ax[0, 0].set_xlim([bins_y[0], bins_y[-1]])
ax[0, 0].grid(axis='y')
ax[0, 1].hist(rtbl_kps[run], bins_y, edgecolor='k', cumulative=True)
ax[0, 1].set_xlabel('Score')
ax[0, 1].set_xticks(bins_y)
ax[0, 1].set_xlim([bins_y[0], bins_y[-1]])
ax[0, 1].grid(axis='y')
ax[0, 1].set_yscale('log')
ax[1, 0].imshow(hist)
for i in range(len(bins_x) - 1):
    for j in range(len(bins_y) - 1):
        ax[1, 0].text(j-0.2, i+0.1, f'{hist[i, j]:.0f}', color='w')
ax[1, 0].set_ylabel(f'{parameter}')
ax[1, 0].set_xticks(np.arange(11) - 0.5)
ax[1, 0].set_yticks(np.arange(len(bins_x)) - 0.5)
ax[1, 0].set_xticklabels(np.round(bins_y, 2))
ax[1, 0].set_yticklabels(bins_x)
# ax[1, 0].set_title(f'{run}')
f.suptitle(f'{run}')

ax[1, 1].barh(y=np.arange(len(bins_x) - 1), width=hist.sum(axis=1), height=np.ones(len(bins_x) - 1), align='edge', edgecolor='k')
# a = ax[1, 1].hist(rtbl_kps[parameter], bins_x, orientation='horizontal', edgecolor='k')
ax[1, 1].set_ylim([0, len(bins_x) - 1])
ax[1, 1].invert_yaxis()
# ax[1, 1].set_yticks(bins_x)
ax[1, 1].set_yticklabels(bins_x)
# ax[1, 1].set_yscale('log')
ax[1, 1].grid(axis='x')
# ax[1, 1].set_yticks(np.arange(len(bins_x)) - 0.5)
# ax[1, 1].set_yticklabels(bins_x)
ax[1, 1].set_xlabel('Counts')
ax[2, 0].imshow(hist_norm)
ax[2, 0].set_xlabel('Score')
ax[2, 0].set_ylabel(f'{parameter}')
for i in range(len(bins_x) - 1):
    for j in range(len(bins_y) - 1):
        ax[2, 0].text(j-0.2, i+0.1, f'{hist_norm[i, j]:.0f}%', color='w')
# ax.grid(axis='both')
ax[2, 0].set_xticks(np.arange(11) - 0.5)
ax[2, 0].set_yticks(np.arange(len(bins_x)) - 0.5)
ax[2, 0].set_xticklabels(np.round(bins_y, 2))
ax[2, 0].set_yticklabels(bins_x)
# ax[0, 1].axis('off')
ax[2, 1].axis('off')
f.tight_layout()
aaa
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
f.savefig(res_dir / f'hist2d_scorevs{parameter}_{run}.png')

aa
#%%
f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(rtbl_kps['transit_depth'], rtbl_kps['wst_depth'], s=8, edgecolor='k')
ax.set_ylabel('TCE wst depth (ppm)')
ax.set_xlabel('TCE depth (ppm)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks([7.1, 10, 15, 20, 100, 250, 500, 750, 1000])
ax.set_yticklabels([7.1, 10, 15, 20, 100, 250, 500, 750, 1000])

ax.set_xticks([250, 500, 750, 1000, 5000, 10000, 25000, 50000, 100000])
ax.set_xticklabels([250, 500, 750, 1000, 5000, 10000, 25000, 50000, 100000], rotation=45)
plt.minorticks_off()
ax.grid(True)
f.savefig(res_dir / 'scatter_tcedepthvstcemes.png')

#%%

f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(rtbl_kps['wst_depth'], rtbl_kps['transit_depth'], s=8, edgecolor='k')
# ax.set_xlabel('TCE prad (R_e)')
ax.set_xlabel('TCE wst depth (ppm)')
ax.set_ylabel('TCE depth (ppm)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks([100, 250, 500, 750, 1000, 2500, 5000, 10000, 100000])
ax.set_yticklabels([100, 250, 500, 750, 1000, 2500, 5000, 10000, 100000])
ax.set_xticks([100, 250, 500, 750, 1000, 2500, 5000, 10000, 100000])
ax.set_xticklabels([100, 250, 500, 750, 1000, 2500, 5000, 10000, 100000], rotation=45)
# ax.set_xticks([1, 5, 10, 15, 20, 30] )
# ax.set_xticklabels([1, 5, 10, 15, 20, 30] , rotation=45)
plt.minorticks_off()
ax.grid(True)
f.savefig(res_dir / 'scatter_tcewstdepthvstcedepth.png')

#%%

from scipy.stats import pearsonr, spearmanr

feat1, feat2 = 'tce_prad', 'tce_prad'

sigma = 2
params_valid = rtbl_kps[np.abs(rtbl_kps - rtbl_kps.mean()) <= (sigma * rtbl_kps.std())]
params_valid = params_valid.loc[(~params_valid[feat1].isna()) & (~params_valid[feat2].isna())]

# param1 = rtbl_kps[feat1].loc[np.abs(rtbl_kps[feat1] - rtbl_kps[feat1].mean()) <= (3 * rtbl_kps[feat1].std())]
# param1 = rtbl_kps[feat2].loc[np.abs(rtbl_kps[feat2] - rtbl_kps[feat1].mean()) <= (3 * rtbl_kps[feat2].std())]
param1, param2 = params_valid[feat1], params_valid[feat2]

pearson_corr = pearsonr(param1, param2)
spearman_corr = spearmanr(param1, param2)

print(f'Pearson corr between {feat1} and {feat2}: {pearson_corr}')
print(f'Spearman corr between {feat1} and {feat2}: {spearman_corr}')

#%%
f, ax = plt.subplots()
histplt = ax.hist2d(rtbl_kps['transit_depth'],
                    rtbl_kps['PC v non-PC, with raw flux parameters_score'],
                    bins=[bins_x, bins_y],
                    edgecolor='k',
                    density=False,
                    norm=LogNorm())#, density=True)
# ax.set_xlabel('Number of PCs per target star')
ax.set_xlabel('Transit depth (ppm)')
ax.set_ylabel('Score')
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_xticks(np.linspace(0.05, 1.05, 10, endpoint=False))
# ax.set_xticklabels(np.linspace(0.1, 1.1, 10, endpoint=False))
# ax.set_xticks(np.arange(1, 21, 2) * 0.05)
# ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# ax.set_xticks(np.arange(0.5, 11.5))
# ax.set_xticklabels(np.arange(0, 11))
# ax.set_yticks(np.arange(0.5, 11.5))
# ax.set_yticklabels(np.arange(0, 11))
# ax.set_ylim(bottom=1)
f.colorbar(histplt[3])


# - Kepler multiplicity boost: changed how number of observations is computed for each scenario (equality instead of greater or equal).
# - Created script data_wrangling.plot_2d_hist_scorevparam.py to plot 2d histograms of score vs some parameter distribution.

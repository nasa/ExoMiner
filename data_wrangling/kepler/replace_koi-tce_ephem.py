"""
Compare the TCE and KOI parameters:
 1) investigate  KOI vs TCE period;
 2) replace TCE by KOI parameters (such as ephemerides).
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#%% Prepare TCE and KOI table with only KOIs

save_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/koi-tce_ephemeris_comparison')

# load Cumulative KOI table
koi_cum_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                          'cumulative_2020.02.21_10.29.22.csv', header=90)
# load TCE table
tce_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)
# remove non-KOIs from the TCE table
tce_tbl = tce_tbl.loc[~tce_tbl['kepoi_name'].isna()].reset_index(inplace=False, drop=True)
print('Number of KOIs in the TCE table: {}'.format(len(tce_tbl)))

# remove KOIs that are not in the TCE table
koi_cum_tbl = koi_cum_tbl.loc[koi_cum_tbl['kepoi_name'].isin(tce_tbl['kepoi_name'])]

assert len(tce_tbl) == len(koi_cum_tbl)

# sort KOI table based on TCE table ordering
# tce_tbl.reset_index(inplace=True)
koi_cum_tbl.set_index('kepoi_name', inplace=True)
koi_cum_tbl = koi_cum_tbl.reindex(index=tce_tbl['kepoi_name'])
koi_cum_tbl.reset_index(inplace=True)


#%% Add KOI parameters to the TCE table

tce_cols = ['tce_period', 'tce_duration', 'tce_time0bk', 'transit_depth', 'ra', 'dec']
koi_cols = ['koi_period', 'koi_duration', 'koi_time0bk', 'koi_depth', 'ra', 'dec']
koi_cols_add = ['koi_period', 'koi_duration', 'koi_time0bk', 'koi_depth', 'koi_ra', 'koi_dec']

tce_tbl = pd.concat([tce_tbl,
                     pd.DataFrame(columns=[col for col in koi_cols_add if col not in tce_tbl.columns])], axis=1)
for tce_i, tce in tce_tbl.iterrows():
    tce_tbl.loc[tce_i, koi_cols_add] = koi_cum_tbl.loc[tce_i, koi_cols].values

tce_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'transit_depth', 'koi_period',
         'koi_duration', 'koi_time0bk', 'koi_depth', 'koi_disposition', 'fpwg_disp_status',
         'label']].to_csv(save_dir / 'q1q17_dr25_tce-koi.csv', index=False)

#%%

tce_tbl = pd.read_csv(save_dir / 'q1q17_dr25_tce-koi.csv')
tce_tbl_original = pd.read_csv(tce_tbl_fp)

tce_tbl['kepoi_name'] = ''
tce_tbl['num_kois_target'] = 1
tce_tbl['num_tces_target'] = 1
tce_tbl['period_koi-tce_ratio'] = -1
tce_tbl['period_category'] = -1
trial_period_ratios = [0.5, 1, 2, 3]
thr_match = 1e-2

for tce_i, tce in tce_tbl.iterrows():

    # get number of TCEs in the same target
    tce_tbl.loc[tce_i, 'num_tces_target'] = len(tce_tbl_original.loc[tce_tbl_original['target_id'] == tce['target_id']])
    # get number of KOIs in the same target
    tce_tbl.loc[tce_i, ['num_kois_target', 'kepoi_name']] = \
        tce_tbl_original.loc[(tce_tbl_original['target_id'] == tce['target_id']) &
                             (tce_tbl_original['tce_plnt_num'] == tce['tce_plnt_num']),
                             ['koi_count', 'kepoi_name']].values[0]
    # compute KOI-TCE period ratio
    tce_tbl.loc[tce_i, 'period_koi-tce_ratio'] = tce['koi_period'] / tce['tce_period']
    # rounding period ratio to the closest factor (0.5, 1, 2, 3, ...)
    period_ratios_diff = np.array([np.abs(tce_tbl.loc[tce_i, 'period_koi-tce_ratio'] - trial_period_ratio)
                                   for trial_period_ratio in trial_period_ratios])

    if np.min(period_ratios_diff * trial_period_ratios) > thr_match:
        continue

    idx_ratio = np.argmin(period_ratios_diff)
    tce_tbl.loc[tce_i, 'period_category'] = trial_period_ratios[idx_ratio]

tce_tbl.to_csv(save_dir / 'q1q17_dr25_tce-koi_periodratio.csv', index=False)

# plot KOI-TCE period ratio
period_ratios = trial_period_ratios + [-1]
plot_tbls = {period_ratio: None for period_ratio in period_ratios}
for period_ratio in period_ratios:
    plot_tbls[period_ratio] = tce_tbl.loc[tce_tbl['period_category'] == period_ratio]
# plot_tbls_colors = {0.5: 'b', 1: 'g', 2: 'r', 3: 'y', -1: 'g'}

f, ax = plt.subplots()
for period_ratio in period_ratios:
    ax.scatter(plot_tbls[period_ratio]['period_koi-tce_ratio'], plot_tbls[period_ratio]['period_category'],
               s=10, c='b')
# ax.legend()
# ax.set_xticks()
ax.set_xlabel('Period KOI-TCE ratio')
ax.set_ylabel('Period Category')
ax.grid(True)
f.savefig(save_dir / 'period_koi-tce_ratio.png')

#%% Plot TCE vs KOI parameters

plot_col_name = ['Orbital Period (day)', 'Transit Duration (hour)', 'Epoch (day)', 'Transit Depth (ppm)', 'RA (deg)',
                 'Dec (deg)']
plot_dict = {'tce_period': {'lim': {'x': [0, 800], 'y': [0, 800]}, 'log': False},
             'tce_duration': {'lim': {'x': [0, 120], 'y': [0, 120]}, 'log': False},
             'tce_time0bk': {'lim': {'x': [120, 600], 'y': [120, 600]}, 'log': False},
             'transit_depth': {'lim': {'x': [1, 1e5], 'y': [1, 1e5]}, 'log': True},
             'ra': {'lim': {'x': None, 'y': None}, 'log': False},
             'dec': {'lim': {'x': None, 'y': None}, 'log': False}
             }

pc_tbl = tce_tbl.loc[tce_tbl['label'] == 'PC']
afp_tbl = tce_tbl.loc[tce_tbl['label'] == 'AFP']
for col_i in range(len(tce_cols)):
    # if tce_cols[col_i] != 'tce_period':
    #     continue
    f, ax = plt.subplots()
    # ax.scatter(tce_tbl[tce_cols[col_i]], koi_cum_tbl[koi_cols[col_i]], s=5, zorder=3)
    ax.scatter(pc_tbl[tce_cols[col_i]], pc_tbl[koi_cols_add[col_i]], s=5, zorder=4, c='b', label='PC', alpha=0.4)
    ax.scatter(afp_tbl[tce_cols[col_i]], afp_tbl[koi_cols_add[col_i]], s=5, zorder=3, c='r', label='AFP')
    if tce_cols[col_i] == 'tce_period':
        ax.plot([0, 360], [0, 720], color='k', label=r'$P_{TCE}/P_{KOI}=0.5$', linestyle='dashed', zorder=1, alpha=0.3)
        ax.plot([0, 720], [0, 360], color='k', label=r'$P_{TCE}/P_{KOI}=2$', linestyle='dashed', zorder=2, alpha=0.3)
        # ax.legend()
    ax.legend()
    ax.set_xlabel('TCE {}'.format(plot_col_name[col_i]))
    if plot_dict[tce_cols[col_i]]['log']:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(plot_dict[tce_cols[col_i]]['lim']['x'])
    ax.set_ylim(plot_dict[tce_cols[col_i]]['lim']['y'])
    ax.set_ylabel('KOI {}'.format(plot_col_name[col_i]))
    ax.grid(True)
    f.savefig(save_dir / '{}.png'.format(plot_col_name[col_i]))
    plt.close()

#%% Plot TCE vs KOI orbital period

f, ax = plt.subplots(figsize=(6, 5))
ax.scatter(tce_tbl['tce_period'], tce_tbl['koi_period'], s=5, zorder=4, c='b', alpha=0.4)
# draw 1/2 and 2 period factor lines
ax.plot([0, 360], [0, 720], color='r', label=r'$Factor = 2$', linestyle='dashed', zorder=1, alpha=0.3)
ax.plot([0, 720], [0, 360], color='g', label=r'$Factor = 0.5$', linestyle='dashed', zorder=2, alpha=0.3)
ax.legend()
ax.set_xlabel('TCE Orbital Period (day)', fontsize=12)
ax.set_xlim([0, 800])
ax.set_ylim([0, 800])
ax.set_ylabel('KOI Orbital Period (day)', fontsize=12)
ax.tick_params(labelsize=12)
ax.grid(True)

#%% Replace TCE parameters by KOI parameters

new_tce_tbl = pd.read_csv(tce_tbl_fp)

cols_to_replace = {'tce_period': 'koi_period', 'tce_time0bk': 'koi_time0bk', 'tce_duration': 'koi_duration',
                   'transit_depth': 'koi_depth'}

for koi_i, koi in koi_cum_tbl.iterrows():

    koi_params = koi[list(cols_to_replace.values())]
    if np.isnan(koi_params['koi_depth']):
        koi_params['koi_depth'] = new_tce_tbl.loc[new_tce_tbl['kepoi_name'] == koi['kepoi_name'], 'transit_depth']

    new_tce_tbl.loc[new_tce_tbl['kepoi_name'] == koi['kepoi_name'], list(cols_to_replace.keys())] = koi_params.values


new_tce_tbl.to_csv('{}_koiephem.csv'.format(tce_tbl_fp[:-4]), index=False)

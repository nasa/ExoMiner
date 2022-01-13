"""
- Compare TCE and TOI periods.
1) between matched TCEs and TOIs.
2) between TCEs and TOIs that could be possibly matched (i.e., they are  part of the same TIC)
- Update TCE ephemerides with TOI ephemerides.
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

res_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/toi_tce_period_1-10-2022')
res_dir.mkdir(exist_ok=True)

# load TCE table
tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label.csv')

# # remove TCEs not matched to TOIs
# tce_tbl = tce_tbl.loc[tce_tbl['label_source'].isin(['TFOPWG Disposition', 'Jon\'s  EBs'])]
# remove TCEs that do not have any TOI/EB in their TIC
tce_tbl = tce_tbl.loc[(~tce_tbl['TOI'].isna()) | (~tce_tbl['eb_bjd0'].isna())]
tce_tbl.loc[tce_tbl['eb_match_dist'].isna(), 'eb_match_dist'] = 2

# select TOI period
tce_tbl['toi_period'] = -1
# tce_tbl.loc[tce_tbl['label_source'] == 'TFOPWG Disposition', 'toi_period'] = tce_tbl.loc[tce_tbl['label_source'] == 'TFOPWG Disposition', 'Period (days)']
# tce_tbl.loc[tce_tbl['label_source'] == 'Jon\'s  EBs', 'toi_period'] = tce_tbl.loc[tce_tbl['label_source'] == 'Jon\'s  EBs', 'eb_period']
for tce_i, tce in tce_tbl.iterrows():
    if tce['match_dist'] <= tce['eb_match_dist']:
        tce_tbl.loc[tce_i, 'toi_period'] = tce_tbl.loc[tce_i, 'Period (days)']
    else:
        tce_tbl.loc[tce_i, 'toi_period'] = tce_tbl.loc[tce_i, 'eb_period']

for tce_i, tce in tce_tbl.iterrows():
    # compute KOI-TCE period ratio
    if tce['toi_period'] >= tce['tce_period']:
        tce_tbl.loc[tce_i, 'period_koi-tce_ratio'] = tce['toi_period'] / tce['tce_period']
    else:
        tce_tbl.loc[tce_i, 'period_koi-tce_ratio'] = - tce['tce_period'] / tce['toi_period']

    # compute KOI-TCE period ratio integer factor
    tce_tbl.loc[tce_i, 'period_koi-tce_ratio_int'] = np.round(tce_tbl.loc[tce_i, 'period_koi-tce_ratio'])
    tce_tbl.loc[tce_i, 'period_koi-tce_ratio_int_rel'] = \
        np.abs((tce_tbl.loc[tce_i, 'period_koi-tce_ratio'] - tce_tbl.loc[tce_i, 'period_koi-tce_ratio_int']) /
               tce_tbl.loc[tce_i, 'period_koi-tce_ratio_int'])

# choose columns for csv file
tbl_cols = [
    'target_id',
    'sector_run',
    'tce_plnt_num',
    'label',
    'TOI',
    'Comments',
    'match_dist',
    'eb_match_dist',
    'tce_period',
    'toi_period',
    'period_koi-tce_ratio',
    'period_koi-tce_ratio_int',
    'period_koi-tce_ratio_int_rel'
]
tce_tbl[tbl_cols].to_csv(res_dir / 'tce_toi_period_nomatchingthr.csv', index=False)

# scatter plot of TCE vs TOI period
f, ax = plt.subplots()
ax.scatter(tce_tbl['tce_period'], tce_tbl['toi_period'], s=5, zorder=3)
# ax.scatter(tce_tbl.loc[tce_tbl['TFOPWG Disposition'].isin(['KP', 'CP'])]['tce_period'],
#            tce_tbl.loc[tce_tbl['TFOPWG Disposition'].isin(['KP', 'CP'])]['toi_period'], s=5, zorder=3, label='KP/CP',
#            c='b', alpha=0.5)
# ax.scatter(tce_tbl.loc[tce_tbl['TFOPWG Disposition'].isin(['FP', 'FA'])]['tce_period'],
#            tce_tbl.loc[tce_tbl['TFOPWG Disposition'].isin(['FP', 'FA'])]['toi_period'], s=5, zorder=2, label='FP/FA',
#            c='r')
# ax.scatter(pc_tbl[tce_cols[col_i]], pc_tbl[koi_cols_add[col_i]], s=5, zorder=4, c='b', label='PC', alpha=0.4)
# ax.scatter(afp_tbl[tce_cols[col_i]], afp_tbl[koi_cols_add[col_i]], s=5, zorder=3, c='r', label='AFP')
ax.plot([0, 360], [0, 720], color='g', label=r'$P_{TCE}/P_{KOI}=0.5$', linestyle='dashed', zorder=1, alpha=0.3)
ax.plot([0, 720], [0, 360], color='k', label=r'$P_{TCE}/P_{KOI}=2$', linestyle='dashed', zorder=2, alpha=0.3)
ax.legend()
ax.set_xlabel('TCE Period (day)')
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim([0, 720])
ax.set_ylim([0, 720])
ax.set_ylabel('TOI Period (day)')
ax.grid(True)
f.savefig(res_dir / 'scatter_tce-toi_period_nomatchingthr.png')
ax.set_xlim([0, 28])
ax.set_ylim([0, 28])
f.savefig(res_dir / 'scatter_tce-toi_period_nomatchingthr_shortperiod.png')
# plt.close()

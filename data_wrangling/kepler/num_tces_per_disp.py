"""
Study the number of TCEs per target star and disposition distribution.
"""

# 3rd  party
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.colors import LogNorm

#%% Plot number of TCEs per target star for each disposition (PC, AFP, NTP)

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/num_tces_per_disp/'
tceTbl = pd.read_csv(os.path.join(saveDir, 'tce_counts_per_target.csv'))

numTcesMin, numTcesMax = tceTbl['num_tces'].min(), tceTbl['num_tces'].max()
bins = np.linspace(numTcesMin, numTcesMax + 1, numTcesMax - numTcesMin + 2, endpoint=True)

dispositions = ['PC', 'AFP', 'NTP']
zorders = {'PC': 3, 'AFP': 2, 'NTP': 1}

f, ax = plt.subplots()
for disp in dispositions:
    ax.hist(tceTbl.loc[tceTbl['label'] == disp]['num_tces'], bins=bins, label=disp, zorder=zorders[disp], edgecolor='k',
            align='left')
ax.set_xlabel('Number of TCEs per target star')
ax.set_ylabel('Count')
ax.set_yscale('log')
ax.set_xticks(bins)
ax.legend()
f.savefig(os.path.join(saveDir, 'hist_num_tces_per_target_star_disp.svg'))

#%% Plot number of PCs per number of TCEs in target star

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/num_tces_per_disp/'

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')
targetTbl = tceTbl.groupby('target_id').count()[['tce_plnt_num']]

targetTbl['num_PCs'] = -1

for target_i, target in targetTbl.iterrows():

    targetTbl.loc[target_i, 'num_PCs'] = len(tceTbl.loc[(tceTbl['target_id'] == target.name) &
                                                        (tceTbl['label'] == 'PC')])

targetTbl.to_csv(os.path.join(saveDir, 'num_PCs_per_target.csv'), index=False)

#%%

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/num_tces_per_disp/'
targetTbl = pd.read_csv(os.path.join(saveDir, 'num_PCs_per_target.csv'))

# numPCsMin, numPCsMax = targetTbl['num_PCs'].min(), targetTbl['num_PCs'].max()
# binspc = np.linspace(0, 11, 12, endpoint=True)
binspc = np.linspace(0, 1, 11, endpoint=True)
binstce = np.linspace(1, 11, 11, endpoint=True)

f, ax = plt.subplots()
# histplt = ax.hist2d(targetTbl['num_PCs']/targetTbl['tce_plnt_num'] * 100, targetTbl['tce_plnt_num'], bins=[bins, bins], edgecolor='k', norm=LogNorm(), density=False)
histplt = ax.hist2d(targetTbl['num_PCs'] / targetTbl['tce_plnt_num'], targetTbl['tce_plnt_num'], bins=[binspc, binstce], edgecolor='k', norm=LogNorm())#, density=True)
# ax.set_xlabel('Number of PCs per target star')
ax.set_xlabel('Fraction of PCs per target star')
ax.set_ylabel('Number of TCEs per target star')
# ax.set_yscale('log')
# ax.set_xticks(np.linspace(0.05, 1.05, 10, endpoint=False))
# ax.set_xticklabels(np.linspace(0.1, 1.1, 10, endpoint=False))
ax.set_xticks(np.arange(1, 21, 2) * 0.05)
ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# ax.set_xticks(np.arange(0.5, 11.5))
# ax.set_xticklabels(np.arange(0, 11))
ax.set_yticks(np.arange(0.5, 11.5))
ax.set_yticklabels(np.arange(0, 11))
ax.set_ylim(bottom=1)
f.colorbar(histplt[3])

# f.savefig(os.path.join(saveDir, 'hist2d_num_tces_numpcs_per_target_star_colorlognorm.svg'))
# f.savefig(os.path.join(saveDir, 'hist2d_num_tces_fractionpcs_per_target_star_colorlognorm.svg'))

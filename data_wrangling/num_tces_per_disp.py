"""
Study the number of TCEs per target star based on the disposition of the TCEs
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

#%%

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

""" Compare orbital period between Q1-Q17 DR24 and DR25 Kepler TCE releases. """

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tce_tbl_dr25 = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar.csv')
tce_tbl_dr24 = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/q1_q17_dr24_tce_2020.03.02_17.51.43_stellar.csv')

# bins = np.logspace(-1, 3, 100)
# bins = np.linspace(0, 800, 100)
bins = np.geomspace(0.5, 800, 100)
f, ax = plt.subplots()
ax.hist(tce_tbl_dr25['tce_period'], bins=bins, label='Q1-Q17 DR25', histtype='step', fill=False, color='b')
ax.hist(tce_tbl_dr24['tce_period'], bins=bins, label='Q1-Q17 DR24', histtype='step', fill=False, color='tab:orange')
ax.set_xscale('log')
ax.set_xticks([0.5, 1., 3, 10, 30, 100, 372])
ax.set_xticklabels([0.5, 1., 3, 10, 30, 100, 372])
ax.set_xlabel('Orbital Period (day)', fontsize=14)
ax.set_ylabel('Number of TCEs', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(loc=2)
ax.minorticks_off()

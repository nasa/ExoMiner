""" Comparing the orbital period of KOIs and TOIs. """

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

save_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/compare_koi-toi_periods')

toi_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/3-11-2021/exofop_toilists_spoc_nomissingpephem_sectors.csv')
koi_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec.csv')

# remove non-KOIs
koi_tbl = koi_tbl.loc[~koi_tbl['kepoi_name'].isna()]

# get only CONFIRMED, CERTIFIED FP, CERTIFIED FA KOIs
koi_tbl = koi_tbl.loc[(koi_tbl['koi_disposition'] == 'CONFIRMED') | (koi_tbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA']))]

# bins_koi = np.logspace(-3, 0, 10)
# bins_toi = np.logspace(-3, 0, 10)
bins_koi = np.linspace(0, 1, 100)
bins_toi = np.linspace(0, 1, 100)

f, ax = plt.subplots()
ax.hist(koi_tbl['tce_period'], bins_koi, label='KOIs', density=False, histtype='step')
ax.hist(toi_tbl['tce_period'], bins_toi, label='TOIs', density=False, histtype='step')
ax.legend()
ax.set_ylabel('Normalized Counts')
ax.set_xlabel('Orbital Period (day)')
ax.set_xlim([0, 1])
# ax.set_yscale('log')
f.savefig(save_dir / 'hist_period_xlogspace_densityy_ylog_1d.png')

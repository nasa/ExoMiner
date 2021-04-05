"""
Investigate whether there is a relationship between Kepler magnitude and difference image centroid offset estimates.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#%%

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_mag_dv_oot_co_kic')
res_dir.mkdir(exist_ok=True)

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

dispositions = ['PC', 'AFP', 'NTP']
disp_col = 'label'
feature_col = 'tce_dicco_msky'
for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(tce_tbl.loc[tce_tbl[disp_col] == disp, 'mag'],
               tce_tbl.loc[tce_tbl[disp_col] == disp, feature_col], s=8)
    ax.set_ylabel(f'{feature_col}')
    ax.set_xlabel('Kepler magnitude')
    ax.set_title(f'{disp}')
    ax.set_yscale('log')
    ax.set_ylim([1e-2, 1e2])
    # ax.set_xlim(right=13)
    f.savefig(res_dir / f'kep_mag_{feature_col}_scatter_{disp}.png')
    plt.close()

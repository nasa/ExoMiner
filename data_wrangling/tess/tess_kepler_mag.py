"""
Compare Kepler and TESS magnitude to target stars that are both observed in the two missions.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs

#%%

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_tess_mag')

#%% Match KIC and TICs using TESS and Kepler TCE tables

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/tce_table_03-22-2021_1733/tess_tce_s1-s34_thr0.25_renamedcols.csv')

# remove TCEs without KIC
tce_tbl = tce_tbl.loc[~tce_tbl['kic_id'].isna()]
print(f'Number of TCEs in the TCE table from a TIC that is also a KIC: {len(tce_tbl)}')

# remove duplicates
tce_tbl.drop_duplicates(subset='target_id', inplace=True, keep='first', ignore_index=True)
print(f'Number of TICs in the TCE table that are also a KIC: {len(tce_tbl)}')

# initialize Kepler mag column
tce_tbl['kepmag'] = np.nan

kic_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar.csv')

for tic_i, tic_id in tce_tbl.iterrows():
    kic_found = kic_tbl.loc[kic_tbl['kepid'] == tic_id['kic_id']]
    if len(kic_found) == 1:
        tce_tbl.loc[tic_i, 'kepmag'] = kic_found['kepmag'].values[0]

print(f'Number of TIC/KICs with both TESS and Kepler magnitudes: {len(tce_tbl.loc[(~tce_tbl["mag"].isna()) & (~tce_tbl["kepmag"].isna())])}')
tce_tbl[['target_id', 'sectors', 'mag', 'kic_id', 'kepmag']].to_csv(res_dir / 'kic-tic_mag_tess_tces.csv', index=False)

#%% Plot Kepler mag vs TESS mag

f, ax = plt.subplots()
ax.scatter(tce_tbl['mag'], tce_tbl['kepmag'], edgecolor='k', s=8)
ax.plot([5, 21], [5, 21], 'r--')
ax.set_ylabel('Kepler Mag')
ax.set_xlabel('TESS Mag')
ax.grid(True)
ax.set_xlim(5, 21)
f.savefig(res_dir / 'scatter_tess-kepler_mag.png')

#%% Match KICs and TICs using Q1-Q17 DR25 Stellar, Supplemental, Gaia DR2 and TIC (astroquery)

tic_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp_gaiadr2.csv')
print(f'Total number of KICs in table: {len(tic_tbl)}')

# remove TICs without Kepler mag
tic_tbl = tic_tbl.loc[~tic_tbl['kepmag'].isna()]
print(f'Number of KICs with Kepler magnitude value: {len(tic_tbl)}')

tic_fields = ['ID', 'Tmag', 'Kmag']
for col in tic_fields:
    tic_tbl[col] = np.nan

kepid_list = tic_tbl['kepid'].tolist()
k_i, q_len = 0, 1000
while True:
    print(f'Checking KICs {k_i}-{k_i + q_len}')
    catalog_data = Catalogs.query_criteria(catalog='TIC', KIC=kepid_list[k_i:k_i + q_len]).to_pandas()
    if k_i + q_len >= len(kepid_list):
        break
    for tic_i, tic in catalog_data.iterrows():
        tic_tbl.loc[tic_tbl['kepid'] == int(tic['KIC']), tic_fields] = tic[tic_fields].values
    k_i += q_len

print(f'Number of targets with both valid Kepler mag and TESS mag: {len(tic_tbl.loc[(~tic_tbl["kepmag"].isna()) & (~tic_tbl["Tmag"].isna())])}')

tic_tbl[['kepid', 'kepmag'] + tic_fields].to_csv(res_dir / 'kic-tic_mag_kic.csv', index=False)

#%% Plot Kepler mag vs TESS mag

f, ax = plt.subplots()
ax.scatter(tic_tbl['Tmag'], tic_tbl['kepmag'], edgecolor=None, s=8)
ax.plot([5, 21], [5, 21], 'r--')
ax.set_ylabel('Kepler Mag')
ax.set_xlabel('TESS Mag')
ax.grid(True)
ax.set_xlim(5, 21)
f.savefig(res_dir / 'scatter_tess-kepler_mag.png')

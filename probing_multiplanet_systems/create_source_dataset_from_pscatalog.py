"""
Create source data set from Planetary Systems Catalog.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from astropy.table import Table
import matplotlib.pyplot as plt

#%%

root_dir = Path('/Users/msaragoc/OneDrive-NASA (Archive)/Projects/exoplanet_transit_classification/experiments/multi_plnt_sys_probing')

#%% Create directory for the data set

experiment = f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
print(f'Started creating data set {experiment}...')
dataset_dir = root_dir / 'processed_datasets' / experiment
dataset_dir.mkdir(exist_ok=True)

#%% Get planet data from a source catalog

plnt_cat_fp = root_dir / 'source_datasets' / 'PSCompPars_2023.06.12_16.52.00.csv'
print(f'Using as source data set for planet catalog {plnt_cat_fp}')
plnt_cat = pd.read_csv(plnt_cat_fp, header=320)
print(f'Total number of observations: {len(plnt_cat)}')

#%%

# # get a single observation per planet
# plnt_cat = plnt_cat.drop_duplicates(subset='pl_name')

# remove planets with no TIC ID associated
plnt_cat = plnt_cat.loc[~plnt_cat['tic_id'].isna()]

# remove planets from single-planet systems
tics = plnt_cat['tic_id'].value_counts()
# single_tics = tics.loc[tics == 1].index.to_numpy()
multi_tics = tics.loc[tics > 1].index.to_numpy()

multi_plnt_cat = plnt_cat.loc[plnt_cat['tic_id'].isin(multi_tics)]

# excluding circumbinary planets (16 planets)
multi_plnt_cat = multi_plnt_cat.loc[multi_plnt_cat['cb_flag'] != 1]

# exclude multi-planetary systems with planetts with radius > 5 Re
tics_plnts_highrad = multi_plnt_cat.loc[multi_plnt_cat['pl_orbper'] > 5, 'tic_id']
a = multi_plnt_cat.loc[~multi_plnt_cat['tic_id'].isin(tics_plnts_highrad)]

#%% check missing features

features = [
    'pl_orbper',
    'pl_rade',
    'pl_orbincl',
    'pl_bmasse',
    'st_teff',
    'st_rad',
    'st_met',
    'st_logg',
    # 'st_dens',
    'st_mass',
]

for feature in features:
    print(f'Checking feature {feature}')
    print(f'Number of NaN values: {multi_plnt_cat[feature].isna().sum()}')
    print(multi_plnt_cat[feature].describe())


# # exclude KOIs that are not planets
# koi_cat = koi_cat.loc[koi_cat['koi_disposition'] == 'CONFIRMED']
# print(f'Number of Confirmed KOIs (planets): {len(koi_cat)}')
# print(f'Number of targets: {len(koi_cat["kepid"].unique())}')
# print(f'Number of targets with N planets:\n{koi_cat.groupby("kepid").count()["kepoi_name"].value_counts()}')
# koi_cat.to_csv(dataset_dir / 'confirmed_kois.csv', index=False)

f, ax = plt.subplots()
ax.hist(multi_plnt_cat['pl_orbper'], np.logspace(-1, 4, 100), edgecolor='k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Orbital Period (day)')
ax.set_ylabel('Counts')

f, ax = plt.subplots()
ax.hist(multi_plnt_cat['pl_rade'], np.logspace(-1, 3, 100), edgecolor='k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e-1, 40])
ax.set_xlabel('Planet Radius (Earth Radius)')
ax.set_ylabel('Counts')

f, ax = plt.subplots()
ax.hist(multi_plnt_cat['pl_orbincl'], np.linspace(0, 180, 100), edgecolor='k')
ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlim([1e-1, 40])
ax.set_xlabel('Planet Inclination (deg)')
ax.set_ylabel('Counts')

f, ax = plt.subplots()
ax.hist(multi_plnt_cat['pl_bmasse'], np.logspace(-2, 5, 200), edgecolor='k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Planet Mass (Earth Mass)')
ax.set_ylabel('Counts')

#%% Update stellar parameters using Gaia

kepler_dr3_fits_fp = root_dir / 'source_datasets' / 'kepler_dr3_good.fits'
print(f'Getting Gaia DR3 solutions from {kepler_dr3_fits_fp}...')
kepler_dr3_fits = Table.read(kepler_dr3_fits_fp, format='fits')
kepler_dr3_fits = kepler_dr3_fits.to_pandas()
print(f'Number of targets in {kepler_dr3_fits_fp}: {len(kepler_dr3_fits)}')

print('Merging table with planet catalog...')
koi_cat = koi_cat.merge(kepler_dr3_fits, on='kepid', validate='many_to_one')

koi_cat.to_csv(dataset_dir / 'confirmed_kois_gaia_stellar.csv', index=False)

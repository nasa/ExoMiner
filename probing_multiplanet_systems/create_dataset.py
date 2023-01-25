"""
Create data set.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from astropy.table import Table

#%% Create directory for the data set

experiment = f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
print(f'Started creating data set {experiment}...')
root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/multi_plnt_sys_probing/')
dataset_dir = root_dir / 'processed_datasets' / experiment
dataset_dir.mkdir(exist_ok=True)

#%% Get planet data from a source catalog

koi_cat_fp = root_dir / 'source_datasets' / 'cumulative_2023.01.23_11.04.46.csv'
print(f'Using as source data set for planet catalog {koi_cat_fp}')
koi_cat = pd.read_csv(koi_cat_fp, header=102)
print(f'Total number of KOIs: {len(koi_cat)}')

# exclude KOIs that are not planets
koi_cat = koi_cat.loc[koi_cat['koi_disposition'] == 'CONFIRMED']
print(f'Number of Confirmed KOIs (planets): {len(koi_cat)}')
print(f'Number of targets: {len(koi_cat["kepid"].unique())}')
print(f'Number of targets with N planets:\n{koi_cat.groupby("kepid").count()["kepoi_name"].value_counts()}')
koi_cat.to_csv(dataset_dir / 'confirmed_kois.csv', index=False)

#%% Update stellar parameters using Gaia

kepler_dr3_fits_fp = root_dir / 'source_datasets' / 'kepler_dr3_good.fits'
print(f'Getting Gaia DR3 solutions from {kepler_dr3_fits_fp}...')
kepler_dr3_fits = Table.read(kepler_dr3_fits_fp, format='fits')
kepler_dr3_fits = kepler_dr3_fits.to_pandas()
print(f'Number of targets in {kepler_dr3_fits_fp}: {len(kepler_dr3_fits)}')

print('Merging table with planet catalog...')
koi_cat = koi_cat.merge(kepler_dr3_fits, on='kepid', validate='many_to_one')

koi_cat.to_csv(dataset_dir / 'confirmed_kois_gaia_stellar.csv', index=False)

#%% Check for missing values

print('Checking features with missing values...')
features = [
    # star
    'mass',
    'feh',
    'radius',
    'logg',
    'teff',
    # planet
    'koi_period',
    'koi_prad',
    'koi_incl',
]

for feat in features:
    print(f'Number of examples with missing value for feature {feat}: {koi_cat[f"{feat}"].isna().sum()}')

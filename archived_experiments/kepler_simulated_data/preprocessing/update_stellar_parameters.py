"""
Updated stellar parameters in the Kepler Simulated DV TCE tables.

- Use Gaia DR3 parameters.
- Use KIC parameters from Q1-Q17 DR25 Stellar Supplemental Catalog.
- Set categorical magnitude based on Kepler magnitude and threshold.
"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table

#%% Getting stellar parameters from Kepler Gaia DR3

kepler_dr3_fits_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/gaia/kepler_dr3_good.fits')

print(f'Getting Gaia DR3 solutions from {kepler_dr3_fits_fp}...')
kepler_dr3_fits = Table.read(kepler_dr3_fits_fp, format='fits')
kepler_dr3_fits = kepler_dr3_fits.to_pandas()
print(f'Number of targets in {kepler_dr3_fits_fp}: {len(kepler_dr3_fits)}')

tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed.csv')

targets_in_tce_tbl = tce_tbl['target_id'].isin(kepler_dr3_fits['kepid'])
n_targets_in_tce_tbl = len(tce_tbl.loc[targets_in_tce_tbl, 'target_id'].unique())
print(f'Number of KICs in Gaia DR3 that are in the TCE table: {n_targets_in_tce_tbl} '
      f'out of {len(tce_tbl["target_id"].unique())}')
print(f'Number of associated TCEs: {targets_in_tce_tbl.sum()} out of {len(tce_tbl)}')

tces_kics_not_in_gaiadr3 = tce_tbl.loc[~targets_in_tce_tbl]
print(tces_kics_not_in_gaiadr3['dataset'].value_counts())

# rename Gaia DR3 catalog columns
gaia_dr3_cols = {
    'kepid': 'target_id',
    # name must match {final_name}_{source_name}
    'ra': 'ra_gaiadr3',
    'dec': 'dec_gaiadr3',
    'ruwe': 'ruwe_gaiadr3',
    'kepmag': 'mag_gaiadr3',
    'teff': 'tce_steff_gaiadr3',
    'logg': 'tce_slogg_gaiadr3',
    'feh': 'tce_smet_gaiadr3',
    'radius': 'tce_sradius_gaiadr3',
    'mass': 'tce_smass_gaiadr3',
}

kepler_dr3_fits_sub = kepler_dr3_fits[gaia_dr3_cols.keys()]
kepler_dr3_fits_sub = kepler_dr3_fits_sub.rename(columns=gaia_dr3_cols)

print('Merging table with TCE catalog...')
tce_tbl = tce_tbl.merge(kepler_dr3_fits_sub, on='target_id', how='left', validate='many_to_one')

# tce_tbl.to_csv(dataset_dir / 'confirmed_kois_gaia_stellar.csv', index=False)

#%% Getting stellar parameters from Kepler Q1-Q17 DR25 stellar + supplemental

q1q17dr25_stellar_supp = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/q1_q17_dr25_stellar_plus_supp.csv')

targets_in_tce_tbl = tce_tbl['target_id'].isin(q1q17dr25_stellar_supp['kepid'])
n_targets_in_tce_tbl = len(tce_tbl.loc[targets_in_tce_tbl, 'target_id'].unique())
print(f'Number of KICs in Kepler Q1-Q17 DR25 stellar + supplemental that are in the TCE table: {n_targets_in_tce_tbl} '
      f'out of {len(tce_tbl["target_id"].unique())}')
print(f'Number of associated TCEs: {targets_in_tce_tbl.sum()} out of {len(tce_tbl)}')

tces_kics_not_in_q1q17dr25_stellar_supp = tce_tbl.loc[~targets_in_tce_tbl]
print(tces_kics_not_in_q1q17dr25_stellar_supp['dataset'].value_counts())

# rename KIC catalog columns
q1q17dr25_stellar_supp_cols = {
    'kepid': 'target_id',
    'ra': 'ra_q1q7dr25_stellar_supp',
    'dec': 'dec_q1q7dr25_stellar_supp',
    # 'ruwe': 'ruwe_gaiadr3',
    'kepmag': 'mag_q1q7dr25_stellar_supp',
    'teff': 'tce_steff_q1q7dr25_stellar_supp',
    'logg': 'tce_slogg_q1q7dr25_stellar_supp',
    'feh': 'tce_smet_q1q7dr25_stellar_supp',
    'radius': 'tce_sradius_q1q7dr25_stellar_supp',
    'mass': 'tce_smass_q1q7dr25_stellar_supp',
    'dens': 'tce_sdens_q1q7dr25_stellar_supp',
}

q1q17dr25_stellar_supp_sub = q1q17dr25_stellar_supp[q1q17dr25_stellar_supp_cols.keys()]
q1q17dr25_stellar_supp_sub = q1q17dr25_stellar_supp_sub.rename(columns=q1q17dr25_stellar_supp_cols)

print('Merging table with TCE catalog...')
tce_tbl = tce_tbl.merge(q1q17dr25_stellar_supp_sub, on='target_id', how='left', validate='many_to_one')

#%% Rename field using best set of stellar parameters

param_final_cols_names = [
    'ra',
    'dec',
    'mag',
    'tce_steff',
    'tce_smet',
    'tce_smass',
    'tce_sradius',
    'tce_sdens',
    'tce_slogg',
    'ruwe',
]
tce_tbl = pd.concat([tce_tbl,
                     pd.DataFrame(np.nan * np.ones((len(tce_tbl), len(param_final_cols_names)), dtype='float'),
                                  columns=param_final_cols_names)],
                    axis=1)

# update final set of parameters using the best estimate
cnts_sources_params = {param: [] for param in param_final_cols_names}
for tce_i, tce in tce_tbl.iterrows():

    for param in param_final_cols_names:

        # 1) check if Gaia DR3 values are available
        if f"{param}_gaiadr3" in gaia_dr3_cols.values() and not np.isnan(tce[f"{param}_gaiadr3"]):
            tce_tbl.loc[tce_i, param] = tce[f"{param}_gaiadr3"]
            cnts_sources_params[param].append('Gaia DR3')
        # 2) use KIC as alternative
        elif f"{param}_q1q7dr25_stellar_supp" in q1q17dr25_stellar_supp_cols.values() \
                and not np.isnan(tce[f"{param}_q1q7dr25_stellar_supp"]):
            tce_tbl.loc[tce_i, param] = tce[f"{param}_q1q7dr25_stellar_supp"]
            cnts_sources_params[param].append('KIC')
        else:
            cnts_sources_params[param].append('')

cnts_sources_params_df = pd.DataFrame(cnts_sources_params)
print(cnts_sources_params_df.value_counts())

#%% save updated table

tce_tbl.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed_updtstellar.csv', index=False)

#%% Compare stellar parameters from DV TCE table with updated estimates

updt_tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed_updtstellar.csv')
# old table
tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed.csv')

plot_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/plots')
plot_dir.mkdir(exist_ok=True)

map_columns = {
    'tce_steff': 'effectiveTemp.value',
    'tce_sradius': 'radius.value',
    'tce_slogg': 'log10SurfaceGravity.value',
    'tce_smet': 'log10Metallicity.value',
}

plot_params = {
    'tce_steff': {'scale': 'linear', 'range': [2e3, 20e3]},
    'tce_sradius': {'scale': 'linear', 'range': [0, 2]},
    'tce_slogg': {'scale': 'linear', 'range': [-0.2, 5.4]},
    'tce_smet': {'scale': 'linear', 'range': [-2, 1]},
}

for param in map_columns:

    f, ax = plt.subplots(2, 1)
    ax[0].scatter(updt_tce_tbl[param], tce_tbl[map_columns[param]], s=8, edgecolor='k')
    ax[0].set_ylabel(f'{map_columns[param]} (DV)')
    ax[0].set_xlabel(f'{param} (Updated)')
    ax[0].set_yscale(plot_params[param]['scale'])
    ax[0].set_xscale(plot_params[param]['scale'])
    ax[0].set_xlim(plot_params[param]['range'])
    ax[0].set_ylim(plot_params[param]['range'])
    ax[0].set_title(f'Missing in DV: {((~updt_tce_tbl[param].isna()) & (tce_tbl[map_columns[param]].isna())).sum()}\n '
                 f'Missing in Updated: {((updt_tce_tbl[param].isna()) & (~tce_tbl[map_columns[param]].isna())).sum()}')
    ax[0].grid(axis='both')

    ax[1].scatter(np.arange(len(updt_tce_tbl)),
                  np.abs((updt_tce_tbl[param] - tce_tbl[map_columns[param]]) /
                         tce_tbl[map_columns[param]] + 1e-32) * 100,
                  s=8, edgecolor='k')
    ax[1].set_ylabel(f'Absolute Relative Difference (%)')
    ax[1].set_xlabel(f'TCE Sample Number')
    ax[1].set_yscale(plot_params[param]['scale'])
    ax[1].grid(axis='y')
    ax[1].set_yscale('log')
    f.tight_layout()
    plt.savefig(plot_dir / f'scatter_{param}_vs_{map_columns[param]}.png')
    plt.close()



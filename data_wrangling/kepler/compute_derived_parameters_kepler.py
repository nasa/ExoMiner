""" Recalculate derived planetary parameters for Kepler data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# 3rd party
import pandas as pd

# local
from data_wrangling.compute_derived_parameters import estimate_sma, estimate_plnt_eq_temp, estimate_plnt_radius, \
    estimate_new_sec_geo_albedo, estimate_new_plnt_eff_temp, estimate_eff_stellar_flux, compute_plnt_eff_temp_stat, \
    compute_sec_geo_albedo_stat, check_valid_sec_geo_albedo, check_valid_sma, check_valid_plnt_eq_temp

# %% recompute derived planetary parameters based on updated stellar parameters

res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/')

tce_tbl_fp = res_dir / \
             'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_5-26-2020.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

# add parameters from original table
tce_tbl_orig = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/q1_q17_dr25_tce_planet_params/'
                           'q1_q17_dr25_tce_2021.05.13_15.04.35.csv', header=67)
orig_params = [
    'tce_steff',
    'tce_steff_err',
    'tce_slogg',
    'tce_slogg_err',
    'tce_sradius',
    'tce_sradius_err',
    'tce_smet',
    'tce_smet_err',
    # 'tce_sdens',
    # 'tce_sdens_err',
    # 'mag',
    # 'mag_err',
    'tce_prad',
    'tce_prad_err',
    'tce_sma',
    'tce_sma_err',
    'tce_insol',
    'tce_insol_err',
    'tce_eqt',
    'tce_eqt_err',
    'tce_ptemp',
    'tce_ptemp_err',
    'tce_ptemp_stat',
    'tce_albedo',
    'tce_albedo_err',
    'tce_albedo_stat'
]
cols_to_keep = {col: f'{col}_orig' for col in orig_params}
tce_tbl = pd.concat([tce_tbl,
                     tce_tbl_orig.rename(columns=cols_to_keep, inplace=False)[list(cols_to_keep.values())]], axis=1)

# set missing stellar eff. temp., surface gravity and radius (NaNs) to 0 and -1 for values and uncertainties,
# respectively
for col in ['tce_steff', 'tce_slogg', 'tce_sradius']:
    tce_tbl.loc[tce_tbl[col].isna(), col] = 0
    tce_tbl.loc[tce_tbl[f'{col}_err'].isna(), f'{col}_err'] = -1

for tce_i, tce in tce_tbl.iterrows():

    if tce_i % 500 == 0:
        print(f'Updating derived planetary parameters for TCE {tce_i + 1}({len(tce_tbl)})')

    # skip TCEs whose stellar parameters were not updated
    if tce['updated_stellar_derived'] == 0:
        continue

    # compute planet radius
    tce[['tce_prad', 'tce_prad_err']] = estimate_plnt_radius(tce['tce_ror'],
                                                             tce['tce_sradius'],
                                                             tce['tce_ror_err'],
                                                             tce['tce_sradius_err'])

    # compute semi-major axis
    tce[['tce_sma', 'tce_sma_err']] = \
        estimate_sma(tce['tce_period'],
                     tce['tce_sradius'],
                     tce['tce_slogg'],
                     tce['tce_period_err'],
                     tce['tce_sradius_err'],
                     tce['tce_slogg_err'])

    # check if semi-major axis is larger than stellar radius plus planet radius
    tce[['tce_sma', 'tce_sma_err']] = \
        check_valid_sma(tce['tce_sma'], tce['tce_sma_err'], tce['tce_sradius'], tce['tce_prad'])

    # compute insolation flux
    tce[['tce_insol', 'tce_insol_err']] = \
        estimate_eff_stellar_flux(tce['tce_sradius'],
                                  tce['tce_sma'],
                                  tce['tce_steff'],
                                  tce['tce_slogg'],
                                  tce['tce_sradius_err'],
                                  tce['tce_steff_err'],
                                  tce['tce_slogg_err'],
                                  )

    # compute planet equilibrium temperature
    tce[['tce_eqt', 'tce_eqt_err']] = \
        estimate_plnt_eq_temp(tce['tce_steff'],
                              tce['tce_sradius'],
                              tce['tce_sma'],
                              tce['tce_steff_err'],
                              tce['tce_sradius_err'],
                              0.3,
                              tce['tce_slogg'],
                              tce['tce_slogg_err'],
                              tce['tce_sma_err'])

    # check if planet equilibrium temperature is smaller than stellar effective temperature
    tce[['tce_eqt', 'tce_eqt_err']] = \
        check_valid_plnt_eq_temp(tce['tce_eqt'], tce['tce_eqt_err'], tce['tce_steff'])

    tce[['tce_ptemp', 'tce_ptemp_err']] = \
        estimate_new_plnt_eff_temp(
            tce['tce_ptemp_orig'],
            tce['tce_steff_orig'],
            tce['tce_steff'],
            tce['tce_ptemp_err_orig'],
            tce['tce_steff_err_orig'],
            tce['tce_steff_err'],
        )

    # compute planet effective temperature comparison stat
    tce[['tce_ptemp_stat']] = \
        compute_plnt_eff_temp_stat(tce['tce_ptemp'], tce['tce_eqt'], tce['tce_ptemp_err'], tce['tce_eqt_err'])

    tce[['tce_albedo', 'tce_albedo_err']] = \
        estimate_new_sec_geo_albedo(
            tce['tce_albedo_orig'],
            tce['tce_prad_orig'],
            tce['tce_sma_orig'],
            tce['tce_prad'],
            tce['tce_sma'],
            tce['tce_albedo_err_orig'],
            tce['tce_prad_err_orig'],
            tce['tce_sma_err_orig'],
            tce['tce_prad_err'],
            tce['tce_sma_err'])

    # check if planet equilibrium temperature is smaller than stellr effective temperature
    tce[['tce_albedo', 'tce_albedo_err']] = \
        check_valid_sec_geo_albedo(tce['tce_albedo'],
                                   tce['tce_albedo_err'],
                                   tce['tce_sma'],
                                   tce['tce_sma_err'],
                                   tce['tce_prad'],
                                   tce['tce_sradius'],
                                   tce['tce_prad_err'],
                                   tce['wst_depth'])

    # compute albedo comparison stat
    tce[['tce_albedo_stat']] = \
        compute_sec_geo_albedo_stat(tce['tce_albedo'], tce['tce_albedo_err'])

    tce_tbl.loc[tce_i] = tce

# %% plot parameters in recomputed table vs original table

plot_cols = {
    'tce_sma': {'lim': [0, 2]},
    'tce_sma_err': {'lim': [0, 2]},
    'tce_insol': {'lim': [0, 2000]},
    'tce_insol_err': {'lim': [0, 2]},
    'tce_prad': {'lim': [0, 20]},
    'tce_prad_err': {'lim': [0, 20]},
    'tce_eqt': {'lim': [0, 10000]},
    'tce_eqt_err': {'lim': [0, 10000]},
    'tce_ptemp': {'lim': [0, 10000]},
    'tce_ptemp_err': {'lim': [0, 10000]},
    'tce_ptemp_stat': {'lim': [-20, 100]},
    'tce_albedo': {'lim': [0, 10000]},
    'tce_albedo_err': {'lim': [0, 10000]},
    'tce_albedo_stat': {'lim': [-100, 100]},
    'tce_steff': {'lim': [0, 20000]},
    'tce_steff_err': {'lim': [0, 4000]},
    'tce_slogg': {'lim': [0, 6]},
    'tce_slogg_err': {'lim': [0, 4]},
    'tce_smet': {'lim': [-2, 1]},
    'tce_smet_err': {'lim': [0, 1]},
    'tce_sradius': {'lim': [0, 10]},
    'tce_sradius_err': {'lim': [0, 10]},
    # 'tce_sdens': {'lim': [0, 100]},
    # 'tce_sdens_err': {'lim': [0, 100]},
    # 'tce_smass',
    # 'tce_smass_err'
    # 'mag': {'lim': [0, 20]},
    # 'mag_err': {'lim': [0, 20]},
}

for col in plot_cols:
    if 'err' in col:
        valid_tces = (tce_tbl[col] != -1) & (tce_tbl_orig[col] != -1)
        tce_tbl_aux = tce_tbl.loc[valid_tces]
        tce_tbl_old_aux = tce_tbl_orig.loc[valid_tces]
        num_missing_new, num_missing_old = (tce_tbl[col] == -1).sum(), (tce_tbl_orig[col] == -1).sum()
    else:
        valid_tces = (tce_tbl[col] != 0) & (tce_tbl_orig[col] != 0)
        tce_tbl_aux = tce_tbl.loc[valid_tces]
        tce_tbl_old_aux = tce_tbl_orig.loc[valid_tces]
        num_missing_new, num_missing_old = (tce_tbl[col] == 0).sum(), (tce_tbl_orig[col] == 0).sum()

    f, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax[0].set_title(f'Number of missing values (original): {num_missing_new} ({num_missing_old})')
    ax[0].scatter(tce_tbl_old_aux[col], tce_tbl_aux[col], c='b', s=8)
    # ax[0].set_xlabel(col)
    ax[0].set_ylabel(f'{col} new')
    ax[1].scatter(tce_tbl_old_aux[col], np.abs(tce_tbl_old_aux[f'{col}'] - tce_tbl_aux[f'{col}']) /
                  tce_tbl_old_aux[f'{col}'], c='b', s=8)
    ax[1].set_xlabel(col)
    ax[1].set_ylabel(f'Relative difference {col}')
    ax[0].set_xlim(plot_cols[col]['lim'])
    ax[0].set_ylim(plot_cols[col]['lim'])
    ax[1].set_xlim(plot_cols[col]['lim'])
    ax[0].grid(True)
    ax[1].grid(True)
    f.savefig(res_dir / f'scatter_{col}.png')
    plt.close(f)

# %% set missing values for parameters (0) and uncertainties (-1) to NaN

# setting missing stellar parameters back to NaN
tce_tbl_norecomp = pd.read_csv(tce_tbl_fp)
stellar_to_nan = [
    'tce_steff',
    'tce_steff_err',
    'tce_slogg',
    'tce_slogg_err',
    'tce_sradius',
    'tce_sradius_err',
]
tce_tbl[stellar_to_nan] = tce_tbl_norecomp[stellar_to_nan]

# set missing values for parameters (0) and uncertainties (-1) to NaN
parameters = [
    'tce_sma',
    'tce_sma_err',
    'tce_insol',
    'tce_insol_err',
    'tce_prad',
    'tce_prad_err',
    'tce_eqt',
    'tce_eqt_err',
    'tce_ptemp',
    'tce_ptemp_err',
    'tce_ptemp_stat',
    'tce_albedo',
    'tce_albedo_err',
    'tce_albedo_stat',
]
for param in parameters:
    if '_err' in param:
        tce_tbl.loc[tce_tbl[param] == -1, param] = np.nan
        tce_tbl.loc[tce_tbl[f'{param}_orig'] == -1, f'{param}_orig'] = np.nan
    else:
        tce_tbl.loc[tce_tbl[param] == 0, param] = np.nan
        tce_tbl.loc[tce_tbl[f'{param}_orig'] == 0, f'{param}_orig'] = np.nan

tce_tbl.to_csv(res_dir /
               'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_recomputedparams_7-26-2021.csv',
               index=False)

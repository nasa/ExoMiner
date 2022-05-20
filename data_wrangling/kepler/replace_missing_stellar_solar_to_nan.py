"""
Script used to revert stellar parameters filled with solar parameters by NaN. Missing values are posteriorly replaced by
the median of the training set.
"""

from pathlib import Path
import numpy as np
import pandas as pd

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

kpl_stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                               'q1_q17_dr25_stellar_plus_supp_gaiadr2.csv')

stellar_fields_out = ['mag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

# reset columns to nan
for col in stellar_fields_out:
    tce_tbl[col] = np.nan

count_nan_stellar_df = pd.DataFrame(data={col: [0] for col in stellar_fields_in})
for tce_i, tce in tce_tbl.iterrows():
    target_found = kpl_stellar_tbl.loc[kpl_stellar_tbl['kepid'] == tce['target_id']]

    tce_tbl.loc[tce_i, stellar_fields_out] = target_found[stellar_fields_in].values[0]

    count_nan_stellar_df += (target_found[stellar_fields_in].isna()).sum()

print(f'Number of missing stellar parameters:')
for col in stellar_fields_in:
    print(f'{col}: {count_nan_stellar_df[col].values[0]}')
tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_nanstellar.csv', index=False)

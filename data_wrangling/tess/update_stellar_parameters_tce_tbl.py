""" Update stellar parameters using the TIC catalog for the TESS TCE table. """

from pathlib import Path

import numpy as np
import pandas as pd
# 3rd party
from astroquery.mast import Catalogs

# %%

res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/9-14-2021')
res_dir.mkdir(exist_ok=True)
tce_tbl_fp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                  '09-14-2021_1754/tess_tces_s1-s40_09-14-2021_1754.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# %% Get TIC parameters from TIC catalog

tic_fields = {
    'Teff': 'tic_teff',
    'e_Teff': 'tic_teff_err',
    'mass': 'tic_mass',
    'e_mass': 'tic_mass_err',
    'MH': 'tic_met',
    'e_MH': 'tic_met_err',
    'rad': 'tic_rad',
    'e_rad': 'tic_rad_err',
    'rho': 'tic_rho',
    'e_rho': 'tic_rho_err',
    'logg': 'tic_logg',
    'e_logg': 'tic_logg_err',
    # 'ra': 'tic_ra',
    # 'dec': 'tic_dec',
    'KIC': 'kic_id',
    'GAIA': 'gaia_id',
    'Tmag': 'tic_tmag',
    'e_Tmag': 'tic_tmag_err',
}

# query from TIC all targets in the TCE table
catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tce_tbl['target_id'].unique().tolist()).to_pandas()
catalog_data = catalog_data.rename(columns={'ID': 'target_id', 'ra': 'tic_ra', 'dec': 'tic_dec'})
catalog_data = catalog_data.astype({'target_id': np.int64})

# add values from the TIC catalog to the TCE table
tce_tbl = tce_tbl.merge(catalog_data[['target_id'] + list(tic_fields.keys())], on=['target_id'])
tce_tbl = tce_tbl.rename(columns=tic_fields)

# tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_ticparams.csv', index=False)

# %% Update TIC parameters for the TCEs

tic_map = {
    'tce_steff': 'tic_teff',
    'tce_steff_err': 'tic_teff_err',
    'tce_slogg': 'tic_logg',
    'tce_slogg_err': 'tic_logg_err',
    'tce_sradius': 'tic_rad',
    'tce_sradius_err': 'tic_rad_err',
    'tce_smet': 'tic_met',
    'tce_smet_err': 'tic_met_err',
    'mag': 'tic_tmag',
    'mag_err': 'tic_tmag_err',
}

# # replace only missing stellar parameters (NaNs) by TIC values
# tce_params_cnt = []
# for tce_i, tce in tce_tbl.iterrows():
#     tce_params_cnt_aux = 0
#     for param in tic_map.keys():
#         if np.isnan(tce[param]):
#             tce[param] = tce[tic_map[param]]
#             tce_params_cnt_aux += 1
#     tce_params_cnt.append(tce_params_cnt_aux)
# print(f'Number of TCEs with stellar parameters changed from TIC: {len(np.where(np.array(tce_params_cnt) > 0)[0])}')

# update stellar parameters using the TIC catalog
tce_tbl[list(tic_map.keys())] = tce_tbl[list(tic_map.values())]

# add stellar mass which is not available in the TCE table
tce_tbl[['tce_smass', 'tce_smass_err']] = tce_tbl[['tic_mass', 'tic_mass_err']]

# # replace missing values by Solar parameters
# solar_params = {
#     'tce_steff': 5777,
#     'tce_slogg': 4.438,
#     'tce_sradius': 1.0,
#     'tce_smet': 0.0,
#     'tce_smass': 1.0,
#     'tce_sdens': 1.408
# }
# for solar_param, solar_param_val in solar_params.items():
#     tce_tbl.loc[tce_tbl[solar_param].isna()] = solar_param_val

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_stellarparams_updated.csv', index=False)

# check missing values (NaN)
stellar_params_cols = ['tce_steff', 'tce_steff_err', 'tce_slogg', 'tce_slogg_err', 'tce_sradius', 'tce_sradius_err',
                       'tce_smet', 'tce_smet_err', 'mag', 'mag_err', 'tce_smass', 'tce_smass_err']
for stellar_col in stellar_params_cols:
    print(f'{stellar_col}: {tce_tbl[stellar_col].isna().sum()}')

""" Update stellar parameters using the TIC catalog for the TESS TCE table. """

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
import logging

# %%

res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/preprocessing_tce_tables/s69-s80_s1s72_s1s69_09-12-2024_1019/tic_stellar')
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='add_stellar_params_tic_to_tce_tbl')
logger_handler = logging.FileHandler(filename=res_dir / 'add_stellar_params_tic_to_tce_tbl.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/preprocessing_tce_tables/s69-s80_s1s72_s1s69_09-12-2024_1019/tess_tces_dv_s69-s80_s1s72_s1s69_1019_ruwe.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Loading TCE table {tce_tbl_fp}...')

# %% Get TIC parameters from TIC catalog

logger.info('Getting stellar parameters from TIC catalog...')
tic_fields = {
    'Teff': 'tic_steff',
    'e_Teff': 'tic_steff_err',
    'mass': 'tic_smass',
    'e_mass': 'tic_smass_err',
    'MH': 'tic_smet',
    'e_MH': 'tic_smet_err',
    'rad': 'tic_sradius',
    'e_rad': 'tic_sradius_err',
    'rho': 'tic_sdens',
    'e_rho': 'tic_sdens_err',
    'logg': 'tic_slogg',
    'e_logg': 'tic_slogg_err',
    'ra': 'tic_ra',
    'dec': 'tic_dec',
    'KIC': 'kic_id',
    'GAIA': 'gaia_id',
    'Tmag': 'tic_tmag',
    'e_Tmag': 'tic_tmag_err',
    'ID': 'target_id'
}

# query from TIC all targets in the TCE table
tics = tce_tbl['target_id'].unique().astype('int').tolist()
logger.info(f'Querying results for {len(tics)} TICs...')
catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tics).to_pandas()
catalog_data = catalog_data.rename(columns=tic_fields)
catalog_data = catalog_data.astype({'target_id': np.int64})

# add values from the TIC catalog to the TCE table
logger.info('Adding TIC values to the TCE table...')
tce_tbl = tce_tbl.merge(catalog_data[list(tic_fields.values())], on=['target_id'], validate='many_to_one')

# %% Update TIC parameters for the TCEs

logger.info(f'Checking number of DV SPOC TCEs with solar parameters...')
solar_params = {
    'tce_steff': 5780,  # 5777
    'tce_slogg': 4.438,
    'tce_sradius': 1.0,
    'tce_smet': 0.0,
    'tce_smass': 1.0,
    'tce_sdens': 1.0,  # 1.408
}
logger.info(f'Solar parameters: {solar_params}')

logger.info(f'Checking number of TCEs with TICs with stellar parameter ~close~ to Solar parameters...')
for solar_param in solar_params:
    if solar_param not in ['tce_smass']:
        # print(f'{solar_param}: {(tce_tbl[solar_param] == solar_params[solar_param]).sum()}')
        logger.info(f'{solar_param}: {(np.abs(tce_tbl[solar_param] - solar_params[solar_param]) <= 0.00001).sum()}')

tic_map = {
    'tce_steff': 'tic_steff',
    'tce_steff_err': 'tic_steff_err',
    'tce_slogg': 'tic_slogg',
    'tce_slogg_err': 'tic_slogg_err',
    'tce_sradius': 'tic_sradius',
    'tce_sradius_err': 'tic_sradius_err',
    'tce_smet': 'tic_smet',
    'tce_smet_err': 'tic_smet_err',
    'tce_sdens': 'tic_sdens',
    'tce_sdens_err': 'tic_sdens_err',
    'mag': 'tic_tmag',
    'mag_err': 'tic_tmag_err',
    'ra': 'tic_ra',
    'dec': 'tic_dec',
}

# keep the stellar parameters in DV
for stellar_param in tic_map.keys():
    tce_tbl[f'{stellar_param}_dv'] = tce_tbl[stellar_param]

# check missing values (NaN)
stellar_params_cols = ['tce_steff', 'tce_steff_err', 'tce_slogg', 'tce_slogg_err', 'tce_sradius', 'tce_sradius_err',
                       'tce_smet', 'tce_smet_err', 'mag', 'mag_err', 'tce_sdens', 'tce_sdens_err', 'tce_smass',
                       'tce_smass_err', 'ra', 'dec']
logger.info('Missing values before updating stellar parameters with TIC')
for stellar_col in stellar_params_cols:
    if stellar_col not in ['tce_smass', 'tce_smass_err']:
        logger.info(f'{stellar_col}: {tce_tbl[stellar_col].isna().sum()}')

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
tce_tbl[['tce_smass', 'tce_smass_err']] = tce_tbl[['tic_smass', 'tic_smass_err']]

# fill missing values with values previously available in the DV TCE table
for tic_i, tic in enumerate(tce_tbl['target_id'].unique()):
    # get TCEs in same tic
    tces_in_tic = tce_tbl['target_id'] == tic
    for stellar_param in tic_map:
        if np.isnan(tce_tbl.loc[tces_in_tic, stellar_param].values[0]):
            tce_tbl.loc[tces_in_tic, stellar_param] = tce_tbl.loc[tces_in_tic, f'{stellar_param}_dv'].values[0]

# # replace missing values by Solar parameters
# for solar_param, solar_param_val in solar_params.items():
#     tce_tbl.loc[tce_tbl[solar_param].isna()] = solar_param_val

# check missing values (NaN)
logger.info('Missing values after updating stellar parameters with TIC')
for stellar_col in stellar_params_cols:
    logger.info(f'{stellar_col}: {tce_tbl[stellar_col].isna().sum()}')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ticstellar.csv', index=False)
logger.info('Saved TCE table with updated stellar parameters from TIC')

print('Finished.')

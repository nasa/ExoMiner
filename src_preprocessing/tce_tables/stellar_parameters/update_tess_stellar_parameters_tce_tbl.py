""" Update stellar parameters using the TIC-8 catalog for the TESS TCE table. """

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
import logging

MAP_TIC8_FIELDS = {  # map TIC-8 column names to desired names to add to TCE table
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

SOLAR_PARAMS = {  # set Solar parameters for potential setting missing stellar parameters to Sun's
    'tce_steff': 5780,  # 5777
    'tce_slogg': 4.438,
    'tce_sradius': 1.0,
    'tce_smet': 0.0,
    'tce_smass': 1.0,
    'tce_sdens': 1.0,  # 1.408
}

SOLAR_PARAM_CLOSE = 1e-5  # threshold used to check how close a stellar parameters is to the corresponding Solar parameter

MAP_DV_FIELDS = {  # map stellar parameter fields in the TCE table to those extracted and renamed from TIC-8 catalog.
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


def updated_stellar_parameters_with_tic8(tce_tbl, res_dir):
    """ Update stellar parameters in TCE table `tce_tbl` using TIC-8.  `tce_tbl` must contain column 'target_id' which
    stores the TIC ID of the targets associated with the TCEs, and columns with the stellar parameters with the names
    given by the keys in dictionary `MAP_DV_FIELDS`.

    :param tce_tbl: pandas DataFrame, TCE table
    :param res_dir: Path, save directory of queried results for TIC-8

    :return: tce_tbl, pandas DataFrame with updated stellar parameters using TIC-8
    """

    # set up logger
    logger = logging.getLogger(name='add_stellar_params_tic_to_tce_tbl')
    logger_handler = logging.FileHandler(filename=res_dir / 'add_stellar_params_tic_to_tce_tbl.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    # get TIC parameters from TIC catalog
    logger.info('Getting stellar parameters from TIC catalog...')

    # query from TIC all targets in the TCE table
    tics = tce_tbl['target_id'].unique().astype('int').tolist()
    logger.info(f'Querying results for {len(tics)} TICs using TIC-8 catalog...')
    catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tics).to_pandas()
    catalog_data.to_csv(res_dir / 'tic8_results.csv', index=False)

    # rename parameters names to expected names in the TCE table
    catalog_data = catalog_data.rename(columns=MAP_TIC8_FIELDS)
    catalog_data = catalog_data.astype({'target_id': np.int64})

    # add values from the TIC catalog to the TCE table
    logger.info('Adding TIC values to the TCE table...')
    tce_tbl = tce_tbl.merge(catalog_data[list(MAP_TIC8_FIELDS.values())], on=['target_id'], validate='many_to_one')

    # logger.info(f'Checking number of DV SPOC TCEs with solar parameters...')
    # logger.info(f'Using Solar parameters: {SOLAR_PARAMS}')
    # logger.info(f'Checking number of TCEs with TICs with stellar parameter ~close~ (diff < ) to Solar parameters...')
    # for solar_param in SOLAR_PARAMS:
    #     if solar_param not in ['tce_smass']:
    #         logger.info(f'{solar_param}: '
    #                     f'{(np.abs(tce_tbl[solar_param] - SOLAR_PARAMS[solar_param]) <= SOLAR_PARAM_CLOSE).sum()}')

    # keep the stellar parameters in SPOC DV
    for stellar_param in MAP_DV_FIELDS.keys():
        tce_tbl[f'{stellar_param}_dv'] = tce_tbl[stellar_param]

    # check missing values (NaN)
    logger.info('Missing values before updating stellar parameters with TIC')
    for stellar_col in list(MAP_DV_FIELDS.keys()) + ['tce_smass', 'tce_smass_err']:
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
    tce_tbl[list(MAP_DV_FIELDS.keys())] = tce_tbl[list(MAP_DV_FIELDS.values())]

    # add stellar mass which is not available in the TCE table
    tce_tbl[['tce_smass', 'tce_smass_err']] = tce_tbl[['tic_smass', 'tic_smass_err']]

    # fill missing values with values previously available in the DV TCE table
    for tic_i, tic in enumerate(tce_tbl['target_id'].unique()):
        # get TCEs in same tic
        tces_in_tic = tce_tbl['target_id'] == tic
        for stellar_param in MAP_DV_FIELDS:
            if np.isnan(tce_tbl.loc[tces_in_tic, stellar_param].values[0]):
                tce_tbl.loc[tces_in_tic, stellar_param] = tce_tbl.loc[tces_in_tic, f'{stellar_param}_dv'].values[0]

    # # replace missing values by Solar parameters
    # for solar_param, solar_param_val in solar_params.items():
    #     tce_tbl.loc[tce_tbl[solar_param].isna(), solar_param] = solar_param_val
    #     tce_tbl.loc[tce_tbl[solar_param].isna(), f'{solar_param}_err'] = -1

    # check missing values (NaN)
    logger.info('Missing values after updating stellar parameters with TIC')
    for stellar_col in list(MAP_DV_FIELDS.keys()) + ['tce_smass', 'tce_smass_err']:
        logger.info(f'{stellar_col}: {tce_tbl[stellar_col].isna().sum()}')

    return tce_tbl


if __name__ == '__main__':

    # set results directory for TIC-8 query
    res_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_tces_dv_s69-s88_s1s69_s2s72_s14s78_3-18-2025_0945/tic_stellar')
    # set filepath to TCE table
    tce_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_tces_dv_s69-s88_s1s69_s2s72_s14s78_3-18-2025_0945/tess_spoc_2min_tces_dv_s69-s88_s1s69_s2s72_s14s78_3-18-2025_0945_uid.csv')

    res_dir.mkdir(exist_ok=True)

    tce_tbl = pd.read_csv(tce_tbl_fp)
    print(f'Loaded TCE table {tce_tbl_fp}.')

    tce_tbl = updated_stellar_parameters_with_tic8(tce_tbl, res_dir)

    tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tic8stellar.csv', index=False)
    print('Saved TCE table with updated stellar parameters from TIC-8')

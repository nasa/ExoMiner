"""
Utility functions used for generating input TFRecords.
"""

# 3rd party
import os
import pickle
import pandas as pd
from tensorflow.compat.v1 import logging as tf_logging
import numpy as np
import json


def load_whitened_data(config):
    """ Loads the whitened data from Kepler into global variables. The tbl files were converted to pickle files.

    # FIXME: we have to eventually go back, talk with TESS team and think about how to use and implement the whitening
            # there are several TCEs whose time series as zero arrays

    :param config: Config object, contains the preprocessing parameters
    :return:
    """

    flux_files = [i for i in os.listdir(config['whitened_dir']) if i.startswith('DR25_readout_flux')
                  and not i.endswith('(copy)')]
    time_files = [i for i in os.listdir(config['whitened_dir']) if i.startswith('DR25_readout_time')
                  and not i.endswith('(copy)')]

    # FIXME: I am not a fan of using global variables...
    global flux_import
    global time_import
    flux_import, time_import = {}, {}

    for file in flux_files:  # [:int(len(flux_files)/4)]
        with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
            flux_import.update(pickle.load(fp))
    for file in time_files:  # [:int(len(time_files)/4)]
        with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
            time_import.update(pickle.load(fp))


def get_kepler_tce_table(config):
    """ Get TCE ephemeris table for Kepler.

    :param config: dict, preprocessing parameters
    :return:
        tce_table: pandas DataFrame, table with complete ephemeris table used when gapping the time series
    """

    # read the CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config['input_tce_csv_file'])
    # tces_lst = [
    #     '407347423-1-S25',  # NTPs with momentum dump
    #     '31313085-1-S4',
    #     '17660071-1-S7',
    #     '142086813-1-S3',
    #     '453101762-1-S3',
    #     '471015674-1-S3',
    #     '167305688-1-S4',
    #     '233684822-1-S14-50',  # EB
    #     '158561566-1-S14-14',  # planet
    # ]
    # tce_table = tce_table.loc[(tce_table['uid'].isin(tces_lst))]
    # tce_table = tce_table.loc[((tce_table['dataset'] == 'INJ1') & (tce_table['uid'] == '3833819-1'))]
    # tce_table[['tce_dikco_smsky', 'tce_dikco_msky_err', 'tce_dicco_msky', 'tce_dicco_msky_err', 'tce_fwm_stat']] = float(0)
    # tce_table['mag_cat'] = 1.0
    # tce_table = tce_table.sample(n=20)
    # tce_table = tce_table.loc[tce_table['uid'].isin(['6307062-1', ])]
    # tce_table = tce_table.loc[tce_table['target_id'].isin([11761169])]
    # tce_table['tce_depth'] = tce_table['transit_depth']
    tce_table["tce_duration"] /= 24  # Convert hours to days.

    tf_logging.info(f'Read TCE CSV file with {len(tce_table)} rows.')

    if config['using_mpi']:  # when using MPI processes to preprocess chunks of the TCE table in parallel

        boundaries = [int(i) for i in np.linspace(0, len(tce_table), config['n_processes'] + 1)]
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(config['n_processes'])][config['process_i']]

        shard_tce_table = tce_table[indices[0]:indices[1]]

        if not config['gapped']:
            tce_table = None

        return shard_tce_table, tce_table

    return tce_table, None


def get_tess_tce_table(config):
    """ Get TCE ephemeris table for TESS.

    :param config: dict, contains the preprocessing parameters
    :return:
        tce_table: pandas DataFrame, table with ephemeris table
    """

    # read TCE table
    tce_table = pd.read_csv(config['input_tce_csv_file'])
    # tce_table = tce_table.loc[tce_table['target_id'] == 127530399]
    # tces_lst = ['142276270-1-S14-19',
        # '164173105-1-S16', '166527623-1-S64', '166527623-1-S1-65',
       # '166527623-2-S38', '464646604-1-S27', '267572272-1-S54',
       # '164173105-1-S56', '383390264-1-S13', '267572272-1-S14-41',
       # '164173105-1-S14-60', '158657354-1-S14-55', '383390264-1-S1-65',
       # '383390264-1-S1-13', '267572272-1-S41', '142276270-4-S14-41',
       # '267572272-1-S53', '267572272-1-S40', '142276270-1-S21',
       # '142276270-1-S14-41', '267572272-1-S14-60', '142276270-4-S14-23',
       # '142276270-1-S14-50', '142276270-1-S14-23', '142276270-2-S21',
       # '142276270-1-S14-19', '142276270-2-S14-19', '142276270-2-S15',
       # '267572272-1-S14-55', '142276270-2-S14-23', '142276270-3-S14-50',
       # '142276270-2-S14-50', '267572272-1-S59', '142276270-1-S22'
       #           ]
    # [
    #     '142276270-2-S15',
        # '307490261-1-S11',
        # '20182165-1-S14-50',
        # '376989048-2-S42',
        # '388105497-2-S38',
        # '417728778-2-S19',
        # '89522181-1-S14-55',
    # ]
    # tce_table = tce_table.loc[tce_table['uid'].isin(tces_lst)]
    cols_change_data_type = {
        # 'sectors': str,
        'sector_run': str,
        # 'TOI': str,
        'label': str,
        # 'toi_sectors': str,
        'Comments': str,
        'TESS Disposition': str,
        'TFOPWG Disposition': str
    }
    tce_table = tce_table.astype(dtype={k: v for k, v in cols_change_data_type.items() if k in tce_table.columns})

    # convert transit duration from hour to day
    tce_table["tce_duration"] /= 24

    tf_logging.info(f'Read TCE CSV file with {len(tce_table)} rows.')

    if config['using_mpi']:  # when using MPI processes to preprocess chunks of the TCE table in parallel

        boundaries = [int(i) for i in np.linspace(0, len(tce_table), config['n_processes'] + 1)]
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(config['n_processes'])][config['process_i']]

        shard_tce_table = tce_table[indices[0]:indices[1]]

        if not config['gapped']:
            tce_table = None

        return shard_tce_table, tce_table

    return tce_table, None


def shuffle_tce(tce_table, seed=123):
    """ Helper function used to shuffle the tce_table if config.shuffle == True

    :param tce_table:   The non-shuffled TCE table
    :param seed:        Seed used for randomization
    :return:
        tce_table, with rows shuffled
    """

    np.random.seed(seed)

    tce_table = tce_table.iloc[np.random.permutation(len(tce_table))]

    tf_logging.info('Randomly shuffled TCEs.')

    return tce_table


def normalize_params_tce_table(config):
    """ Normalize parameters in the TCE table.

    :param config: dict with preprocessing configuration parameters
    :return:
        tce_tbl_norm_fp: str, filepath to TCE table with normalized parameters
    """

    # load the TCE table
    tce_tbl = pd.read_csv(config['input_tce_csv_file'])

    # filepath to normalized TCE table
    tce_tbl_norm_fp = f'{config["input_tce_csv_file"].replace(".csv", "")}_normalized.csv'

    # save TCE table with normalized parameters
    tce_tbl.to_csv(tce_tbl_norm_fp, index=False)

    return tce_tbl_norm_fp


def is_jsonable(x):
    """ Test if object is JSON serializable.

    :param x: object
    :return:
    """

    try:
        json.dumps(x)
        return True
    except:
        return False

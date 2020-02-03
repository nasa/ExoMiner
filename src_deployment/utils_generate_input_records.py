"""
Utility functions used for generating input TFRecords: manipulate TCE tables, load whitened data.
"""

# 3rd party
import os
import pickle
import pandas as pd
from scipy import io
import tensorflow as tf
import numpy as np


def load_whitened_data(config):
    """ Loads the whitened data from Kepler into global variables. The tbl files were converted to pickle files.

    # FIXME: we have to eventually go back, talk with TESS team and think about how to use and implement the whitening
            # there are several TCEs whose time series as zero arrays

    :param config: Config object, contains the preprocessing parameters
    :return:
    """

    flux_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_flux')
                  and not i.endswith('(copy)')]
    time_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_time')
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

    :param config: Config object, preprocessing parameter
    :return:
        tce_table: pandas DataFrame, table with complete ephemeris table used when gapping the time series
    """

    # eph_table = None
    # if config.gapped:  # get the ephemeris table for the gapped time series
    #     with open(config.eph_tbl_fp, 'rb') as fp:
    #         eph_table = pickle.load(fp)

    # name of the column in the TCE table with the label/disposition
    _LABEL_COLUMN = "av_training_set"
    # labels used to filter TCEs in the TCE table
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

    # extract these fields from the TPS mat file
    fields = ['kepid', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration', 'av_training_set']

    # Read the CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")
    tce_table["tce_duration"] /= 24  # Convert hours to days.
    tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]

    # print('len before filter by kepids: {}'.format(len(tce_table)))
    # # FILTER TCE TABLE FOR A SET OF KEPIDS
    # filt_kepids = os.listdir('/data5/tess_project/Data/DV_summaries_promising_before_pixeldata_fix')
    # filt_kepids = [int(el.split('-')[0][2:]) for el in filt_kepids if '.pdf' in el]
    # allowed_tces = tce_table['kepid'].apply(lambda l: l in filt_kepids)
    # tce_table = tce_table[allowed_tces]
    # print('len after filter by kepids: {}'.format(len(tce_table)))

    # use TPS ephemeris from the TPS TCE struct MATLAB file
    if config.use_tps_ephem:

        tf.logging.info("Using TPS ephemeris from {}.".format(config.use_tps_ephem))

        mat = io.loadmat(config.tps_ephem_tbl)['tpsTceStructV4_KSOP2536'][0][0]

        d = {name: [] for name in fields}

        # iterate over each row to get the ephemeris for each TCE
        for i in tce_table.iterrows():
            if i[1]['tce_plnt_num'] == 1:
                tpsStructIndex = np.where(mat['keplerId'] == i[1]['kepid'])[0]

                d['kepid'].append(i[1]['kepid'])
                d['tce_plnt_num'].append(1)

                d['tce_duration'].append(float(mat['maxMesPulseDurationHours'][0][tpsStructIndex]) / 24.0)
                d['tce_period'].append(float(mat['periodDays'][tpsStructIndex][0][0]))
                d['tce_time0bk'].append(float(mat['epochKjd'][tpsStructIndex][0][0]))

                d['av_training_set'].append(i[1]['av_training_set'])

                # d['av_training_set'].append('PC' if float(mat['isPlanetACandidate'][tpsStructIndex][0][0]) == 1.0 else
                #                             'AFP' if float(mat['isOnEclipsingBinaryList'][tpsStructIndex][0][0]) == 1.0
                #                             else 'NTP')

        # convert from dictionary to Pandas DataFrame
        tce_table = pd.DataFrame(data=d)

    # else:
    #     if config.whitened:  # get flux and cadence time series for the whitened data
    #         flux_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_flux')
    #                       and not i.endswith('(copy)')]
    #         time_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_time')
    #                       and not i.endswith('(copy)')]
    #
    #         # print('doing one quarter of all tces')
    #
    #         global flux_import
    #         global time_import
    #         flux_import, time_import = {}, {}
    #
    #         for file in flux_files:  # [:int(len(flux_files)/4)]
    #             with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
    #                 flux_import.update(pickle.load(fp))
    #         for file in time_files:  # [:int(len(time_files)/4)]
    #             with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
    #                 time_import.update(pickle.load(fp))

    if config.using_mpi:  # when using MPI processes to preprocess chunks of the TCE table in parallel

        boundaries = [int(i) for i in np.linspace(0, len(tce_table), config.n_processes + 1)]
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]

        shard_tce_table = tce_table[indices[0]:indices[1]]

        # Filter TCE table to allowed labels.
        # allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)

        if not config.gapped:
            tce_table = None

        return shard_tce_table, tce_table

    return tce_table


def get_tess_tce_table(config):
    """ Get TCE ephemeris table for TESS.

    :param config: Config object, contains the preprocessing parameters
    :return:
        tce_table: pandas DataFrame, table with ephemeris table
    """

    # name of the column in the TCE table with the label/disposition
    _LABEL_COLUMN = "disposition"
    # labels used to filter TCEs in the TCE table
    # KP, CP, PC, EB, IS, V, O, FP
    _ALLOWED_LABELS = {"KP", "PC", "EB", "IS", "V", "O"}

    # map from fields' names in the TCE table to fields' names in the TCE TPS table for TESS that we want to extract
    # if fields is None:
    # extract these fields from the mat file (map from TCE table column fields to TPS fields)
    # TODO: add other fields besides ephemeris?
    tpsFields = {'tic': 'catId', 'tce_plnt_num': 1, 'orbitalPeriodDays': 'detectedOrbitalPeriodDays',
                 'transitEpochBtjd': 'epochTjd', 'transitDurationHours': 'maxMesPulseDurationHours',
                 'disposition': 'isPlanetACandidate'}
    # tpsFields = {'tic': 'catId', 'tce_plnt_num': 1, 'orbitalPeriodDays': 'detectedOrbitalPeriodDays',
    #              'transitEpochBtjd': 'epochTjd', 'transitDurationHours': 'maxMesPulseDurationHours', 'tessMag': 'tessMag',
    #              'disposition': 'isPlanetACandidate', 'mes': 'maxMultipleEventStatistic}

    # Read the CSV file of TESS TOIs.
    # tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")
    tce_table = pd.read_csv(config.input_tce_csv_file, comment="#")
    tce_table["transitDurationHours"] /= 24  # convert hours to days.
    tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]

    # use TPS ephemeris from the TPS TCE struct MATLAB file
    if config.use_tps_ephem:

        tf.logging.info("Using TPS ephemeris from {}.".format(config.use_tps_ephem))

        tpsFiles = [os.path.join(config.tps_ephem_tbl, tpsFile) for tpsFile in os.listdir(config.tps_ephem_tbl)]

        tpsTCEtable = {name: [] for name in tpsFields}
        tpsTCEtable['sector'] = []

        for tpsFile in tpsFiles:

            # load TPS mat file
            tpsMat = io.loadmat(tpsFile)['tpsTceStruct'][0][0]

            sectorTPSFile = int(tpsFile.split('-')[-1][:-4])

            # iterate over each row to get the ephemeris for each TCE
            for row_i, tce in tce_table.iterrows():

                # only for TCEs detected by TPS module
                if tce['tce_plnt_num'] == 1:
                    # look for TCE in the TCE table
                    tpsStructIndex = np.where(tpsMat['catId'] == tce['tic'])[0]

                    tpsTCEtable['sector'].append(sectorTPSFile)

                    tpsTCEtable['tic'].append(tce['tic'])
                    # by default TPS assigns 1 to the first TCE if finds. DV iterates to find others in the same target
                    tpsTCEtable['tce_plnt_num'].append(1)

                    # ephemeris
                    # convert from hours to days
                    tpsTCEtable['orbitalPeriodDays'].append(
                        float(tpsMat[tpsFields['orbitalPeriodDays']][0][tpsStructIndex]) / 24.0)
                    tpsTCEtable['transitDurationHours'].append(
                        float(tpsMat[tpsFields['transitDurationHours']][tpsStructIndex][0][0]))
                    tpsTCEtable['transitEpochBtjd'].append(
                        float(tpsMat[tpsFields['transitEpochBtjd']][tpsStructIndex][0][0]))

                    # dispositions based on DV ephemeris
                    tpsTCEtable['disposition'].append(tce[tpsFields['disposition']])

                    # TODO: check dispositions when using TPS labels - use 'O' for the rest?
                    # TPS detection - PC, EB or else
                    # d['disposition'].append('PC' if float(mat['isPlanetACandidate'][tpsStructIndex][0][0]) == 1.0 else
                    #                         'EB' if float(mat['isOnEclipsingBinaryList'][tpsStructIndex][0][0]) == 1.0
                    #                         else 'O')

        # convert from dictionary to Pandas DataFrame
        tce_table = pd.DataFrame(data=tpsTCEtable)

        if config.using_mpi:  # when using MPI processes to preprocess chunks of the TCE table in parallel

            boundaries = [int(i) for i in np.linspace(0, len(tce_table), config.n_processes + 1)]
            indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]

            shard_tce_table = tce_table[indices[0]:indices[1]]

            # Filter TCE table to allowed labels.
            # allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)

            if not config.gapped:
                tce_table = None

            return shard_tce_table, tce_table

    return tce_table


def shuffle_tce(tce_table, seed=123):
    """ Helper function used to shuffle the tce_table if config.shuffle == True

    :param tce_table:   The un-shuffled TCE table
    :param seed:        Seed used for randomization
    :return:
        tce_table, with rows shuffled
    """

    np.random.seed(seed)

    tce_table = tce_table.iloc[np.random.permutation(len(tce_table))]
    #tce_table = tce_table.iloc[np.arange(4674,0,-1)]

    tf.logging.info("Randomly shuffled TCEs.")

    return tce_table

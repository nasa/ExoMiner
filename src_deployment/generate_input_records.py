""""
Main script usec to generate tfrecords --on the supercomputer cluster-- used as input to deep learning model
for classifying Threshold Crossing Events.
"""

# 3rd party
import sys
sys.path.append('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/')
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from mpi4py import MPI
import datetime
import socket
from scipy import io

# local
from src_preprocessing.preprocess import _process_tce


class Config:
    """"
    Config class to specify time series to be processed into tfrecords
    """

    satellite = 'kepler'  # choose from: ['kepler', 'tess']

    # Specify the fraction of data assigned to validation and test
    test_frac = 0.1
    val_frac = 0.1

    whitened = False  # Use pre-whitened time series as source?

    # minimum gap size( in time units) for a split
    gapWidth = 0.75

    # gapping - remove other TCEs belonging to the same target
    gapped = True
    gap_imputed = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    # binning parameters
    num_bins_glob = 2001  # Number of bins to discretize global view
    num_bins_loc = 201  # Number of bins to discretize local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    parser = argparse.ArgumentParser(description='Argument parser for batch pre processing')
    parser.add_argument('--gapped', action='store_true', default=gapped)
    parser.add_argument('--whitened', action='store_true', default=whitened)
    parser.add_argument('--num_bins_glob', type=int, default=num_bins_glob)
    parser.add_argument('--num_bins_loc', type=int, default=num_bins_loc)
    args = parser.parse_args()

    # if True, CCD module pixel coordinates are used. If False, pixel coordinates are transformed into RA and Dec
    # (world coordinates)
    px_coordinates = False

    use_tps_ephem = False  # Use TPS output ephemeris to phase phold data? (as opposed to DV ephemeris output)

    # if True, saves plots of several preprocessing steps
    plot_figures = True

    # use_ground_truth = False  # for TESS

    omit_missing = True  # skips Kepler IDs that are not in the fits files

    # save_stats = False

    # filepath to numpy file with stats used to preprocess the data
    stats_preproc_filepath = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/stats_trainingset.npy'

    # output directory
    output_dir = "tfrecords/tfrecord{}dr25_flux-centroid_selfnormalized".format(satellite)
    # working directory
    w_dir = '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/src_preprocessing'  # os.path.dirname(__file__)
    output_dir = os.path.join(w_dir, output_dir)

    # Ephemeris table (complete with the gapped TCEs) for the 34k TCEs Kepler
    # eph_tbl_fp = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/Kepler/DR25_readout_table'
    # eph_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/180k_tce.csv'

    # whitened data directory - 34k TCEs Kepler DR25 flux data only
    whitened_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/DR25_readouts'

    # Ephemeris table from the TPS module
    tps_ephem_tbl = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/' \
                    'tpsTceStructV4_KSOP2536.mat'

    if satellite.startswith('kepler'):
        # Ephemeris and labels for the 34k TCEs Kepler DR25 with TCERT labels
        input_tce_csv_file = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/Kepler/' \
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_' \
                             'updt_normstellarparamswitherrors.csv'
        # 34k TCEs Kepler DR25
        lc_data_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/PDC_timeseries/DR25/pdc-tce-time-series-fits'  # "/nobackupp2/lswilken/dr_25_all"
    elif satellite == 'tess':
        lc_str = 'lc_init_white' if whitened else 'lc_init'

        dict_savedir = ''
        if gap_with_confidence_level:
            NotImplementedError('TESS PC confidences not yet implemented')

    # input checks
    assert(satellite in ['kepler', 'tess'])

    if not gapped:
        assert not gap_imputed

    if gap_with_confidence_level:
        assert gapped

    # if use_ground_truth:
    #     assert satellite == 'tess'

    process_i = MPI.COMM_WORLD.rank
    n_processes = MPI.COMM_WORLD.size

    print('Process {} ({})'.format(process_i, n_processes))
    sys.stdout.flush()

    # Check that the number of tfrecord files for each dataset (train, val, test) falls into an integer number
    assert (n_processes % (1 / test_frac) == 0)
    assert (n_processes % (1 / val_frac) == 0)

    output_dir += '_whitened' if whitened else '_nonwhitened'
    if gapped:
        output_dir += '_gapped'
    if gap_imputed:
        output_dir += '_imputed'
    if gap_with_confidence_level:
        output_dir += '_conf%d' % int(gap_confidence_level * 100)
    # if use_ground_truth:
    #     output_dir += '_groundtruth'
    if use_tps_ephem:
        output_dir += '_tps'
    output_dir += '_%d-%d' % (num_bins_glob, num_bins_loc)

    if process_i == 0:
        print('Pre processing the following time series:\nGapped: %s\nWhitened: %s\nnum_bins_glob: %d\nnum_bins_loc: %d'
              % ('True' if gapped else 'False', 'True' if whitened else 'False', num_bins_glob, num_bins_loc))


def _process_file_shard(tce_table, file_name, eph_table):
    """Processes a single file shard.

    Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
    """

    config = Config()

    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)

    tf.logging.info("%s: Processing %d items in shard %s", config.process_i, shard_size, shard_name)

    confidence_dict = pickle.load(open(config.dict_savedir, 'rb')) if config.gap_with_confidence_level else {}

    start_time = int(datetime.datetime.now().strftime("%s"))

    with tf.python_io.TFRecordWriter(file_name) as writer:
        num_processed = 0
        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows

            # get flux and cadence data (if using whitened data)
            if config.satellite == 'kepler':
                if config.whitened and not (tce['kepid'] in flux_import and tce['kepid'] in time_import):
                    continue

                lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
                    if config.whitened else (None, None)
            else:  # TESS
                # aux_id = (tce['tessid'], 1, tce['sector'])  # tce_n = 1, import non-previous-tce-gapped light curves
                # lc, time = aux_dict[aux_id][0], aux_dict[aux_id][1]

                NotImplementedError('TESS preprocessing not yet implemented')

            example = _process_tce(tce, eph_table, lc, time, config, confidence_dict)
            if example is not None:
                writer.write(example.SerializeToString())

            num_processed += 1
            if config.n_processes < 50 or config.process_i == 0:
                if not num_processed % 10:
                    if config.process_i == 0:
                        cur_time = int(datetime.datetime.now().strftime("%s"))
                        eta = (cur_time - start_time) / num_processed * (shard_size - num_processed)
                        eta = str(datetime.timedelta(seconds=eta))
                        printstr = "%s: Processed %d/%d items in shard %s,   time remaining (HH:MM:SS): %s" \
                                   % (config.process_i, num_processed, shard_size, shard_name, eta)
                    else:
                        printstr = "%s: Processed %d/%d items in shard %s" % (config.process_i, num_processed,
                                                                              shard_size, shard_name)

                    tf.logging.info(printstr)

    if config.n_processes < 50:
        tf.logging.info("%s: Wrote %d items in shard %s", config.process_i, shard_size, shard_name)


def get_kepler_tce_table(config):
    """ Get TCE ephemeris tables.

    :param config: Config object, preprocessing parameters
    :return:
        shard_tce_table: pandas DataFrame, subset of ephemeris table with the TCEs to be processed in this shard
        tce_table: pandas DataFrame, table with complete ephemeris table used when gapping the time series
    """

    # eph_table = None
    # if config.gapped:  # get the ephemeris table for the gapped time series
    #     with open(config.eph_tbl_fp, 'rb') as fp:
    #         eph_table = pickle.load(fp)

    # Read CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")
    tce_table["tce_duration"] /= 24  # Convert hours to days.
    tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

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

    if config.use_tps_ephem:  # use TPS ephemeris from the TPS TCE struct MATLAB file for the 34k TCEs in Kepler DR25

        # extract these fields from the mat file
        fields = ['kepid', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration', 'av_training_set']

        mat = \
        io.loadmat('/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/'
                   'tpsTceStructV4_KSOP2536.mat')['tpsTceStructV4_KSOP2536'][0][0]

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

                d['av_training_set'].append('PC' if float(mat['isPlanetACandidate'][tpsStructIndex][0][0]) == 1.0
                                            else 'NTP')

        # convert from dictionary to Pandas DataFrame
        tce_table = pd.DataFrame(data=d)

    # else:
    #     tce_table["tce_duration"] /= 24  # Convert hours to days.
    #
    #     if config.process_i == 0:
    #         tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))
    #
    #     # load whitened flux time series and respective cadences
    #     if config.whitened:
    #         # get filepaths for the flux and cadences
    #         flux_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_flux')
    #                       and not i.endswith('(copy)')]
    #         time_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_time')
    #                       and not i.endswith('(copy)')]
    #
    #         # create global variables so that the data are available in other functions without needing to pass it as argument
    #         # FIXME: is it possible to have Kepler IDs in the cadence pickle file that are not in the flux pickle file, and vice-versa?
    #         #   no need to do 2 for loops, one to add Kepler IDs to remove, the second one to actually remove the data
    #         global flux_import
    #         global time_import
    #         flux_import, time_import = {}, {}
    #         # flux time series
    #         for file in flux_files:
    #             with open(os.path.join(whitened_dir, file), 'rb') as fp:
    #                 flux_import_i = pickle.load(fp)
    #
    #             # remove flux time series pertaining to Kepler IDs that are not in the TCE table
    #             remove_kepids = []
    #             for kepid in flux_import_i:
    #                 if kepid not in tce_table['kepid'].values:
    #                     remove_kepids.append(kepid)
    #
    #             for kepid in remove_kepids:
    #                 del flux_import_i[kepid]
    #
    #             # add the flux time series loaded from the pickle file to the dictionary
    #             flux_import.update(flux_import_i)
    #
    #         # cadences
    #         for file in time_files:
    #             with open(os.path.join(whitened_dir, file), 'rb') as fp:
    #                 time_import_i = pickle.load(fp)
    #
    #             remove_kepids = []
    #             for kepid in time_import_i:
    #                 if kepid not in tce_table['kepid'].values:
    #                     remove_kepids.append(kepid)
    #
    #             for kepid in remove_kepids:
    #                 del time_import_i[kepid]
    #
    #             time_import.update(time_import_i)

    boundaries = [int(i) for i in np.linspace(0, len(tce_table), config.n_processes + 1)]
    indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]

    shard_tce_table = tce_table[indices[0]:indices[1]]

    # Filter TCE table to allowed labels.
    # allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)

    if not config.gapped:
        tce_table = None

    return shard_tce_table, tce_table


def load_whitened_data(config):
    """ Loads the whitened data into global variables

    :param config: Config object, contains the preprocessing parameters
    :return:
    """

    flux_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_flux')
                  and not i.endswith('(copy)')]
    time_files = [i for i in os.listdir(config.whitened_dir) if i.startswith('DR25_readout_time')
                  and not i.endswith('(copy)')]

    global flux_import
    global time_import
    flux_import, time_import = {}, {}

    for file in flux_files:  # [:int(len(flux_files)/4)]
        with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
            flux_import.update(pickle.load(fp))
    for file in time_files:  # [:int(len(time_files)/4)]
        with open(os.path.join(config.whitened_dir, file), 'rb') as fp:
            time_import.update(pickle.load(fp))


# def _update_tess_lists(table_dict, ephemdict, match_dict, tid, sector_n, tce_n, config, time_vectors):
#     tce_dict = ephemdict['tce'][(tid, sector_n)][tce_n]
#
#     if config.use_ground_truth and match_dict is not None:  # only matched transits [pc, eb, beb]
#         ephem_dict = match_dict['truth']
#     else:
#         ephem_dict = tce_dict
#
#     update_id = False
#     for i in tce_dict[config.lc_str]:
#         if np.isfinite(i):
#             update_id = True
#             break
#
#     if update_id:  # check if light curve is not all NaN's
#         table_dict['tessid'] += [tid]
#         table_dict['sector'] += [sector_n]
#         table_dict['tce_n'] += [tce_n]
#         table_dict['tce_period'] += [ephem_dict['period']]
#         table_dict['tce_duration'] += [ephem_dict['duration']]
#         table_dict['tce_time0bk'] += [ephem_dict['epoch']]
#
#         aux_dict[(tid, tce_n, sector_n)] = [tce_dict[config.lc_str], time_vectors[sector_n - 1]]
#
#     return table_dict, update_id


# def get_tess_table(ephemdict, table_dict, label_map, config):
#     # count = 0
#     tce_tid_dict = {tce: None for tce in ephemdict['tce']}
#     time_vectors = ephemdict['info_dict'].pop('time')
#     for (tid, sector_n) in ephemdict['info_dict']:
#         tces_processed = []
#         n_tces = len(ephemdict['tce'][(tid, sector_n)])
#         for class_id in ephemdict['info_dict'][(tid, sector_n)]:
#             for match_n, match_dict in ephemdict['info_dict'][(tid, sector_n)][class_id].items():
#                 tce_n = match_dict['tce_id']['pred_tce_i']
#                 table_dict, update_id = _update_tess_lists(table_dict, ephemdict, match_dict, tid, sector_n,
#                                                            tce_n, config.lc_str, time_vectors)
#                 if update_id:
#                     table_dict['av_training_set'] += [label_map[class_id]]
#
#                 if tce_n not in tces_processed:
#                     tces_processed += [tce_n]
#         # count += 1
#         # if count % int(len(ephemdict['info_dict'])/100) == 0:
#         #     print('info dict percentage: %d, tce count: %d' % (int(count/len(ephemdict['info_dict'])*100), count))
#
#         # add all non-ephemeris matched tce's in 'info_dict'
#         if len(tces_processed) < n_tces:
#             for tce_n in set(range(1, n_tces + 1)) - set(tces_processed):
#                 table_dict, update_id = _update_tess_lists(table_dict, ephemdict, None, tid, sector_n,
#                                                            tce_n, config.lc_str, time_vectors)
#                 if update_id:
#                     table_dict['av_training_set'] += ['NTP']
#
#         tce_tid_dict.pop((tid, sector_n))
#
#     # all tess id's which are not present in 'info_dict' (info_dict: only tid's with matched tce's)
#     # count = 0
#     for (tid, sector_n) in tce_tid_dict:
#         for tce_n in ephemdict['tce'][(tid, sector_n)]:
#             table_dict, update_id = _update_tess_lists(table_dict, ephemdict, None, tid, sector_n,
#                                                        tce_n, config.lc_str, time_vectors)
#             if update_id:
#                 table_dict['av_training_set'] += ['NTP']
#
#         # count += 1
#         # if count % int(len(tce_tid_dict)/100) == 0:
#         #     print('post-info dict tce_t percentage: %d, tce count: %d' % (int(count/len(tce_tid_dict) * 100), count))
#
#     return table_dict


# def get_tess_tce_table(config):
#     ephemdict_str = ('/nobackupp2/lswilken/Astronet' if 'Documents' not in os.path.dirname(__file__)
#                      else '/home/lswilken/Documents/TESS_classifier') + '/TSOP-301_DV_ephem_dict'
#     with open(ephemdict_str, 'rb') as fp:
#         eph_table = pickle.load(fp)
#
#     table_dict = {'tessid': [], 'sector': [], 'tce_period': [], 'tce_duration': [],
#                   'tce_time0bk': [], 'av_training_set': [], 'tce_n': []}
#
#     label_map = {'planet': 'PC', 'eb': 'EB', 'backeb': 'BEB'}  # map TESS injected transit labels to AutoVetter labels
#
#     global aux_dict
#     aux_dict = {}
#
#     table_dict = get_tess_table(eph_table, table_dict, label_map, config)
#
#     tce_table = pd.DataFrame(table_dict)
#
#     num_tces = len(tce_table)
#
#     boundaries = [int(i) for i in np.linspace(0, num_tces, config.n_processes + 1)]
#     indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]
#
#     tce_table = tce_table[indices[0]:indices[1]]
#
#     # for kepid in flux_import:
#     #     if kepid not in tce_table:
#     #         del flux
#
#     return tce_table, eph_table


def get_tess_tce_table(config):
    """ Get TCE ephemeris tables.

    :param config:
    :return:
        tce_table: pandas DataFrame, table with ephemeris table
        eph_table: pandas DataFrame, table with complete ephemeris table used when gapping the time series
    """

    # name of the column in the TCE table with the label/disposition
    _LABEL_COLUMN = "disposition"
    # labels used to filter TCEs in the TCE table
    _ALLOWED_LABELS = {"KP", "PC", "EB", "IS", "V", "O"}

    # map from fields' names in the TCE table to fields' names in the TCE TPS table for TESS that we want to extract
    # if fields is None:
    fields = {'mes': 'maxMultipleEventStatistic', 'orbitalPeriodDays': 'detectedOrbitalPeriodInDays',
              'transitEpochBtjd': 'epochTjd', 'label': 'isPlanetACandidate'}

    # eph_table = None
    # if config.gapped:  # get the ephemeris table for the gapped time series
    #     with open(config.eph_tbl_fp, 'rb') as fp:
    #         eph_table = pickle.load(fp)

    # Read the CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")
    tce_table["transitDurationHours"] /= 24  # convert hours to days.
    tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]

    if config.use_tps_ephem:  # use TPS ephemeris from the TPS TCE struct MATLAB file

        tf.logging.info("Using TPS ephemeris from {}.".format(config.use_tps_ephem))

        tps_files = os.path.join(config.tps_ephem_tbl, os.listdir(config.tps_ephem_tbl))

        for tps_file in tps_files:

            mat = io.loadmat(tps_file)['tpsTceStruct'][0][0]

            d = {name: [] for name in fields}

            # iterate over each row to get the ephemeris for each TCE
            for i in tce_table.iterrows():

                # only for TCEs detected by TPS module
                if i[1]['tce_plnt_num'] == 1:

                    tpsStructIndex = np.where(mat['catId'] == i[1][''])[0]

                    d['tic'].append(i[1]['tic'])
                    d['tce_plnt_num'].append(1)

                    # convert from hours to days
                    d['transitDurationHours'].append(float(mat['maxMesPulseDurationHours'][0][tpsStructIndex]) / 24.0)
                    d['orbitalPeriodDays'].append(float(mat['detectedOrbitalPeriodInDays'][tpsStructIndex][0][0]))
                    d['transitEpochBtjd'].append(float(mat['epochTjd'][tpsStructIndex][0][0]))

                    # dispositions based on DV ephemeris
                    d['disposition'].append(i[1]['disposition'])

                    # TODO: check dispositions when using TPS labels - use 'O' for the rest?
                    # TPS detection - PC, EB or else
                    # d['disposition'].append('PC' if float(mat['isPlanetACandidate'][tpsStructIndex][0][0]) == 1.0 else
                    #                         'EB' if float(mat['isOnEclipsingBinaryList'][tpsStructIndex][0][0]) == 1.0
                    #                         else 'O')

        # convert from dictionary to Pandas DataFrame
        tce_table = pd.DataFrame(data=d)

    return tce_table


def main(_):

    # get the configuration parameters
    config = Config()

    rank_frac = config.process_i / config.n_processes

    # make the output directory if it doesn't already exist
    tf.gfile.MakeDirs(config.output_dir)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config.plot_figures:
        tf.gfile.MakeDirs(os.path.join(config.output_dir, 'plots'))

    # if config.save_stats:
    #     tf.gfile.MakeDirs(os.path.join(config.output_dir, 'stats'))

    if rank_frac < (1 - config.test_frac - config.val_frac):

        # get TCE and gapping ephemeris tables
        tce_table, eph_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                                else get_tess_tce_table(config))

        if config.whitened:  # get flux and cadence time series for the whitened data
            load_whitened_data(config)

        set_id_str = ('train' if rank_frac < (1 - config.test_frac - config.val_frac)
                      else 'val' if rank_frac >= 1 - config.test_frac
                      else 'test')
        # set_id_str = 'predict'

        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
        filename = set_id_str + "-{:05d}-of-{:05d}-node-{:s}".format(config.process_i, config.n_processes, node_id)
        file_name_i = os.path.join(config.output_dir, filename)

        _process_file_shard(tce_table, file_name_i, eph_table)

        tf.logging.info("Finished processing %d total file shards", len(tce_table))


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=main)

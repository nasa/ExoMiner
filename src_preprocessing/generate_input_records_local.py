"""
Generate preprocessed tfrecords locally.
"""

# 3rd party
import multiprocessing
import os
from scipy import io
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# local
from src_preprocessing.preprocess import _process_tce


class Config:

    satellite = 'kepler'  # choose from: ['kepler', 'tess']

    # output directory
    output_dir = 'tfrecords/tfrecord_dr25_manual_2d_few180k_' + satellite  # "tfrecords/tfrecord_dr25_manual_2d" + satellite
    gapped = False  # remove other TCEs from the light curve
    whitened = False  # whiten data
    gap_imputed = False  # add noise to gapped light curves

    use_tps_ephem = False  # use TPS ephemeris instead of DV

    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view

    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    use_ground_truth = False

    # working directory
    w_dir = os.path.dirname(__file__)
    output_dir = os.path.join(w_dir, output_dir)

    # path to updated TCE table, PDC time series fits files and confidence level dictionary
    if satellite.startswith('kepler'):
        input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/180k_tce.csv'  # w_dir + '/dr25_tce_upd_label.csv'
        lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'  # "/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits"
        dict_savedir = ''  # '/home/lswilken/Documents/Astronet_Simplified/pc_confidence_kepler_q1q17'
    elif satellite == 'tess':
        lc_str = 'lc_init_white' if whitened else 'lc_init'

        dict_savedir = ''
        if gap_with_confidence_level:
            NotImplementedError('TESS PC confidences not yet implemented')

    output_dir += 'whitened' if whitened else 'nonwhitened'

    if gapped:
        output_dir += '_gapped'
    if gap_imputed:
        output_dir += '_imputed'
    if gap_with_confidence_level:
        output_dir += '_conf%d' % int(gap_confidence_level * 100)
    if use_ground_truth:
        output_dir += '_groundtruth'
    if use_tps_ephem:
        output_dir += '_tps'

    # input checks
    assert(satellite in ['kepler', 'tess'])

    if not gapped:
        assert not gap_imputed

    if gap_with_confidence_level:
        assert gapped

    if use_ground_truth:
        assert satellite == 'tess'

    # 8 training shards, 1 validation and 1 test
    num_train_shards = 1  # 8
    num_worker_processes = 1  # number of workers

    omit_missing = True  # skips Kepler IDs that are not in the fits files


def _process_file_shard(tce_table, file_name, eph_table):
    """Processes a single file shard.

    Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
    """

    # (tce_table, file_name) = inputi
    # tce_table = tce_shard[0]
    # lcs, times = tce_shard[1], tce_shard[2]
    process_name = multiprocessing.current_process().name
    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)
    tf.logging.info("%s: Processing %d items in shard %s", process_name, shard_size, shard_name)
    config = Config()

    # load confidence dictionary
    confidence_dict = pickle.load(open(config.dict_savedir, 'rb')) if config.gap_with_confidence_level else {}

    with tf.python_io.TFRecordWriter(file_name) as writer:
        num_processed = 0
        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows
            if config.satellite == 'kepler':
                if config.whitened and not (tce['kepid'] in flux_import and tce['kepid'] in time_import):
                    continue

                lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
                    if config.whitened else (None, None)
            else:  # TESS
                aux_id = (tce['tessid'], 1, tce['sector'])  # tce_n = 1, import non-previous-tce-gapped light curves
                lc, time = aux_dict[aux_id][0], aux_dict[aux_id][1]

            # if tce['kepid'] in flux_import and tce['kepid'] in time_import:
            #     if config.satellite == 'kepler':
            #         lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
            #             if config.whitened else (None, None)
            #     else:  # TESS
            #         aux_id = (tce['tessid'], 1, tce['sector'])  # tce_n = 1, import non-previous-tce-gapped light curves
            #         lc, time = aux_dict[aux_id][0], aux_dict[aux_id][1]

            example = _process_tce(tce, eph_table, lc, time, config, confidence_dict)
            if example is not None:
                writer.write(example.SerializeToString())

            num_processed += 1
            if not num_processed % 10:
                tf.logging.info("%s: Processed %d/%d items in shard %s",
                                process_name, num_processed, shard_size, shard_name)

    tf.logging.info("%s: Wrote %d items in shard %s", process_name, shard_size, shard_name)


def get_kepler_tce_table(config):

    eph_savstr = '/data5/tess_project/Data/Ephemeris_tables/DR25_readout_table'  # config.w_dir + '/DR25_readouts'
    whitened_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/DR25_readouts'

    eph_table = None
    if config.gapped:  # get the ephemeris table for the gapped time series
        with open(eph_savstr, 'rb') as fp:
            eph_table = pickle.load(fp)

    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

    # Read the CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")
    tce_table["tce_duration"] /= 24  # Convert hours to days.
    tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]

    print('len before filter by kepids: {}'.format(len(tce_table)))
    # FILTER TCE TABLE FOR A SET OF KEPIDS
    filt_kepids = os.listdir('/data5/tess_project/Data/DV_summaries_promising_before_pixeldata_fix')
    filt_kepids = [int(el.split('-')[0][2:]) for el in filt_kepids if '.pdf' in el]
    allowed_tces = tce_table['kepid'].apply(lambda l: l in filt_kepids)
    tce_table = tce_table[allowed_tces]
    print('len after filter by kepids: {}'.format(len(tce_table)))

    if config.use_tps_ephem:  # load TPS ephemeris from the TPS TCE struct MATLAB file

        # extract these fields from the mat file
        fields = ['kepid', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration', 'av_training_set']

        mat = \
        io.loadmat('/data5/tess_project/Data/Ephemeris_tables/'
                   'tpsTceStructV4_KSOP2536.mat')['tpsTceStructV4_KSOP2536'][0][0]

        d = {name: [] for name in fields}

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

    else:
        if config.whitened:  # get flux and cadence time series for the whitened data
            flux_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_flux')
                          and not i.endswith('(copy)')]
            time_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_time')
                          and not i.endswith('(copy)')]

            # print('doing one quarter of all tces')

            global flux_import
            global time_import
            flux_import, time_import = {}, {}

            for file in flux_files:  # [:int(len(flux_files)/4)]
                with open(os.path.join(whitened_dir, file), 'rb') as fp:
                    flux_import.update(pickle.load(fp))
            for file in time_files:  # [:int(len(time_files)/4)]
                with open(os.path.join(whitened_dir, file), 'rb') as fp:
                    time_import.update(pickle.load(fp))

    return tce_table, eph_table


def _update_tess_lists(table_dict, ephemdict, match_dict, tid, sector_n, tce_n, config, time_vectors):
    tce_dict = ephemdict['tce'][(tid, sector_n)][tce_n]

    if config.use_ground_truth and match_dict is not None:  # only matched transits [pc, eb, beb]
        ephem_dict = match_dict['truth']
    else:
        ephem_dict = tce_dict

    update_id = False
    for i in tce_dict[config.lc_str]:
        if np.isfinite(i):
            update_id = True
            break

    if update_id:  # check if light curve is not all NaN's
        table_dict['tessid'] += [tid]
        table_dict['sector'] += [sector_n]
        table_dict['tce_n'] += [tce_n]
        table_dict['tce_period'] += [ephem_dict['period']]
        table_dict['tce_duration'] += [ephem_dict['duration']]
        table_dict['tce_time0bk'] += [ephem_dict['epoch']]

        aux_dict[(tid, tce_n, sector_n)] = [tce_dict[config.lc_str], time_vectors[sector_n - 1]]

    return table_dict, update_id


def get_tess_table(ephemdict, table_dict, label_map, config):
    # count = 0
    tce_tid_dict = {tce: None for tce in ephemdict['tce']}
    time_vectors = ephemdict['info_dict'].pop('time')
    for (tid, sector_n) in ephemdict['info_dict']:
        tces_processed = []
        n_tces = len(ephemdict['tce'][(tid, sector_n)])
        for class_id in ephemdict['info_dict'][(tid, sector_n)]:
            for match_n, match_dict in ephemdict['info_dict'][(tid, sector_n)][class_id].items():
                tce_n = match_dict['tce_id']['pred_tce_i']
                table_dict, update_id = _update_tess_lists(table_dict, ephemdict, match_dict, tid, sector_n,
                                                           tce_n, config.lc_str, time_vectors)
                if update_id:
                    table_dict['av_training_set'] += [label_map[class_id]]

                if tce_n not in tces_processed:
                    tces_processed += [tce_n]
        # count += 1
        # if count % int(len(ephemdict['info_dict'])/100) == 0:
        #     print('info dict percentage: %d, tce count: %d' % (int(count/len(ephemdict['info_dict'])*100), count))

        # add all non-ephemeris matched tce's in 'info_dict'
        if len(tces_processed) < n_tces:
            for tce_n in set(range(1, n_tces + 1)) - set(tces_processed):
                table_dict, update_id = _update_tess_lists(table_dict, ephemdict, None, tid, sector_n,
                                                           tce_n, config.lc_str, time_vectors)
                if update_id:
                    table_dict['av_training_set'] += ['NTP']

        tce_tid_dict.pop((tid, sector_n))

    # all tess id's which are not present in 'info_dict' (info_dict: only tid's with matched tce's)
    # count = 0
    for (tid, sector_n) in tce_tid_dict:
        for tce_n in ephemdict['tce'][(tid, sector_n)]:
            table_dict, update_id = _update_tess_lists(table_dict, ephemdict, None, tid, sector_n,
                                                       tce_n, config.lc_str, time_vectors)
            if update_id:
                table_dict['av_training_set'] += ['NTP']

        # count += 1
        # if count % int(len(tce_tid_dict)/100) == 0:
        #     print('post-info dict tce_t percentage: %d, tce count: %d' % (int(count/len(tce_tid_dict) * 100), count))

    return table_dict


def get_tess_tce_table(config):
    ephemdict_str = ('/nobackupp2/lswilken/Astronet' if 'Documents' not in os.path.dirname(__file__)
                     else '/home/lswilken/Documents/TESS_classifier') + '/TSOP-301_DV_ephem_dict'
    with open(ephemdict_str, 'rb') as fp:
        eph_table = pickle.load(fp)

    table_dict = {'tessid': [], 'sector': [], 'tce_period': [], 'tce_duration': [],
                  'tce_time0bk': [], 'av_training_set': [], 'tce_n': []}

    label_map = {'planet': 'PC', 'eb': 'EB', 'backeb': 'BEB'}  # map TESS injected transit labels to AutoVetter labels

    global aux_dict
    aux_dict = {}

    table_dict = get_tess_table(eph_table, table_dict, label_map, config)

    return pd.DataFrame(table_dict), eph_table


def main(_):

    # get the configuration parameters
    config = Config()

    # Make the output directory if it doesn't already exist.
    tf.gfile.MakeDirs(config.output_dir)

    tce_table, eph_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                            else get_tess_tce_table(config))

    num_tces = len(tce_table)

    # # Randomly shuffle the TCE table.
    # np.random.seed(123)
    # tce_table = tce_table.iloc[np.random.permutation(num_tces)]
    # tf.logging.info("Randomly shuffled TCEs.")

    # Partition the TCE table as follows:
    pred_tces = tce_table
    # train_cutoff = int(0.80 * num_tces)
    # val_cutoff = int(0.90 * num_tces)
    # train_tces = tce_table[0:train_cutoff]
    # val_tces = tce_table[train_cutoff:val_cutoff]
    # test_tces = tce_table[val_cutoff:]
    # tf.logging.info("Partitioned %d TCEs into training (%d), validation (%d) and test (%d)",
    #               num_tces, len(train_tces), len(val_tces), len(test_tces))

    # Further split training TCEs into file shards.
    file_shards = []  # List of (tce_table_shard, file_name).
    # boundaries = np.linspace(0, len(train_tces), config.num_train_shards + 1).astype(np.int)
    boundaries = np.linspace(0, len(pred_tces), config.num_train_shards + 1).astype(np.int)
    for i in range(config.num_train_shards):
        start = boundaries[i]
        end = boundaries[i + 1]
        # filedicname = os.path.join(config.output_dir, "train-{:05d}-of-{:05d}".format(i, config.num_train_shards))
        # file_shards.append((train_tces[start:end], filename, eph_table))
        filename = os.path.join(config.output_dir, "predict-{:05d}-of-{:05d}".format(i, config.num_train_shards))
        file_shards.append((pred_tces[start:end], filename, eph_table))

    # # Validation and test sets each have a single shard.
    # file_shards.append((val_tces, os.path.join(config.output_dir, "val-00000-of-00001"), eph_table))
    # file_shards.append((test_tces, os.path.join(config.output_dir, "test-00000-of-00001"), eph_table))

    num_file_shards = len(file_shards)

    # Launch subprocesses for the file shards.
    num_processes = min(num_file_shards, config.num_worker_processes)
    tf.logging.info("Launching %d subprocesses for %d total file shards", num_processes, num_file_shards)

    pool = multiprocessing.Pool(processes=num_processes)
    async_results = [pool.apply_async(_process_file_shard, file_shard) for file_shard in file_shards]
    pool.close()

    # Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
    for async_result in async_results:
        async_result.get()

    tf.logging.info("Finished processing %d total file shards", num_file_shards)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

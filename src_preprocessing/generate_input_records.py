""""
Main script to generate tfrecords --on Pleiades Supercomputer-- used as input to deep learning model
for classifying Threshold Crossing Events.
"""

# 3rd party
import sys
# sys.path.append('/home6/msaragoc/work_dir/HPO_Kepler_TESS/')
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

    gapped = False  # Gap transits in lightcurve correponding to other TCE's?
    # Gap transits in lightcurve correponding to other TCE's only if highly confident other TCE is planet?
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    whitened = False  # Use pre-whitened time series as source?
    gap_imputed = False  # impute gaps (if gapped) with normally distributed random noise?

    use_tps_ephem = False  # Use TPS output ephemeris to phase phold data? (as opposed to DV ephemeris output)

    num_bins_glob = 2001  # Number of bins to discretize global view
    num_bins_loc = 201  # Number of bins to discretize local view

    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    use_ground_truth = False  # for TESS

    omit_missing = True  # skips Kepler IDs that are not in the fits files

    output_dir = "tfrecords/tfrecord_dr25_manual_2d" + satellite

    parser = argparse.ArgumentParser(description='Argument parser for batch pre processing')
    parser.add_argument('--gapped', action='store_true', default=gapped)
    parser.add_argument('--whitened', action='store_true', default=whitened)
    parser.add_argument('--num_bins_glob', type=int, default=num_bins_glob)
    parser.add_argument('--num_bins_loc', type=int, default=num_bins_loc)
    args = parser.parse_args()

    # Overwrite inputs if arguments passed from batch pre processor (run from batch_preprocessor.py)
    gapped = args.gapped
    whitened = args.whitened
    num_bins_glob = args.num_bins_glob
    num_bins_loc = args.num_bins_loc

    w_dir = os.path.dirname(__file__)
    output_dir = os.path.join(w_dir, output_dir)

    if satellite.startswith('kepler'):
        input_tce_csv_file = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/dr25_tce_upd_label.csv'  # os.path.join(w_dir, "dr25_tce_upd_label.csv")
        lc_data_dir = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/PDC_timeseries/DR25/pdc-tce-time-series-fits'  # "/nobackupp2/lswilken/dr_25_all"
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

    if use_ground_truth:
        assert satellite == 'tess'

    process_i = MPI.COMM_WORLD.rank
    n_processes = MPI.COMM_WORLD.size

    # Check that the number of tfrecord files for each dataset (train, val, test) falls into an integer number
    assert (n_processes % (1 / test_frac) == 0)
    assert (n_processes % (1 / val_frac) == 0)

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
        for index, tce in tce_table.iterrows():
            if config.satellite == 'kepler':
                if config.whitened and not (tce['kepid'] in flux_import and tce['kepid'] in time_import):
                    continue

                lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
                    if config.whitened else (None, None)
            else:  # TESS
                aux_id = (tce['tessid'], 1, tce['sector'])  # tce_n = 1, import non-previous-tce-gapped light curves
                lc, time = aux_dict[aux_id][0], aux_dict[aux_id][1]

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
                        printstr = "%s: Processed %d/%d items in shard %s" % (config.process_i, num_processed, shard_size, shard_name)

                    tf.logging.info(printstr)

    if config.n_processes < 50:
        tf.logging.info("%s: Wrote %d items in shard %s", config.process_i, shard_size, shard_name)


def get_kepler_tce_table(config):

    eph_savstr = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/DR25_readout_table'  # config.w_dir + '/DR25_readouts'
    whitened_dir = '/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/DR25_readouts'

    eph_table = None
    if config.gapped:
        with open(eph_savstr, 'rb') as fp:
            eph_table = pickle.load(fp)

    # Read CSV file of Kepler KOIs.
    tce_table = pd.read_csv(config.input_tce_csv_file, index_col="rowid", comment="#")

    if config.use_tps_ephem:
        fields = ['kepid', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration', 'av_training_set']

        mat = \
        io.loadmat('/home6/msaragoc/work_dir/data/EXOPLNT_Kepler-TESS/Ephemeris_tables/'
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

                d['av_training_set'].append('PC' if float(mat['isPlanetACandidate'][tpsStructIndex][0][0]) == 1.0
                                            else 'NTP')

        tce_table = pd.DataFrame(data=d)

    else:
        tce_table["tce_duration"] /= 24  # Convert hours to days.

        if config.process_i == 0:
            tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

        # load whitened flux time series and respective cadences
        if config.whitened:
            # get filepaths for the flux and cadences
            flux_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_flux')
                          and not i.endswith('(copy)')]
            time_files = [i for i in os.listdir(whitened_dir) if i.startswith('DR25_readout_time')
                          and not i.endswith('(copy)')]

            # create global variables so that the data are available in other functions without needing to pass it as argument
            # FIXME: is it possible to have Kepler IDs in the cadence pickle file that are not in the flux pickle file, and vice-versa?
            #   no need to do 2 for loops, one to add Kepler IDs to remove, the second one to actually remove the data
            global flux_import
            global time_import
            flux_import, time_import = {}, {}
            # flux time series
            for file in flux_files:
                with open(os.path.join(whitened_dir, file), 'rb') as fp:
                    flux_import_i = pickle.load(fp)

                # remove flux time series pertaining to Kepler IDs that are not in the TCE table
                remove_kepids = []
                for kepid in flux_import_i:
                    if kepid not in tce_table['kepid'].values:
                        remove_kepids.append(kepid)

                for kepid in remove_kepids:
                    del flux_import_i[kepid]

                # add the flux time series loaded from the pickle file to the dictionary
                flux_import.update(flux_import_i)

            # cadences
            for file in time_files:
                with open(os.path.join(whitened_dir, file), 'rb') as fp:
                    time_import_i = pickle.load(fp)

                remove_kepids = []
                for kepid in time_import_i:
                    if kepid not in tce_table['kepid'].values:
                        remove_kepids.append(kepid)

                for kepid in remove_kepids:
                    del time_import_i[kepid]

                time_import.update(time_import_i)

    # number of TCEs
    num_tces = len(tce_table)


    boundaries = [int(i) for i in np.linspace(0, num_tces, config.n_processes + 1)]
    indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]

    tce_table = tce_table[indices[0]:indices[1]]

    # Filter TCE table to allowed labels.
    # allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)

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

    tce_table = pd.DataFrame(table_dict)

    num_tces = len(tce_table)

    boundaries = [int(i) for i in np.linspace(0, num_tces, config.n_processes + 1)]
    indices = [(boundaries[i], boundaries[i + 1]) for i in range(config.n_processes)][config.process_i]

    tce_table = tce_table[indices[0]:indices[1]]

    # for kepid in flux_import:
    #     if kepid not in tce_table:
    #         del flux

    return tce_table, eph_table


def main(_):
    config = Config()
    # Make the output directory if it doesn't already exist.
    tf.gfile.MakeDirs(config.output_dir)

    tce_table, eph_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                            else get_tess_tce_table(config))

    rank_frac = config.process_i / config.n_processes
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

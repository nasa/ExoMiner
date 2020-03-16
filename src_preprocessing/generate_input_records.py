""""
Main script usec to generate tfrecords --on the supercomputer cluster-- used as input to deep learning model
for classifying Threshold Crossing Events.
"""

# 3rd party
import sys
# sys.path.append('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/')
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import pickle
from mpi4py import MPI
import datetime
import socket
import numpy as np

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_tess_tce_table, get_kepler_tce_table, \
    load_whitened_data, shuffle_tce


class Config:
    """"
    Config class to specify time series to be processed into tfrecords
    """

    satellite = 'kepler'  # choose from: ['kepler', 'tess']
    multisector = False  # True for TESS multi-sector runs

    training = True  # choose from: 'training' or 'predict'
    # partition the data set
    datasets_frac = {'training': 0.8, 'validation': 0.1, 'test': 0.1}

    assert np.sum(list(datasets_frac.values())) <= 1

    whitened = False  # use whitened data (currently only available for Kepler DR25 34k TCEs dataset)

    # minimum gap size( in time units - day) for a split
    gapWidth = 0.75

    # gapping - remove other TCEs belonging to the same target
    gapped = True
    gap_imputed = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    # if True, CCD module pixel coordinates are used. If False, pixel coordinates are transformed into RA and Dec
    # (world coordinates)
    px_coordinates = False

    # if True, saves plots of several preprocessing steps
    plot_figures = True

    omit_missing = True  # skips target IDs that are not in the FITS files

    # list with the names of the scalar parameters from the TCE table (e.g. stellar parameters) that are also added to
    # the TFRecords along with the time-series features (views). Set list to empty to not add any scalar parameter.
    # These parameters are added to the example in the TFRecord as a list of float values.
    scalar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

    use_tps_ephem = False  # use TPS ephemeris instead of DV

    # binning parameters
    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    # save_stats = False

    # filepath to numpy file with stats used to preprocess the data
    stats_preproc_filepath = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/stats_trainingset.npy'

    # output directory
    output_dir = "tfrecords/tfrecord{}dr25_flux-centroid_selfnormalized-oddeven".format(satellite)
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
        # input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/' \
        #                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_' \
        #                      'updt_normstellarparamswitherrors.csv'
        input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/' \
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_' \
                             'koidatalink_processed'
        # 34k TCEs Kepler DR25
        lc_data_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/PDC_timeseries/DR25/pdc-tce-time-series-fits'
    elif satellite == 'tess':
        input_tce_csv_file = ''

        lc_data_dir = ''

    # shuffle TCE table
    shuffle = False

    output_dir += '_whitened' if whitened else '_nonwhitened'
    if gapped:
        output_dir += '_gapped'
    if gap_imputed:
        output_dir += '_imputed'
    if gap_with_confidence_level:
        output_dir += '_conf%d' % int(gap_confidence_level * 100)
    if use_tps_ephem:
        output_dir += '_tps'
    output_dir += '_%d-%d' % (num_bins_glob, num_bins_loc)

    # # input checks
    # assert(satellite in ['kepler', 'tess'])
    #
    # if not gapped:
    #     assert not gap_imputed
    #
    # if gap_with_confidence_level:
    #     assert gapped

    # multiprocessing parameters
    using_mpi = True

    process_i = MPI.COMM_WORLD.rank
    n_processes = MPI.COMM_WORLD.size

    print('Process {} ({})'.format(process_i, n_processes))
    sys.stdout.flush()

    # # TODO: what about for TESS?
    # # Check that the number of tfrecord files for each dataset (train, val, test) falls into an integer number
    # assert (n_processes % (1 / test_frac) == 0)
    # assert (n_processes % (1 / val_frac) == 0)

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

            lc, time = None, None
            # get flux and cadence data (if using whitened data)
            if config.whitened:
            # # get flux and cadence data (if using whitened data)
            # # check if TCE is in the whitened dataset
            # if config.satellite == 'kepler' and config.whitened and not (tce['kepid'] in flux_import and tce['kepid']
            #                                                              in time_import):
            #     lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
            #         if config.whitened else (None, None)
                raise NotImplementedError('Whitening still not implemented-ish.')
            # else:  # TESS
            #     # aux_id = (tce['tessid'], 1, tce['sector'])  # tce_n = 1, import non-previous-tce-gapped light curves
            #   continue

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

    # if rank_frac < (1 - config.test_frac - config.val_frac):

    # get TCE and gapping ephemeris tables
    tce_table, eph_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                            else get_tess_tce_table(config))

    # shuffle tce table
    if config.shuffle:
        tce_table = shuffle_tce(tce_table, seed=123)
        print('Shuffled TCE Table')

    if config.whitened:  # get flux and cadence time series for the whitened data
        load_whitened_data(config)

    if config.training:
        set_id_str = ('train' if rank_frac < (1 - config.datasets_frac['test'] - config.datasets_frac['validation'])
                      else 'val' if rank_frac >= 1 - config.datasets_frac['test']
                      else 'test')
    else:
        set_id_str = 'predict'

    node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
    filename = set_id_str + "-{:05d}-of-{:05d}-node-{:s}".format(config.process_i, config.n_processes, node_id)
    file_name_i = os.path.join(config.output_dir, filename)

    _process_file_shard(tce_table, file_name_i, eph_table)

    tf.logging.info("Finished processing %d total file shards", len(tce_table))


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=main)

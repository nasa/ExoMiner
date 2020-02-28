"""
Generate preprocessed tfrecords locally.
"""

# 3rd party
import sys
# sys.path.append('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/')
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
import numpy as np
import tensorflow as tf
import pickle

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_kepler_tce_table, get_tess_tce_table, \
    load_whitened_data, shuffle_tce


class Config:
    """ Class that creates configuration objects that hold parameters required for the preprocessing."""

    satellite = 'tess'  # choose from: ['kepler', 'tess']
    multisector = True  # True for TESS multi-sector runs
    sectors = np.arange(1, 19)
    tce_identifier = 'oi'

    training = True  # choose from: 'training' or 'predict'
    # partition the data set; only used with training set to True
    datasets_frac = {'training': 0.8, 'validation': 0.1, 'test': 0.1}

    assert np.sum(list(datasets_frac.values())) <= 1

    whitened = False  # use whitened data (currently only available for Kepler DR25 34k TCEs dataset)

    # minimum gap size(in time units - day) for a split
    gapWidth = 0.75

    # gapping - remove other TCEs belonging to the same target; if conducting a multi-sector run, check only for TCEs
    # in the same sector
    gapped = True
    gap_imputed = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    use_tps_ephem = False  # use TPS ephemeris instead of DV

    # binning parameters
    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    # if True, CCD module pixel coordinates are used. If False, local CCD pixel coordinates are transformed into RA and
    # Dec (world coordinates)
    px_coordinates = False

    # if True, saves plots of several preprocessing steps
    plot_figures = True

    omit_missing = True  # skips target IDs that are not in the FITS files

    # list with the names of the scalar parameters from the TCE table (e.g. stellar parameters) that are also added to
    # the TFRecords along with the time-series features (views). Set list to empty to not add any scalar parameter.
    # These parameters are added to the example in the TFRecord as a list of float values.
    scalar_params = []

    # save_stats = True

    # filepath to numpy file with stats used to preprocess the data
    stats_preproc_filepath = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/' \
                             'tfrecords/tfrecord_keplerdr25_centroidnonnormalized_radec_nonwhitened_gapped_2001-201/' \
                             'stats_trainingset.npy'

    # output directory
    # output_dir = "tfrecords/tfrecord{}dr25_centroidnormalized_test".format(satellite)
    output_dir = "tfrecords/tfrecord{}_testmulisector".format(satellite)
    # working directory
    w_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing'
    output_dir = os.path.join(w_dir, output_dir)

    # Ephemeris table (complete with the gapped TCEs)
    # eph_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/DR25_readout_table'
    # eph_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/180k_tce.csv'

    # whitened Kepler data directory
    whitened_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/DR25_readouts'

    # Ephemeris table from the TPS module
    if satellite == 'kepler':
        tps_ephem_tbl = '/data5/tess_project/Data/Ephemeris_tables/tpsTceStructV4_KSOP2536.mat'  # Kepler 180k
    else:  # TESS TPS ephemeris directory with TPS TCE struct MATLAB files for 1 year of TESS data (sectors 1-13)
        tps_ephem_tbl = '/data5/tess_project/Data/Ephemeris_tables/TESS/TPS_TCE_struct_TESS_1yr'  # TESS (sectors 1-13)

    # path to updated TCE table, PDC time series fits files and confidence level dictionary
    if satellite.startswith('kepler'):

        # TCE table filepath
        input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/Kepler/' \
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_' \
                             'updt_normstellarparamswitherrors.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/180k_tce.csv'

        # FITS files directory
        lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'
        # lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'

        dict_savedir = ''  # '/home/lswilken/Documents/Astronet_Simplified/pc_confidence_kepler_q1q17'

    elif satellite == 'tess':
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/' \
        #                      'toi_list_ssectors_dvephemeris_ephmatchnoepochthr0,25.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/final_tce_tables/TOI_2020.01.21_13.55.10.csv_TFOPWG_processed.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/final_tce_tables/toi-plus-tev.mit.edu_2020-01-15_TOI Disposition_processed.csv'
        input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/final_tce_tables/exofop_ctoilists_Community_processed.csv'

        lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

    # shuffle TCE table
    shuffle = False

    output_dir += 'whitened' if whitened else 'nonwhitened'

    if gapped:
        output_dir += '_gapped'
    if gap_imputed:
        output_dir += '_imputed'
    if gap_with_confidence_level:
        output_dir += '_conf%d' % int(gap_confidence_level * 100)
    if use_tps_ephem:
        output_dir += '_tps'

    # # input checks
    # assert(satellite in ['kepler', 'tess'])
    #
    # if not gapped:
    #     assert not gap_imputed
    #
    # if gap_with_confidence_level:
    #     assert gapped
    #
    # # if use_ground_truth:
    # #     assert satellite == 'tess'

    # multiprocessing parameters
    using_mpi = False  # parallelization without MPI processes

    # number of processes spawned
    num_worker_processes = 10  # number of workers

    # TODO: do I need to assert that the sum equal the number of worker processes? In one case processes are not used,
    #       in the other, shards wait for others to be finished, right?
    # number of shards (tfrecords)
    if training:
        num_train_shards, num_test_shards, num_val_shards = 8, 1, 1
        assert num_train_shards + num_val_shards + num_test_shards == num_worker_processes
    else:
        num_pred_shards = 10
        assert num_pred_shards == num_worker_processes


def _process_file_shard(tce_table, file_name, eph_table):
    """ Processes a single file shard.

    Args:
    tce_table: A Pandas DataFrame containing the TCEs ephemeris in the shard
    file_name: The output TFRecord file
    eph_table: A Pandas DataFrame containing the complete TCEs ephemeris database - needed when gapping TCEs from the
    light curve
    """

    # (tce_table, file_name) = inputi
    # tce_table = tce_shard[0]
    # lcs, times = tce_shard[1], tce_shard[2]
    process_name = multiprocessing.current_process().name
    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)
    tf.logging.info("%s: Processing %d items in shard %s", process_name, shard_size, shard_name)

    # get preprocessing configuration parameters
    config = Config()

    # load confidence dictionary
    confidence_dict = pickle.load(open(config.dict_savedir, 'rb')) if config.gap_with_confidence_level else {}

    with tf.python_io.TFRecordWriter(file_name) as writer:
        num_processed = 0
        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows

            lc, time = None, None
            if config.whitened:
            # # get flux and cadence data (if using whitened data)
            # # check if TCE is in the whitened dataset
            # if config.satellite == 'kepler' and config.whitened and not (tce['kepid'] in flux_import and tce['kepid']
            #                                                              in time_import):
            #     lc, time = (flux_import[tce['kepid']][1], time_import[tce['kepid']][1]) \
            #         if config.whitened else (None, None)
                raise NotImplementedError('Whitening still not implemented-ish.')
            # else:
            #     continue

            # preprocess TCE and add it to the tfrecord
            example = _process_tce(tce, eph_table, lc, time, config, confidence_dict)
            if example is not None:
                writer.write(example.SerializeToString())

            num_processed += 1
            if not num_processed % 10:
                tf.logging.info("%s: Processed %d/%d items in shard %s",
                                process_name, num_processed, shard_size, shard_name)

    tf.logging.info("%s: Wrote %d items in shard %s", process_name, shard_size, shard_name)


def create_shards(config, tce_table):
    """ Distributes TCEs across shards for preprocessing.

    :param config:      The config object
    :param tce_table:   TCE table
    :param eph_table:   Ephemeris table
    :return:            shards
    """

    file_shards = []
    num_tces = len(tce_table)

    if config.training:  # Training/Validation/Test

        train_cutoff = int(config.datasets_frac['training'] * num_tces)
        val_cutoff = int(config.datasets_frac['validation'] * num_tces)

        train_tces = tce_table[:train_cutoff]
        val_tces = tce_table[train_cutoff:val_cutoff+train_cutoff]
        test_tces = tce_table[val_cutoff+train_cutoff:]

        tf.logging.info("Partitioned %d TCEs into training (%d), validation (%d) and test (%d)",
                      num_tces, len(train_tces), len(val_tces), len(test_tces))

        boundaries = np.linspace(0, len(train_tces), config.num_train_shards + 1).astype(np.int)
        for i in range(config.num_train_shards):
            start = boundaries[i]
            end = boundaries[i + 1]
            filename = os.path.join(config.output_dir, "train-{:05d}-of-{:05d}".format(i, config.num_train_shards))
            file_shards.append((train_tces[start:end], filename, tce_table))

        boundaries = np.linspace(0, len(test_tces), config.num_test_shards + 1).astype(np.int)
        for i in range(config.num_test_shards):
            start = boundaries[i]
            end = boundaries[i + 1]
            filename = os.path.join(config.output_dir, "test-{:05d}-of-{:05d}".format(i, config.num_test_shards))
            file_shards.append((test_tces[start:end], filename, tce_table))

        boundaries = np.linspace(0, len(val_tces), config.num_val_shards + 1).astype(np.int)
        for i in range(config.num_test_shards):
            start = boundaries[i]
            end = boundaries[i + 1]
            filename = os.path.join(config.output_dir, "val-{:05d}-of-{:05d}".format(i, config.num_test_shards))
            file_shards.append((test_tces[start:end], filename, tce_table))

        # # Validation has a single shard
        # file_shards.append((val_tces, os.path.join(config.output_dir, "val-00000-of-00000"), tce_table))
        # #file_shards.append((test_tces, os.path.join(config.output_dir, "test-00000-of-00000"), eph_table))

    else:  # Predictions

        tf.logging.info("Partitioned %d TCEs into predict (%d)", num_tces, len(tce_table))
        boundaries = np.linspace(0, len(tce_table), config.num_pred_shards + 1).astype(np.int)

        for i in range(config.num_train_shards):
            start = boundaries[i]
            end = boundaries[i + 1]
            filename = os.path.join(config.output_dir, "predict-{:05d}-of-{:05d}".format(i, config.num_train_shards))
            file_shards.append((tce_table[start:end], filename, tce_table))

    return file_shards


def main(_):

    # get the configuration parameters
    config = Config()

    # make the output directory if it doesn't already exist
    tf.gfile.MakeDirs(config.output_dir)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config.plot_figures:
        tf.gfile.MakeDirs(os.path.join(config.output_dir, 'plots'))

    # if config.save_stats:
    #     tf.gfile.MakeDirs(os.path.join(config.output_dir, 'stats'))

    # get TCE and gapping ephemeris tables
    tce_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                 else get_tess_tce_table(config))

    # shuffle TCE table
    if config.shuffle:
        tce_table = shuffle_tce(tce_table, seed=123)
        print('Shuffled TCE Table')

    if config.whitened:  # get flux and cadence time series for the whitened data
        load_whitened_data(config)

    file_shards = create_shards(config, tce_table)

    num_file_shards = len(file_shards)

    # launch subprocesses for the file shards
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

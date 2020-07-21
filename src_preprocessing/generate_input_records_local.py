"""
Generate preprocessed TFRecords locally.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_kepler_tce_table, get_tess_tce_table, \
    load_whitened_data, shuffle_tce, normalize_params_tce_table


class Config:
    """ Class that creates configuration objects that hold parameters required for the preprocessing.
    """

    # TFRecords base name
    tfrecords_base_name = 'tfrecordstess_spoctois_g2001-l201_gbal_spline_nongapped_flux-centroid-oddeven-scalarnoDV'
    # tfrecords_base_name = 'test_tfrecordskeplerdr25_norobovetterkois_g2001-l201_gbal_spline_nongapped_flux-centroid-oddeven-scalar'

    # TFRecords root directory
    tfrecords_dir = os.path.join('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords',
                                 'TESS',  # either 'Kepler' of 'TESS'
                                 # 'DR25'
                                 )

    # output directory
    output_dir = os.path.join(tfrecords_dir, tfrecords_base_name)

    satellite = 'tess'  # choose from: ['kepler', 'tess']
    # multisector = True  # True for TESS multi-sector runs
    # sectors = np.arange(1, 19)  # only for TESS
    tce_identifier = 'oi'  # either 'tce_plnt_num' or 'oi'

    augmentation = False  # if True, it augments the dataset by applying augmentation techniques to the TCE data
    num_examples_per_tce = 1 if augmentation else 1  # number of examples created per TCE
    aug_noise_factor = 0.05
    # values used to replace ephemerides uncertainties which were not estimated
    tce_min_err = {'tce_period': 6.39723e-9, 'tce_duration': 0.000385 / 24, 'tce_time0bk': 2.44455e-6}

    whitened = False  # use whitened data (currently only available for Kepler DR25 34k TCEs dataset)

    # minimum gap size(in time units - day) for a split
    gapWidth = 0.75

    # gapping - remove other TCEs belonging to the same target; if conducting a multi-sector run, check only for TCEs
    # in the same sector
    gapped = False
    gap_keep_ovelap = False
    gap_padding = True
    gap_imputed = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    # binning parameters
    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view
    bin_width_factor_glob = 0.16  # 1 / num_bins_glob
    bin_width_factor_loc = 0.16
    num_durations = 4  # number of transit duration to include in the local view: 2 * num_durations + 1

    # which time-series data to compute additionaly to the flux time-series features
    # odd-even flux time-series are computed based on the flux time-series, so by default this feature is computed
    # TODO: add if statements to the preprocessing to take care of this; currently not being used
    time_series_extracted = ['centroid', 'weak_secondary_flux']

    # True to load denoised centroid time-series instead of the raw from the FITS files
    get_denoised_centroids = False

    # if True, CCD module pixel coordinates are used. If False, local CCD pixel coordinates are transformed into RA and
    # Dec (world coordinates)
    px_coordinates = False

    # if True, saves plots of several preprocessing steps
    plot_figures = True

    prefer_psfcentr = False  # if True, PSF centroids are used, when available, instead of MOM centroids

    # simulated data
    injected_group = False  # either 'False' or inject group name
    light_curve_extension = 'LIGHTCURVE'  # either 'LIGHTCURVE' of 'INJECTED LIGHTCURVE' for injected data
    # either 'None' for not scrambling the quarters, or 'SCR1', 'SCR2' and 'SCR3' to use one of the scrambled groups
    scramble_type = None
    invert = False  # if True, raw light curves are inverted

    omit_missing = True  # skips target IDs that are not in the FITS files

    # list with the names of the scalar parameters from the TCE table (e.g. stellar parameters) that are also added to
    # the TFRecords along with the time-series features (views). Set list to empty to not add any scalar parameter.
    # These parameters are added to the example in the TFRecord as a list of float values.
    # scalar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens', 'wst_robstat',
    #                  'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']
    # Kepler with DV
    # scalar_params = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth',
    #                  'tce_bin_oedp_stat', 'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat']
    # Kepler with TPS or TESS (for now)
    scalar_params = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens']

    # path to updated TCE table, PDC time series fits files and confidence level dictionary
    if satellite.startswith('kepler'):

        # TCE table filepath
        input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
                             'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv'

        # FITS files directory
        lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'

        dict_savedir = ''  # '/home/lswilken/Documents/Astronet_Simplified/pc_confidence_kepler_q1q17'

    elif satellite == 'tess':
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/' \
        #                      'toi_list_ssectors_dvephemeris_ephmatchnoepochthr0,25.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/final_tce_tables/TOI_2020.01.21_13.55.10.csv_TFOPWG_processed.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/final_tce_tables/toi-plus-tev.mit.edu_2020-01-15_TOI Disposition_processed.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/final_tce_tables/' \
        #                      'exofop_ctoilists_Community_processed.csv'
        input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/tois_stellar_missvaltosolar_procols_spoc.csv'

        lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

        dict_savedir = ''

    # shuffle TCE table
    shuffle = False
    shuffle_seed = 24

    # multiprocessing parameters
    using_mpi = False  # parallelization without MPI processes

    # number of processes spawned
    num_worker_processes = 10

    # number of shards generated
    num_shards = 10


def _process_file_shard(tce_table, file_name, eph_table):
    """ Processes a single file shard.

    Args:
    tce_table: A Pandas DataFrame containing the TCEs ephemeris in the shard
    file_name: The output TFRecord filename
    eph_table: A Pandas DataFrame containing the complete TCEs ephemeris database - needed when gapping TCEs from the
               data
    config: The preprocessing config object
    """

    # get preprocessing configuration parameters
    config = Config()

    process_name = multiprocessing.current_process().name
    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)

    tceColumns = ['target_id', config.tce_identifier]
    columnsDf = tceColumns + ['augmentation_idx', 'shard']
    firstTceInDf = True

    tf.logging.info('{}: Processing {} items in shard {}'.format(process_name, shard_size, shard_name))

    # load confidence dictionary
    confidence_dict = pickle.load(open(config.dict_savedir, 'rb')) if config.gap_with_confidence_level else {}

    with tf.python_io.TFRecordWriter(file_name) as writer:

        num_processed = 0

        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows
            tf.logging.info('{}: Processing TCE {}-{} in shard {}'.format(process_name, tce['target_id'],
                                                                          tce[config.tce_identifier],
                                                                          shard_name))

            # preprocess TCE and add it to the TFRecord
            for example_i in range(config.num_examples_per_tce):

                tce['augmentation_idx'] = example_i

                example = _process_tce(tce, eph_table, config, confidence_dict)

                tf.logging.info('{}: Finished processing TCE {}-{} in shard {}'.format(process_name, tce['target_id'],
                                                                                       tce[config.tce_identifier],
                                                                                       shard_name))
                if example is not None:
                    tf.logging.info('{}: Writing TCE {}-{} to shard {}'.format(process_name, tce['target_id'],
                                                                               tce[config.tce_identifier],
                                                                               shard_name))
                    writer.write(example.SerializeToString())
                    tf.logging.info('{}: Finished writing TCE {}-{} to shard {}'.format(process_name, tce['target_id'],
                                                                                        tce[config.tce_identifier],
                                                                                        shard_name))

                    tceData = {column: [tce[column]] for column in tceColumns}
                    tceData['shard'] = [shard_name]
                    tceData['augmentation_idx'] = [example_i]
                    exampleDf = pd.DataFrame(data=tceData,
                                             columns=columnsDf)
                    if firstTceInDf:
                        tf.logging.info('first TCE in shard {}'.format(shard_name))
                        examplesDf = exampleDf
                        firstTceInDf = False
                    else:
                        tf.logging.info('Not first TCE in shard {}'.format(shard_name))
                        examplesDf = pd.read_csv(os.path.join(config.output_dir, '{}.csv'.format(shard_name)))
                        examplesDf = pd.concat([examplesDf, exampleDf])

                    tf.logging.info('Saving TCE info to shard {} csv file'.format(shard_name))
                    examplesDf.to_csv(os.path.join(config.output_dir, '{}.csv'.format(shard_name)), index=False)

            num_processed += 1
            if not num_processed % 1:
                tf.logging.info('{}: Processed {}/{} items in shard {}'.format(process_name, num_processed, shard_size,
                                                                               shard_name))

    tf.logging.info('{}: Finished processing {} items in shard {}'.format(process_name, shard_size, shard_name))


def create_shards(config, tce_table):
    """ Distributes TCEs across shards for preprocessing.

    :param config:      The config object
    :param tce_table:   TCE table
    :return:
       file_shards
    """

    file_shards = []

    tf.logging.info('Partitioned {} TCEs into {} shards'.format(len(tce_table), config.num_shards))
    boundaries = np.linspace(0, len(tce_table), config.num_shards + 1).astype(np.int)

    for i in range(config.num_shards):
        start = boundaries[i]
        end = boundaries[i + 1]
        filename = os.path.join(config.output_dir, 'shard-{:05d}-of-{:05d}'.format(i, config.num_shards))
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

    # get TCE and gapping ephemeris tables
    tce_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                 else get_tess_tce_table(config))

    # shuffle TCE table
    if config.shuffle:
        tce_table = shuffle_tce(tce_table, seed=config.shuffle_seed)
        print('Shuffled TCE Table')

    file_shards = create_shards(config, tce_table)

    # launch subprocesses for the file shards
    # num_processes = min(config.num_shards, config.num_worker_processes)
    tf.logging.info('Launching {} subprocesses for {} total file shards'.format(config.num_worker_processes,
                                                                                config.num_shards))

    pool = multiprocessing.Pool(processes=config.num_worker_processes)
    async_results = [pool.apply_async(_process_file_shard, file_shard) for file_shard in file_shards]
    pool.close()

    # Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
    for async_result in async_results:
        async_result.get()

    tf.logging.info('Finished processing {} total file shards'.format(config.num_shards))


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=main)

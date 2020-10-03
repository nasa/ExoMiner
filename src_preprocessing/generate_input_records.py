""""
Main script used to generate TFRecords --on the supercomputer cluster-- used as input to deep learning model
for classifying Threshold Crossing Events.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from mpi4py import MPI
import datetime
import socket
import pandas as pd
from tensorflow.compat.v1 import logging as tf_logging
from tensorflow.io import gfile, TFRecordWriter

# local
from src_preprocessing.preprocess import _process_tce
# from src_preprocessing.preprocess_sv_fdl import _process_tce
from src_preprocessing.utils_generate_input_records import get_tess_tce_table, get_kepler_tce_table, \
    shuffle_tce


class Config:
    """"
    Config class to specify time series to be processed into TFRecords.
    """

    # TFRecords base name
    tfrecords_base_name = 'tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-6stellar-bfap-ghost-rollingband'

    # TFRecords root directory
    tfrecords_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/tfrecords'

    # output directory
    output_dir = os.path.join(tfrecords_dir,
                              'Kepler',
                              'Q1-Q17_DR25',
                              tfrecords_base_name)

    satellite = 'kepler'  # choose from: ['kepler', 'tess']
    multisector = False  # True for TESS multi-sector runs
    tce_identifier = 'tce_plnt_num'  # either 'tce_plnt_num' or 'oi'

    # offline data augmentation
    augmentation = False  # if True, it augments the dataset by applying augmentation techniques to the TCE data
    num_examples_per_tce = 1 if augmentation else 1  # number of examples created per TCE
    aug_noise_factor = 0.05
    # values used to replace ephemerides uncertainties which were not estimated
    tce_min_err = {'tce_period': 6.39723e-9, 'tce_duration': 0.000385 / 24, 'tce_time0bk': 2.44455e-6}

    # minimum gap size( in time units - day) for a split
    gapWidth = 0.75

    # gapping - remove other TCEs belonging to the same target
    gapped = False
    gap_keep_overlap = False
    gap_padding = 1
    gap_padding_primary = 2
    gap_imputed = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    gap_with_confidence_level = False
    gap_confidence_level = 0.75

    # which time-series data to compute additionaly to the flux time-series features
    # odd-even flux time-series are computed based on the flux time-series, so by default this feature is computed
    # TODO: add if statements to the preprocessing to take care of this; currently not being used
    time_series_extracted = ['centroid', 'weak_secondary_flux']

    # if True, CCD module pixel coordinates are used. If False, pixel coordinates are transformed into RA and Dec
    # (world coordinates)
    px_coordinates = False

    prefer_psfcentr = False  # if True, PSF centroids are used, when available, instead of MOM centroids

    # simulated data
    injected_group = False  # either 'False' or inject group name
    light_curve_extension = 'LIGHTCURVE'  # either 'LIGHTCURVE' of 'INJECTED LIGHTCURVE' for injected data
    # either 'None' for not scrambling the quarters, or 'SCR1', 'SCR2' and 'SCR3' to use one of the scrambled groups
    scramble_type = None
    invert = False  # if True, raw light curves are inverted

    # if True, saves plots of several preprocessing steps
    plot_figures = True

    omit_missing = True  # skips target IDs that are not in the FITS files

    # list with the names of the scalar parameters from the TCE table (e.g. stellar parameters) that are also added to
    # the TFRecords along with the time-series features (views). Set list to empty to not add any scalar parameter.
    # These parameters are added to the example in the TFRecord as a list of float values.
    # scalar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens', 'wst_robstat',
    #                  'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']
    # # Kepler with DV
    scalar_params = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth',
                     'tce_bin_oedp_stat', 'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat',
                     'tce_rb_tcount0']
    # Kepler with TPS
    # scalar_params = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens']

    use_tps_ephem = False  # use TPS ephemeris instead of DV

    # binning parameters
    num_bins_glob = 301  # number of bins in the global view
    num_bins_loc = 21  # number of bins in the local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16
    num_durations = 2.5  # number of transit duration to include in the local view: 2 * num_durations + 1

    # True to load denoised centroid time-series instead of the raw from the FITS files
    get_denoised_centroids = False

    if satellite.startswith('kepler'):

        # TCE table filepath

        # q1-q17 dr25 DV TCEs
        input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
                             'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_rmcandandfpkois_norogues.csv'
        # # q1-q17 dr25 TPS TCE-1s
        # input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
        #                      'tps/keplerTPS_KSOP2536_dr25.csv'
        # # q1-q17 dr25 non-TCEs
        # input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
        #                      'tps/keplerTPS_KSOP2536_nontces.csv'
        # # scrambled TCEs
        # input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
        #                      'simulated_data/scrambled/kplr_dr25_scr3_tces_stellar_processed_withlabels.csv'
        # # inverted TCEs
        # input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
        #                      'simulated_data/inverted/kplr_dr25_inv_tces_stellar_processed_withlabels.csv'

        # PDC light curve FITS files root directory
        # q1-q17 dr25 TCEs
        lc_data_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/Kepler/DR25/' \
                      'pdc-tce-time-series-fits'
        # q1-q17 dr25 TCEs + non-TCEs
        # lc_data_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/Kepler/DR25/' \
        #               'dr_25_all_final'

        dict_savedir = ''

    elif satellite == 'tess':
        input_tce_csv_file = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/TESS/' \
                             ''

        lc_data_dir = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/TESS/pdc-lc/' \
                      'pdc-tce-time-series-fits'

        dict_savedir = ''

    # shuffle TCE table
    shuffle = False
    shuffle_seed = 24

    # multiprocessing parameters
    using_mpi = True

    process_i = MPI.COMM_WORLD.rank
    n_processes = MPI.COMM_WORLD.size

    print('Process {} ({})'.format(process_i, n_processes))
    sys.stdout.flush()


def _process_file_shard(tce_table, file_name, eph_table, config):
    """ Processes a single file shard.

    Args:
    tce_table: A Pandas DataFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
    eph_table: A Pandas DataFrame containing all TCEs in the dataset.
    config: Config object with preprocessing parameters.
    """

    # get shard name and size
    shard_name = os.path.basename(file_name)
    shard_size = len(tce_table)

    # defined columns in the shard table
    tceColumns = ['target_id', config.tce_identifier]
    columnsDf = tceColumns + ['augmentation_idx', 'shard']
    firstTceInDf = True

    tf_logging.info('{}: Processing {} items in shard {}'.format(config.process_i, shard_size, shard_name))

    confidence_dict = pickle.load(open(config.dict_savedir, 'rb')) if config.gap_with_confidence_level else {}

    start_time = int(datetime.datetime.now().strftime("%s"))

    with TFRecordWriter(file_name) as writer:

        num_processed = 0

        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows

            # preprocess TCE and add it to the TFRecord
            for example_i in range(config.num_examples_per_tce):

                tce['augmentation_idx'] = example_i

                example = _process_tce(tce, eph_table, config, confidence_dict)
                if example is not None:
                    writer.write(example.SerializeToString())

                    tceData = {column: [tce[column]] for column in tceColumns}
                    tceData['shard'] = [shard_name]
                    tceData['augmentation_idx'] = [example_i]
                    exampleDf = pd.DataFrame(data=tceData, columns=columnsDf)

                    if firstTceInDf:
                        examplesDf = exampleDf
                        firstTceInDf = False
                    else:
                        examplesDf = pd.read_csv(os.path.join(config.output_dir, '{}.csv'.format(shard_name)))
                        examplesDf = pd.concat([examplesDf, exampleDf])

                    examplesDf.to_csv(os.path.join(config.output_dir, '{}.csv'.format(shard_name)), index=False)

            num_processed += 1
            if config.n_processes < 50 or config.process_i == 0:
                if not num_processed % 10:
                    if config.process_i == 0:
                        cur_time = int(datetime.datetime.now().strftime("%s"))
                        eta = (cur_time - start_time) / num_processed * (shard_size - num_processed)
                        eta = str(datetime.timedelta(seconds=eta))
                        printstr = '{}: Processed {}/{} items in shard {}, ' \
                                   'time remaining (HH:MM:SS): {}'.format(config.process_i, num_processed, shard_size,
                                                                          shard_name, eta)
                    else:
                        printstr = '{}: Processed {}/{} items in shard {}'.format(config.process_i, num_processed,
                                                                                  shard_size, shard_name)

                    tf_logging.info(printstr)

    if config.n_processes < 50:
        tf_logging.info('{}: Wrote {} items in shard {}', config.process_i, shard_size, shard_name)


def main():

    # get the configuration parameters
    config = Config()

    # make the output directory if it doesn't already exist
    gfile.makedirs(config.output_dir)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config.plot_figures:
        gfile.makedirs(os.path.join(config.output_dir, 'plots'))

    # get TCE and gapping ephemeris tables
    tce_table, eph_table = (get_kepler_tce_table(config) if config.satellite == 'kepler'
                            else get_tess_tce_table(config))

    # TODO: does this ensure that all processes shuffle the same way? it does call np.random.seed inside the function
    # shuffle tce table
    if config.shuffle:
        tce_table = shuffle_tce(tce_table, seed=config.shuffle_seed)
        print('Shuffled TCE Table')

    node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
    filename = 'shard-{:05d}-of-{:05d}-node-{:s}'.format(config.process_i, config.n_processes, node_id)
    file_name_i = os.path.join(config.output_dir, filename)

    _process_file_shard(tce_table, file_name_i, eph_table, config)

    tf_logging.info('Finished processing {} items in shard {}'.format(len(tce_table), filename))


if __name__ == "__main__":

    tf_logging.set_verbosity(tf_logging.INFO)

    # tf_app.run(main=main)
    main()

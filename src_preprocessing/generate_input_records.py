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
from tensorflow.io import TFRecordWriter
from pathlib import Path
import json
from datetime import datetime

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_tess_tce_table, get_kepler_tce_table, \
    shuffle_tce, is_jsonable


def create_preprocessing_config():
    """" Creates configuration objects that hold parameters required for the preprocessing. """

    config = {}

    # TFRecords base name
    config['tfrecords_base_name'] = f'tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-6stellar-bfap-ghost-rollingband_{datetime.now().strftime("%m-%d-%Y_%H-%M")}'

    # TFRecords root directory
    config['tfrecords_dir'] = Path('/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/tfrecords')

    # output directory
    config['output_dir'] = config['tfrecords_dir'] / 'Kepler' / 'Q1-Q17_DR25' / config['tfrecords_base_name']

    config['satellite'] = 'kepler'  # choose from 'kepler', 'tess'
    config['tce_identifier'] = 'tce_plnt_num'  # either 'tce_plnt_num' or 'oi'

    # offline data augmentation
    # if True, it augments the dataset by applying augmentation techniques to the TCE data
    config['augmentation'] = False
    config['num_examples_per_tce'] = 1 if config['augmentation'] else 1  # number of examples created per TCE
    config['aug_noise_factor'] = 0.05
    # values used to replace ephemerides uncertainties which were not estimated
    config['tce_min_err'] = {'tce_period': 6.39723e-9, 'tce_duration': 0.000385 / 24, 'tce_time0bk': 2.44455e-6}

    # minimum gap size( in time units - day) for a split
    config['gapWidth'] = 0.75

    # gapping - remove other TCEs belonging to the same target
    config['gapped'] = False
    config['gap_keep_overlap'] = False
    config['gap_padding'] = 1
    config['gap_padding_primary'] = 3
    config['gap_imputed'] = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    config['gap_with_confidence_level'] = False
    config['gap_confidence_level'] = 0.75

    # if True, CCD module pixel coordinates are used. If False, pixel coordinates are transformed into RA and Dec
    # (world coordinates)
    config['px_coordinates'] = False

    config['prefer_psfcentr'] = False  # if True, PSF centroids are used, when available, instead of MOM centroids

    # simulated data
    config['injected_group'] = False  # either 'False' or inject group name
    config['light_curve_extension'] = 'LIGHTCURVE'  # either 'LIGHTCURVE' of 'INJECTED LIGHTCURVE' for injected data
    # either 'None' for not scrambling the quarters, or 'SCR1', 'SCR2' and 'SCR3' to use one of the scrambled groups
    config['scramble_type'] = None
    config['invert'] = False  # if True, raw light curves are inverted

    # if True, saves plots of several preprocessing steps
    config['plot_figures'] = True

    config['omit_missing'] = True  # skips target IDs that are not in the FITS files

    # remove positive outliers from the phase folded flux time series
    config['pos_outlier_removal'] = False
    config['pos_outlier_removal_sigma'] = 5
    config['pos_outlier_removal_fill'] = True

    # list with the names of the scalar parameters from the TCE table to be added to the plot of the preprocessed views
    config['scalar_params'] = [
        # 'tce_period',
        # 'tce_duration',
        # 'tce_time0bk',
        'transit_depth',
        'tce_max_mult_ev',
        # secondary parameters
        'tce_maxmes',
        'tce_maxmesd',
        # 'wst_robstat',
        'wst_depth',
        'tce_ptemp_stat',
        'tce_albedo_stat',
        # odd-even
        # 'tce_bin_oedp_stat',
        # centroid
        'tce_fwm_stat',
        'tce_dikco_msky',
        'tce_dikco_msky_err',
        'tce_dicco_msky',
        'tce_dicco_msky_err',
        # other diagnostics
        'tce_cap_stat',
        'tce_hap_stat',
        'tce_rb_tcount0',
        'boot_fap',
        # stellar parameters
        'tce_smass',
        'tce_sdens',
        'tce_steff',
        'tce_slogg',
        'tce_smet',
        'tce_sradius',
    ]

    # binning parameters
    config['num_bins_glob'] = 301  # number of bins in the global view
    config['num_bins_loc'] = 31  # number of bins in the local view
    config['bin_width_factor_glob'] = 1 / config['num_bins_glob']
    config['bin_width_factor_loc'] = 0.16
    config['num_durations'] = 2.5  # number of transit duration to include in the local view: 2 * num_durations

    # True to load denoised centroid time-series instead of the raw from the FITS files
    config['get_denoised_centroids'] = False

    if config['satellite'].startswith('kepler'):

        # TCE table filepath

        # q1-q17 dr25 DV TCEs
        config['input_tce_csv_file'] = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/Kepler/' \
                                       'Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_' \
                                       'norobovetterlabels_renamedcols_nomissingval_rmcandandfpkois_norogues.csv'
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
        config['lc_data_dir'] = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/Kepler/DR25/' \
                                'pdc-tce-time-series-fits'
        # q1-q17 dr25 TCEs + non-TCEs
        # config['lc_data_dir'] = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/Kepler/DR25/' \
        #                         'dr_25_all_final'

        config['dict_savedir'] = ''

    elif config['satellite'] == 'tess':
        config['input_tce_csv_file'] = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/Ephemeris_tables/TESS/' \
                                       ''

        config['lc_data_dir'] = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/TESS/pdc-lc/' \
                                'pdc-tce-time-series-fits'

        config['dict_savedir'] = ''

    # shuffle TCE table
    config['shuffle'] = False
    config['shuffle_seed'] = 24

    # multiprocessing parameters
    config['using_mpi'] = True

    config['process_i'] = MPI.COMM_WORLD.rank
    config['n_processes'] = MPI.COMM_WORLD.size

    print(f'Process {config["process_i"]} ({config["n_processes"]})')
    sys.stdout.flush()

    return config


def _process_file_shard(tce_table, file_name, eph_table, config):
    """ Processes a single file shard.

    Args:
    tce_table: Pandas DataFrame, containing the TCEs in the shard.
    file_name: The output TFRecord file.
    eph_table: Pandas DataFrame, containing all TCEs in the dataset.
    config: dict, with preprocessing parameters.
    """

    # get shard name and size
    shard_name = file_name.name
    shard_size = len(tce_table)

    # defined columns in the shard table
    tceColumns = ['target_id', config['tce_identifier']]
    # columnsDf = tceColumns + ['augmentation_idx', 'shard']
    firstTceInDf = True

    tf_logging.info(f'{config["process_i"]}: Processing {shard_size} items in shard {shard_name}')

    confidence_dict = pickle.load(open(config['dict_savedir'], 'rb')) if config['gap_with_confidence_level'] else {}

    start_time = int(datetime.datetime.now().strftime("%s"))

    with TFRecordWriter(str(file_name)) as writer:

        num_processed = 0

        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows

            # preprocess TCE and add it to the TFRecord
            for example_i in range(config['num_examples_per_tce']):

                tce['augmentation_idx'] = example_i

                example = _process_tce(tce, eph_table, config, confidence_dict)
                if example is not None:
                    example, example_stats = example
                    writer.write(example.SerializeToString())

                    tceData = {column: [tce[column]] for column in tceColumns}
                    tceData['shard'] = [shard_name]
                    tceData['augmentation_idx'] = [example_i]
                    tceData.update({key: [val] for key, val in example_stats.items()})
                    exampleDf = pd.DataFrame(data=tceData)  # , columns=columnsDf)

                    if firstTceInDf:
                        examplesDf = exampleDf
                        firstTceInDf = False
                    else:
                        examplesDf = pd.read_csv(config['output_dir'] / f'{shard_name}.csv', index_col=0)
                        examplesDf = pd.concat([examplesDf, exampleDf], ignore_index=True)

                    examplesDf.to_csv(config['output_dir'] / f'{shard_name}.csv', index=True)

            num_processed += 1
            if config['n_processes'] < 50 or config['process_i'] == 0:
                if not num_processed % 10:
                    if config['process_i'] == 0:
                        cur_time = int(datetime.datetime.now().strftime("%s"))
                        eta = (cur_time - start_time) / num_processed * (shard_size - num_processed)
                        eta = str(datetime.timedelta(seconds=eta))
                        printstr = f'{config["process_i"]}: Processed {num_processed}/{shard_size} items in shard ' \
                                   f'{shard_name}, time remaining (HH:MM:SS): {eta}'
                    else:
                        printstr = f'{config["process_i"]}: Processed {num_processed}/{shard_size} items in shard ' \
                                   f'{shard_name}'

                    tf_logging.info(printstr)

    if config['n_processes'] < 50:
        tf_logging.info(f'{config["process_i"]}: Wrote {shard_size} items in shard {shard_name}')


def main():

    # get the configuration parameters
    config = create_preprocessing_config()

    # make the output directory if it doesn't already exist
    config['output_dir'].mkdir(exist_ok=True)

    # save the JSON file with preprocessing parameters that are JSON serializable
    if config['process_i'] == 0:
        json_dict = {key: val for key, val in config.items() if is_jsonable(val)}
        with open(config['output_dir'] / 'preprocessing_parameters.json', 'w') as params_file:
            json.dump(json_dict, params_file)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config['plot_figures']:
        (config['output_dir'] / 'plots').mkdir(exist_ok=True)

    # get TCE and gapping ephemeris tables
    tce_table, eph_table = (get_kepler_tce_table(config) if config['satellite'] == 'kepler'
                            else get_tess_tce_table(config))

    # TODO: does this ensure that all processes shuffle the same way? it does call np.random.seed inside the function
    # shuffle tce table
    if config['shuffle']:
        tce_table = shuffle_tce(tce_table, seed=config['shuffle_seed'])
        print('Shuffled TCE Table')

    node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
    filename = f'shard-{config["process_i"]:05d}-of-{config["n_processes"]:05d}-node-{node_id:s}'
    file_name_i = config['output_dir'] / filename

    _process_file_shard(tce_table, file_name_i, eph_table, config)

    tf_logging.info(f'Finished processing {len(tce_table)} items in shard {filename}')


if __name__ == "__main__":

    tf_logging.set_verbosity(tf_logging.INFO)

    main()

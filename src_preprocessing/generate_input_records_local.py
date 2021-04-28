""" Generate preprocessed TFRecords locally. """

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
import numpy as np
import pickle
import pandas as pd
from tensorflow.io import TFRecordWriter
from tensorflow.compat.v1 import logging as tf_logging
# import logging
from pathlib import Path
import json

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_kepler_tce_table, get_tess_tce_table, shuffle_tce, \
    is_jsonable


def create_preprocessing_config():
    """ Creates configuration objects that hold parameters required for the preprocessing. """

    config = {}

    # TFRecords base name
    # config['tfrecords_base_name'] = 'tfrecordstess_spoctois_g301-l31_spline_nongapped_flux-loe-wks-centroid-noDV_nosecparams'
    config['tfrecords_base_name'] = '9705459-2_checkoddeven_4-27-2021'

    # TFRecords root directory
    config['tfrecords_dir'] = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords',
                                   'Kepler',  # either 'Kepler' of 'TESS'
                                   'Q1-Q17_DR25'
                                   )

    # output directory
    config['output_dir'] = config['tfrecords_dir'] / config['tfrecords_base_name']

    config['satellite'] = 'kepler'  # choose from: 'kepler', 'tess'
    config['tce_identifier'] = 'tce_plnt_num'  # either 'tce_plnt_num' or 'oi'

    # if True, it augments the dataset by applying augmentation techniques to the TCE data
    config['augmentation'] = False
    config['num_examples_per_tce'] = 1 if config['augmentation'] else 1  # number of examples created per TCE
    config['aug_noise_factor'] = 0.05
    # values used to replace ephemerides uncertainties which were not estimated
    config['tce_min_err'] = {'tce_period': 6.39723e-9, 'tce_duration': 0.000385 / 24, 'tce_time0bk': 2.44455e-6}

    # minimum gap size(in time units - day) for a split
    config['gapWidth'] = 0.75

    # gapping - remove other TCEs belonging to the same target; if conducting a multi-sector run, check only for TCEs
    # in the same sector
    config['gapped'] = False
    config['gap_keep_overlap'] = False
    config['gap_padding'] = 1  # 1 for us, 0.25 for Shallue&Vand/FDL
    config['gap_padding_primary'] = 3
    config['gap_imputed'] = False  # add noise to gapped light curves
    # gap transits of other TCEs only if highly confident these TCEs are planets
    config['gap_with_confidence_level'] = False
    config['gap_confidence_level'] = 0.75

    # binning parameters
    config['num_bins_glob'] = 301  # 301 for us, 2001 for Shallue&Vand/FDL # number of bins in the global view
    config['num_bins_loc'] = 31  # 31 for us, 201 for Shallue&Vand/FDL # number of bins in the local view
    config['bin_width_factor_glob'] = 1 / config['num_bins_glob']  # 0.16
    config['bin_width_factor_loc'] = 0.16
    # number of transit duration to include in the local view: 2 * num_durations + 1
    config['num_durations'] = 2.5  # 2.5 for us, 4 for Shallue&Vand/FDL

    # # True to load denoised centroid time-series instead of the raw from the FITS files
    # get_denoised_centroids = False

    # if True, CCD module pixel coordinates are used. If False, local CCD pixel coordinates are transformed into RA and
    # Dec (world coordinates)
    config['px_coordinates'] = False

    # if True, saves plots of several preprocessing steps
    config['plot_figures'] = True

    config['prefer_psfcentr'] = False  # if True, PSF centroids are used, when available, instead of MOM centroids

    # simulated data
    config['injected_group'] = False  # either 'False' or inject group name
    config['light_curve_extension'] = 'LIGHTCURVE'  # either 'LIGHTCURVE' of 'INJECTED LIGHTCURVE' for injected data
    # either 'None' for not scrambling the quarters, or 'SCR1', 'SCR2' and 'SCR3' to use one of the scrambled groups
    config['scramble_type'] = None  # 'SCR1'  # either None or SCRX where X = {1,2,3}
    config['invert'] = False  # if True, raw light curves are inverted

    config['omit_missing'] = True  # skips target IDs that are not in the FITS files

    # remove positive outliers from the phase folded flux time series
    config['pos_outlier_removal'] = False
    config['pos_outlier_removal_sigma'] = 5
    config['pos_outlier_removal_fill'] = True

    # list with the names of the scalar parameters from the TCE table to be added to the plot of the preprocessed views
    config['scalar_params'] = [
        # 'sectors',
        # 'tce_period',
        # 'tce_duration',
        # 'tce_time0bk',
        # 'transit_depth',
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
        # 'tce_fwm_stat',
        'tce_dikco_msky',
        'tce_dikco_msky_err',
        'tce_dicco_msky',
        'tce_dicco_msky_err',
        # other diagnostics
        'tce_cap_stat',
        'tce_hap_stat',
        # 'tce_cap_hap_stat_diff',
        # 'tce_rb_tcount0',
        # 'tce_rb_tcount0n',
        'boot_fap',
        # stellar parameters
        'tce_smass',
        'tce_sdens',
        'tce_steff',
        'tce_slogg',
        'tce_smet',
        'tce_sradius',
        'mag',
        # transit fit parameters
        # 'tce_impact',
        'tce_prad',
    ]

    # path to updated TCE table, PDC time series fits files and confidence level dictionary
    if config['satellite'].startswith('kepler'):

        # TCE table filepath
        config['input_tce_csv_file'] = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/' \
                                       'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/' \
        #                      'keplerTPS_KSOP2536_dr25_symsecphase_confirmedkoiperiod.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled_Q1-Q17_DR25/' \
        #                      'kplr_dr25_scr1_tces_stellar_processed_withlabels.csv'
        # input_tce_csv_file = '/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/' \
        #                      'keplerTPS_KSOP2536_nontces.csv'

        # FITS files directory
        config['lc_data_dir'] = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'
        # config['lc_data_dir'] = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'

        config['dict_savedir'] = ''  # '/home/lswilken/Documents/Astronet_Simplified/pc_confidence_kepler_q1q17'

    elif config['satellite'] == 'tess':

        config['input_tce_csv_file'] = '/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-22-2021/exofop_toilists_nomissingpephem_sectors.csv'

        config['lc_data_dir'] = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

        config['dict_savedir'] = ''

    # shuffle TCE table
    config['shuffle'] = False
    config['shuffle_seed'] = 24

    # multiprocessing parameters
    config['using_mpi'] = False  # parallelization without MPI processes

    # number of processes spawned
    config['num_worker_processes'] = 10

    # number of shards generated
    config['num_shards'] = 10

    return config


def _process_file_shard(tce_table, file_name, eph_table):
    """ Processes a single file shard.

    Args:
    tce_table: Pandas DataFrame, containing the TCEs ephemeris in the shard
    file_name: Path, output TFRecord filename
    eph_table: Pandas DataFrame, containing the complete TCEs ephemeris database - needed when gapping TCEs from the
               data
    config: dict, preprocessing parameters
    """

    # get preprocessing configuration parameters
    config = create_preprocessing_config()

    process_name = multiprocessing.current_process().name
    shard_name = file_name.name
    shard_size = len(tce_table)

    tceColumns = ['target_id', config['tce_identifier']]
    # columnsDf = tceColumns + ['augmentation_idx', 'shard']
    firstTceInDf = True

    tf_logging.info(f'{process_name}: Processing {shard_size} items in shard {shard_name}')

    # load confidence dictionary
    confidence_dict = pickle.load(open(config['dict_savedir'], 'rb')) if config['gap_with_confidence_level'] else {}

    with TFRecordWriter(str(file_name)) as writer:

        num_processed = 0

        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows
            tf_logging.info(f'{process_name}: Processing TCE {tce["target_id"]}-{tce[config["tce_identifier"]]} in '
                            f'shard {shard_name}')

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
            if not num_processed % 1:
                tf_logging.info(f'{process_name}: Processed {num_processed}/{shard_size} items in shard {shard_name}')

    tf_logging.info(f'{process_name}: Finished processing {shard_size} items in shard {shard_name}')


def create_shards(config, tce_table):
    """ Distributes TCEs across shards for preprocessing.

    :param config: dict, preprocessing parameters
    :param tce_table: Pandas DataFrame, TCE table
    :return:
       list, subset of the TCE table for each shard
    """

    file_shards = []

    tf_logging.info(f'Partitioned {len(tce_table)} TCEs into {config["num_shards"]} shards')
    boundaries = np.linspace(0, len(tce_table), config['num_shards'] + 1).astype(np.int)

    for i in range(config['num_shards']):
        start = boundaries[i]
        end = boundaries[i + 1]
        filename = config['output_dir'] / f'shard-{i:05d}-of-{config["num_shards"]:05d}'
        file_shards.append((tce_table[start:end], filename, tce_table))

    return file_shards


def main():

    # get the configuration parameters
    config = create_preprocessing_config()

    # make the output directory if it doesn't already exist
    config['output_dir'].mkdir(exist_ok=True)

    # save the JSON file with preprocessing parameters that are JSON serializable
    json_dict = {key: val for key, val in config.items() if is_jsonable(val)}
    with open(config['output_dir'] / 'preprocessing_parameters.json', 'w') as params_file:
        json.dump(json_dict, params_file)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config['plot_figures']:
        (config['output_dir'] / 'plots').mkdir(exist_ok=True)

    # get TCE and gapping ephemeris tables
    tce_table = (get_kepler_tce_table(config) if config['satellite'] == 'kepler'
                 else get_tess_tce_table(config))

    # shuffle TCE table
    if config['shuffle']:
        tce_table = shuffle_tce(tce_table, seed=config['shuffle_seed'])
        print('Shuffled TCE Table')

    file_shards = create_shards(config, tce_table)

    # launch subprocesses for the file shards
    # num_processes = min(config.num_shards, config.num_worker_processes)
    tf_logging.info(f'Launching {config["num_worker_processes"]} subprocesses for {config["num_shards"]} total file '
                    f'shards')

    pool = multiprocessing.Pool(processes=config['num_worker_processes'])
    async_results = [pool.apply_async(_process_file_shard, file_shard) for file_shard in file_shards]
    pool.close()

    # async_result.get() to ensure any exceptions raised by the worker processes are raised here
    for async_result in async_results:
        async_result.get()

    tf_logging.info(f'Finished processing {config["num_shards"]} total file shards')


if __name__ == "__main__":

    tf_logging.set_verbosity(tf_logging.INFO)

    main()

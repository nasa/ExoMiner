"""
Main script used to generate TFRecords to be used as input to models.
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
import multiprocessing
import numpy as np
import yaml

# local
from src_preprocessing.preprocess import _process_tce
from src_preprocessing.utils_generate_input_records import get_tess_tce_table, get_kepler_tce_table, shuffle_tce
from utils.utils_dataio import is_yamlble
from src_preprocessing.merge_tfrecords_csv_files import create_shards_table
from paths import path_main


def _process_file_shard(tce_table, file_name, eph_table, config):
    """ Processes a single file shard using an MPI process.

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
    tceColumns = [
        # 'target_id',
        # config['tce_identifier'],
        'uid'
    ]
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
            if config['process_i'] == 0:
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

    if config['n_shards'] < 50:
        tf_logging.info(f'{config["process_i"]}: Wrote {shard_size} items in shard {shard_name}')


def _process_file_shard_local(tce_table, file_name, eph_table, config):
    """ Processes a single file shard locally.

    Args:
    tce_table: Pandas DataFrame, containing the TCEs ephemeris in the shard
    file_name: Path, output TFRecord filename
    eph_table: Pandas DataFrame, containing the complete TCEs ephemeris database - needed when gapping TCEs from the
               data
    config: dict, preprocessing parameters
    """

    process_name = multiprocessing.current_process().name
    shard_name = file_name.name
    shard_size = len(tce_table)

    tceColumns = [
        # 'target_id',
        # config['tce_identifier'],
        'uid'
    ]
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

    tf_logging.info(f'Partitioned {len(tce_table)} TCEs into {config["n_shards"]} shards')
    boundaries = np.linspace(0, len(tce_table), config['n_shards'] + 1).astype(int)

    for i in range(config['n_shards']):
        start = boundaries[i]
        end = boundaries[i + 1]
        filename = config['output_dir'] / f'shard-{i:05d}-of-{config["n_shards"]:05d}'
        file_shards.append((tce_table[start:end], filename, tce_table, config))

    return file_shards


def main():
    # get the configuration parameters
    path_to_yaml = Path(path_main + 'src_preprocessing/config_preprocessing.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    config['output_dir'] = Path(config['output_dir'])
    config['bin_width_factor_glob'] = 1 / config['num_bins_glob']
    config['tess_to_kepler_px_scale_factor'] = config['tess_px_scale'] / config['kepler_px_scale']
    config['tce_min_err']['tce_duration'] = config['tce_min_err']['tce_duration'] / 24

    if config['using_mpi']:
        config['process_i'] = MPI.COMM_WORLD.rank
        config['n_shards'] = MPI.COMM_WORLD.size
        config['n_processes'] = MPI.COMM_WORLD.size
        print(f'Process {config["process_i"]} ({config["n_shards"]})')
        sys.stdout.flush()
    else:
        config['process_i'] = -1
        config['n_shards'] = 10
        config['n_processes'] = 10

    # make the output directory if it doesn't already exist
    config['output_dir'].mkdir(exist_ok=True)

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

    if config['process_i'] in [0, -1]:
        # save the YAML file with preprocessing parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['output_dir'] / 'preprocessing_params.yaml', 'w') as preproc_run_file:
            yaml.dump(json_dict, preproc_run_file)

    if config['using_mpi']:  # use MPI

        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
        filename = f'shard-{config["process_i"]:05d}-of-{config["n_shards"]:05d}-node-{node_id:s}'
        file_name_i = config['output_dir'] / filename

        _process_file_shard(tce_table, file_name_i, eph_table, config)

        tf_logging.info(f'Finished processing {len(tce_table)} items in shard {filename}')

        if config['process_i'] == 0:
            tf_logging.info(f'END-PI:{config["output_dir"]}')

    else:  # use multiprocessing.Pool

        # tce_table = (tce_table.loc[tce_table['label'].isin(['KP', 'CP', 'FP'])]).sample(n=20).reset_index(drop=True)
        file_shards = create_shards(config, tce_table)

        # launch subprocesses for the file shards
        # n_process = min(config.num_shards, config.num_worker_processes)
        tf_logging.info(
            f'Launching {config["n_processes"]} processes for {config["n_shards"]} total file '
            f'shards')

        pool = multiprocessing.Pool(processes=config['n_shards'])
        async_results = [pool.apply_async(_process_file_shard_local, file_shard) for file_shard in file_shards]
        pool.close()

        # async_result.get() to ensure any exceptions raised by the worker processes are raised here
        for async_result in async_results:
            async_result.get()

        tf_logging.info(f'Finished processing {config["n_shards"]} total file shards')

        # concatenates shard tables into a single one
        create_shards_tbl_flag = create_shards_table(config['output_dir'])
        if not create_shards_tbl_flag:
            tf_logging.info('Merged shard table not created.')


if __name__ == "__main__":
    tf_logging.set_verbosity(tf_logging.INFO)

    main()

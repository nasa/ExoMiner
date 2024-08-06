"""
Utility functions used for generating input TFRecords.
"""

# 3rd party
import pandas as pd
import numpy as np
import json
import logging
import multiprocessing
import datetime
from tensorflow.io import TFRecordWriter

# local
from src_preprocessing.lc_preprocessing.preprocess import process_tce
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import report_exclusion

logger = logging.getLogger(__name__)


def process_file_shard(tce_table, file_name, eph_table, config):
    """ Processes a single file shard.

    Args:
    tce_table: Pandas DataFrame, containing the TCEs in the shard.
    file_name: The output TFRecord file.
    eph_table: Pandas DataFrame, containing all TCEs in the dataset.
    config: dict, with preprocessing parameters.
    """

    if not config['using_mpi']:
        config["process_i"] = multiprocessing.current_process().name
        logging.basicConfig(filename=config['output_dir'] / 'preprocessing_logs' /
                                     f'preprocessing_{config["process_i"]}.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filemode='w',
                            )

    # get shard name and size
    shard_name = file_name.name
    shard_size = len(tce_table)

    # defined columns in the shard table
    tceColumns = [
        'uid'
    ]
    firstTceInDf = True

    logger.info(f'{config["process_i"]}: Processing {shard_size} items in shard {shard_name}.')

    start_time = int(datetime.datetime.now().strftime("%s"))

    with TFRecordWriter(str(file_name)) as writer:

        num_processed = 0

        for index, tce in tce_table.iterrows():  # iterate over DataFrame rows

            logger.info(f'{config["process_i"]}: Processing TCE {tce["uid"]} in shard {shard_name} '
                        f'({num_processed}/{shard_size})...')

            # preprocess TCE and add it to the TFRecord
            for example_i in range(config['num_examples_per_tce']):

                tce['augmentation_idx'] = example_i

                try:
                    example = process_tce(tce, eph_table, config)
                except Exception as error:
                    if len(error.args) == 1:
                        error_log = {'etype': None, 'value': error, 'tb': error.__traceback__}
                    if len(error.args) == 2:
                        error_log = {'etype': None, 'value': error.args[0], 'tb': error.args[1]}
                    report_exclusion(
                        '',
                        config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt',
                        error_log)
                    continue

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
                else:
                    logger.info(f'Example {tce.uid} was `None`.')

            num_processed += 1
            if config['process_i'] == 0:
                if not num_processed % 10:
                    if config['process_i'] == 0:
                        cur_time = int(datetime.datetime.now().strftime("%s"))
                        eta = (cur_time - start_time) / num_processed * (shard_size - num_processed)
                        eta = str(datetime.timedelta(seconds=eta))
                        printstr = f'{config["process_i"]}: Processed {num_processed}/{shard_size} items in shard ' \
                                   f'{shard_name}, time remaining (HH:MM:SS): {eta}.'
                    else:
                        printstr = f'{config["process_i"]}: Processed {num_processed}/{shard_size} items in shard ' \
                                   f'{shard_name}.'

                    logger.info(printstr)

    if config['n_shards'] < 50:
        logger.info(f'{config["process_i"]}: Wrote {shard_size} items in shard {shard_name}.')


def create_shards(config, tce_table):
    """ Distributes examples across shards for preprocessing.

    :param config: dict, preprocessing parameters
    :param tce_table: Pandas DataFrame, TCE table
    :return:
       list, subset of the TCE table for each shard
    """

    file_shards = []

    logger.info(f'Partitioned {len(tce_table)} TCEs into {config["n_shards"]} shards')
    boundaries = np.linspace(0, len(tce_table), config['n_shards'] + 1).astype(int)

    for i in range(config['n_shards']):
        start = boundaries[i]
        end = boundaries[i + 1]
        filename = config['output_dir'] / f'shard-{i:05d}-of-{config["n_shards"]:05d}'
        file_shards.append((tce_table[start:end], filename, tce_table, config))

    return file_shards


def get_tce_table(config):
    """ Get TCE table.

    :param config: dict, preprocessing parameters
    :return:
        tce_table: pandas DataFrame, TCE table
    """

    # read the table with examples
    tce_table = pd.read_csv(config['input_tce_csv_file'])

    # filt_tbl = pd.read_csv('/Users/msaragoc/Downloads/ranking_planets_in_variable_stars_comparison.csv')
    # tce_table = tce_table.loc[tce_table['uid'].isin(filt_tbl['uid'])]
    # tce_table = tce_table.loc[tce_table['uid'].isin(['158657354-1-S14-55'])]
    tce_table = tce_table.sample(n=100, replace=False, random_state=config['random_seed'])

    tce_table["tce_duration"] /= 24  # Convert hours to days.

    logger.info(f'Read TCE table with {len(tce_table)} examples.')

    cols_change_data_type = {
        # 'sectors': str,
        'sector_run': str,
        # 'TOI': str,
        'label': str,
        # 'toi_sectors': str,
        'Comments': str,
        'TESS Disposition': str,
        'TFOPWG Disposition': str
    }
    tce_table = tce_table.astype(dtype={k: v for k, v in cols_change_data_type.items() if k in tce_table.columns})

    # when using external parallelization framework to preprocess chunks of the TCE table in parallel
    if config['using_mpi']:

        boundaries = [int(i) for i in np.linspace(0, len(tce_table), config['n_processes'] + 1)]
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(config['n_processes'])][config['process_i']]

        shard_tce_table = tce_table[indices[0]:indices[1]]

        if not config['gapped']:
            tce_table = None

        return shard_tce_table, tce_table

    return tce_table, None


def shuffle_tce(tce_table, seed=123):
    """ Helper function used to shuffle the tce_table if config.shuffle == True

    :param tce_table:   The non-shuffled TCE table
    :param seed:        Seed used for randomization
    :return:
        tce_table, with rows shuffled
    """

    np.random.seed(seed)

    tce_table = tce_table.iloc[np.random.permutation(len(tce_table))]

    return tce_table


def is_jsonable(x):
    """ Test if object is JSON serializable.

    :param x: object
    :return:
    """

    try:
        json.dumps(x)
        return True

    except Exception as error:
        print(f'Error: {error}')
        return False

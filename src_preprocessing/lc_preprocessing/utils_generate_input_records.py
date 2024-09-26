"""
Utility functions used for generating input TFRecords.
"""

# 3rd party
import pandas as pd
import numpy as np
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
                            force=True
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

        for _, tce in tce_table.iterrows():  # iterate over DataFrame rows

            logger.info(f'{config["process_i"]}: Processing TCE {tce["uid"]} in shard {shard_name} '
                        f'({num_processed}/{shard_size})...')

            # preprocess TCE and add it to the TFRecord
            for example_i in range(config['num_examples_per_tce']):

                tce['augmentation_idx'] = example_i

                try:
                    example = process_tce(tce, eph_table, config)
                except Exception as error:

                    # for python 3.11
                    report_exclusion(
                        '',
                        config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt',
                        error)

                    continue

                if example is not None:
                    example, example_stats = example
                    writer.write(example.SerializeToString())

                    tceData = {column: [tce[column]] for column in tceColumns}
                    tceData['shard'] = [shard_name]
                    tceData['augmentation_idx'] = [example_i]
                    tceData.update({key: [val] for key, val in example_stats.items()})
                    exampleDf = pd.DataFrame(data=tceData)

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
            if not num_processed % 10:
                cur_time = int(datetime.datetime.now().strftime("%s"))
                eta = (cur_time - start_time) / num_processed * (shard_size - num_processed)
                eta = str(datetime.timedelta(seconds=eta))
                printstr = f'{config["process_i"]}: Processed {num_processed}/{shard_size} items in shard ' \
                           f'{shard_name}, time remaining (HH:MM:SS): {eta}.'

                logger.info(printstr)

    logger.info(f'{config["process_i"]}: Wrote {num_processed} items (out of {shard_size} total) in shard '
                f'{shard_name}.')


def create_shards(config, shards_tce_tables, tce_table):
    """ Distributes examples across shards for preprocessing.

    :param config: dict, preprocessing parameters
    :param shards_tce_tables: NumPy array of pandas DataFrame, TCE tables for the set of shards/processes
    :param tce_table: Pandas DataFrame, TCE table
    :return:
       list, each element is a tuple for each shard with a subset of the TCE table, the shard filename, the complete
       TCE table, and the config dict parameters of the preprocessing run
    """

    file_shards = []

    for shard_i in range(config['n_shards']):
        filename = config['output_dir'] / f'shard-{shard_i:05d}-of-{config["n_shards"]:05d}'
        file_shards.append((shards_tce_tables[shard_i], filename, tce_table, config))

    return file_shards


def get_tce_table(config):
    """ Get TCE table.

    :param config: dict, preprocessing parameters
    :return:
        shards_tce_tables: NumPy array of pandas DataFrame, TCE tables for the set of shards/processes
        tce_table: pandas DataFrame, full TCE table
    """

    # read the table with examples
    tce_table = pd.read_csv(config['input_tce_csv_file'])
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

    tce_table["tce_duration"] /= 24  # Convert hours to days.
    # FIXME: this is temporary!!
    tce_table = tce_table.loc[tce_table['sector_run'] != '68']

    # table with TCEs to be preprocessed
    preprocess_tce_table = tce_table.copy(deep=True)
    # preprocess_tce_table = preprocess_tce_table.loc[preprocess_tce_table['uid'].isin(['305633328-1-S14-60'])]
    # preprocess_tce_table = preprocess_tce_table.sample(n=100, replace=False, random_state=config['random_seed'])

    # when using external parallelization framework to preprocess chunks of the TCE table in parallel
    if config['using_mpi']:
        shards_tce_tables = np.array_split(preprocess_tce_table, config['n_processes'])
    else:
        shards_tce_tables = np.array_split(preprocess_tce_table, config['n_shards'])

    return shards_tce_tables, tce_table

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
from src_preprocessing.lc_preprocessing.preprocess import process_target_tces
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import report_exclusion

logger = logging.getLogger(__name__)

LOG_EVERY_N_TCES = 10

def split_list(lst, n):
    """ Splits a list into n sublists as evenly as possible. It can deal with inhomogeneous lists.

    Args:
        lst (list): list of items to be split
        n (float): number of elements to split the list into

    Returns:
        list: list of lists, each containing a subset of the original list
    """
    
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def process_file_shard(tce_table, file_name, config):
    """ Processes a single file shard.

    Args:
        tce_table: Pandas DataFrame, containing the TCEs in the shard.
        file_name: The output TFRecord file.
        config: dict, with preprocessing parameters.
    """

    if not config['external_parallelization']:
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
    # shard_size = len(tce_table)
    num_tces_per_target_lst = [len(tbl) for tbl in tce_table]
    num_tces_total = np.sum(num_tces_per_target_lst)  # number of TCEs to be preprocessed in the shard
    num_targets = len(tce_table)  # number of targets in the tables

    # defined columns in the shard table
    tceColumns = [
        'uid'
    ]
    shard_df_empty = True

    logger.info(f'{config["process_i"]}: Processing {num_tces_total} TCEs ({num_targets} targets) in shard {shard_name}.')

    start_time = int(datetime.datetime.now().timestamp())

    num_targets_processed = 0
    num_tces_processed = 0
    with TFRecordWriter(str(file_name)) as writer:

        for target_i, target_tces_tbl in enumerate(tce_table):
            
            target_tces_tbl = target_tces_tbl.reset_index(drop=True)  # reset index for the target TCEs table
            n_tces_target = len(target_tces_tbl)
            target_uid = target_tces_tbl['target_uid'].iloc[0]
            
            logger.info(f'{config["process_i"]}: Processing {n_tces_target} TCEs for target {target_uid} in shard {shard_name} '
                        f'({num_tces_processed}/{num_tces_total})...')
            
            start_preproc_target_tces = int(datetime.datetime.now().timestamp())
            try:
                examples_tces_dict = process_target_tces(target_uid, target_tces_tbl, config)
            except Exception as error:  # uncaught exception/target-level exception

                # for python 3.11
                report_exclusion(
                    'Uncaught exception; Target-level issue',
                    config['exclusion_logs_dir'] / f'exclusions-{target_uid}.txt',
                    error)

                preproc_target_tces = (int(datetime.datetime.now().timestamp()) - start_preproc_target_tces) / 60
                logger.info(f'Spent {preproc_target_tces} minutes preprocessing {n_tces_target} TCEs for target {target_uid}. Failed preprocessing TCEs with error: {error}.')

                continue

            preproc_target_tces = (int(datetime.datetime.now().timestamp()) - start_preproc_target_tces) / 60
            logger.info(f'Spent {preproc_target_tces} minutes preprocessing {n_tces_target} TCEs for target {target_uid}.')

            if len(examples_tces_dict) > 0:  # at least one TCE preprocessed successfully
                
                # write data from TCE table into the auxiliary shard table for TCEs that were successfully preprocessed
                preprocessed_tce_uids = list([example_tce_uid for example_tce_uid, example_tce in examples_tces_dict.items() if example_tce['processed']])
                tces_data_to_tbl = target_tces_tbl.loc[target_tces_tbl['uid'].isin(preprocessed_tce_uids), tceColumns].to_dict(orient='list')
                
                for example_tce_uid, example_tce in examples_tces_dict.items():
                    
                    tce_in_tbl = target_tces_tbl.loc[target_tces_tbl['uid'] == example_tce_uid]

                    if example_tce['processed']:
                                                
                        # write to TFRecord shard
                        example_tce_data, example_stats = example_tce['data']
                        writer.write(example_tce_data.SerializeToString())

                        # # write data extracted from the preprocessing pipeline into the auxiliary shard table for the TCEs that were successfully preprocessed
                        # tceData['augmentation_idx'] = [example_i]
                        for key, val in example_stats.items():
                            if key not in tces_data_to_tbl:
                                tces_data_to_tbl[key] = [val]
                            else:
                                tces_data_to_tbl[key].append(val)
                    
                    else:  # report issue for TCE
                        report_exclusion(
                            f'TCE {tce_in_tbl["uid"].values[0]}',
                            config['exclusion_logs_dir'] / f'exclusions-{target_uid}.txt',
                            example_tce['error'],
                            )
                    
                examples_df = pd.DataFrame(data=tces_data_to_tbl)
                examples_df['shard'] = shard_name
                logger.info(f'Preprocessed {len(examples_df)}/{n_tces_target} TCEs for target {target_uid} in shard {shard_name}.')

                if shard_df_empty:
                    shard_df = examples_df
                    shard_df_empty = False
                else:
                    shard_df = pd.concat([shard_df, examples_df], ignore_index=True)

                # shard_df.to_csv(config['output_dir'] / f'{shard_name}.csv', index=True)
                shard_df.to_csv(config['output_dir'] / f'{shard_name}.csv', mode='a', header=not (config['output_dir'] / f'{shard_name}.csv').exists(), index=True)

                num_tces_processed += len(examples_df)
                num_targets_processed += 1
                
            num_tces_remaining = np.sum(num_tces_per_target_lst[target_i + 1:])  # remaining TCEs in the tables
            num_tces_failed = np.sum(num_tces_per_target_lst[:target_i + 1]) - num_tces_processed
            if num_tces_processed % LOG_EVERY_N_TCES == 0 and num_tces_processed > 0:
                cur_time = int(datetime.datetime.now().timestamp())
                eta = (cur_time - start_time) / num_tces_processed * num_tces_remaining
                eta = str(datetime.timedelta(seconds=eta))
                printstr = f'{config["process_i"]}: Processed {num_tces_processed}/{num_tces_total} TCEs in {num_targets_processed}/{num_targets} targets in shard ' \
                            f'{shard_name}. Failed to preprocess {num_tces_failed} TCEs. Time remaining (HH:MM:SS): {eta}.'

                logger.info(printstr)

    cur_time = int(datetime.datetime.now().timestamp())
    all_time = cur_time - start_time
    all_time = str(datetime.timedelta(seconds=all_time))
    num_tces_failed_total = num_tces_total - num_tces_processed
    logger.info(f'{config["process_i"]}: Wrote {num_tces_processed}/{num_tces_total} TCEs in {num_targets_processed}/{num_targets} targets to shard ' \
                f'{shard_name}. Failed to preprocess {num_tces_failed_total} TCEs. Spent (HH:MM:SS): {all_time}.')
    

def create_shards(config, shards_tce_tables):
    """ Distributes examples across shards for preprocessing.

    :param config: dict, preprocessing parameters
    :param shards_tce_tables: NumPy array of pandas DataFrame, TCE tables for the set of shards/processes
    :return: list, each element is a tuple for each shard with a subset of the TCE table, the shard filename, 
        and the config dict parameters of the preprocessing run
    """

    file_shards = []

    for shard_i in range(config['n_shards']):
        filename = config['output_dir'] / f'shard-{shard_i:05d}-of-{config["n_shards"]:05d}'
        file_shards.append((shards_tce_tables[shard_i], filename, config))

    return file_shards


def get_tce_table(config):
    """ Get TCE table.

    :param config: dict, preprocessing parameters
    :return:
        shards_tce_tables: NumPy array of pandas DataFrame, TCE tables for the set of shards/processes
    """

    # read the table with examples
    tce_table = pd.read_csv(config['input_tce_csv_file'])
    logger.info(f'Read TCE table with {len(tce_table)} examples.')

    cols_change_data_type = {
        'sector_run': str,
        'label': str,
        'Comments': str,
        'TESS Disposition': str,
        'TFOPWG Disposition': str,
        'sectors_observed': str,
    }
    tce_table = tce_table.astype(dtype={k: v for k, v in cols_change_data_type.items() if k in tce_table.columns})

    tce_table["tce_duration"] /= 24  # Convert hours to days.

    # FIXME: add wst_depth_err to the Kepler Q1-Q17 DR25 TCE table
    if 'wst_depth_err' not in tce_table:
        logger.info('Adding `wst_depth_err` to the TCE table. Setting value to all TCEs as -1')
        tce_table['wst_depth_err'] = -1

    # table with TCEs to be preprocessed
    preprocess_tce_table = tce_table.copy(deep=True)

    # split TCEs into tables for parallel preprocessing
    if config['satellite'] == 'tess':
        group_cols = ['target_id', 'sector_run']
        # set unique target/sector identifier
        preprocess_tce_table['target_uid'] = preprocess_tce_table.apply(lambda x: f'{x["target_id"]}-{x["sector_run"]}', axis=1)
    elif config['satellite'] == 'kepler':
        group_cols = ['target_id']
        # set unique target identifier (same as target id for Kepler)
        preprocess_tce_table['target_uid'] = preprocess_tce_table['target_id'].astype(str)
    else:
        raise ValueError(f'Unknown mission {config["satellite"]}.')
    
    tces_groups = preprocess_tce_table.groupby(group_cols)
    tces_groups = [grp for _, grp in tces_groups]
    
    if config['external_parallelization']:
        shards_tce_tables = split_list(tces_groups, config['n_processes'])
    else:
        shards_tce_tables = split_list(tces_groups, config['n_shards'])
    # # when using external parallelization framework to preprocess chunks of the TCE table in parallel
    # if config['external_parallelization']:
    #     shards_tce_tables = np.array_split(preprocess_tce_table, config['n_processes'])
    # else:
    #     shards_tce_tables = np.array_split(preprocess_tce_table, config['n_shards'])

    return shards_tce_tables

"""
Utility functions for CV dataset preprocessing.
"""

# 3rd party
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path
from tqdm import tqdm
import traceback

# local
from src_preprocessing.utils_manipulate_tfrecords import parse_uid


def log_info(message, logger=None, include_traceback=False):
    """Log information either to stdout or Python Logger if `logger` is not `None`.

    :param str message: log message
    :param Python Logger logger: logger. If `None`, message is printed to stdout
    :param bool include_traceback: if True, includes traceback (requires being called under and try/exception block). Defaults to False
    """
    
    if include_traceback:
        message += "\n" + traceback.format_exc()
        
    if logger:
        logger.info(message)
    else:
        print(message)


def create_shard_fold(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_fps, batch_size=128, log_fp=None):
    """ Create a TFRecord fold for cross-validation based on source TFRecords and fold TCE table.

    :param shard_tbl_fp: Path, shard TCE table filepath
    :param dest_tfrec_dir: Path, destination TFRecord directory
    :param fold_i: int, fold id
    :param src_tfrec_fps: list, Paths to source TFRecord shards
    :param batch_size: int, batch size for source TFRecord
    :param log_fp: Path, log file path; if None, does not create a logger
    :return:
        tces_not_found: pandas DataFrame, fold table for TCEs not found in source TFRecord
    """

    if log_fp:
        # set up logger
        logger = logging.getLogger(name=f'log_fold_{fold_i}')
        logger_handler = logging.FileHandler(filename=log_fp,
                                             mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Starting run...')
    else:
        logger = None

    log_info(f'[Fold {fold_i}] Reading fold TCE table: {shard_tbl_fp}', logger=logger)

    fold_tce_tbl = pd.read_csv(shard_tbl_fp)
    # convert list of TCE uids into a TF tensor
    uids_tensor = tf.constant(fold_tce_tbl['uid'].to_list(), dtype=tf.string)

    tfrec_new_fp = dest_tfrec_dir / f'shard-{f"{fold_i}".zfill(4)}'
    log_info(f'[Fold {fold_i}] Destination TFRecord file: {tfrec_new_fp}', logger=logger)

    log_info(f'[Fold {fold_i}] Iterating over the fold TCE table...', logger=logger)
    n_tces_in_shard = 0
    
    dataset = tf.data.TFRecordDataset(src_tfrec_fps)
    dataset = dataset.map(parse_uid, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda uid, _: tf.reduce_any(tf.equal(uid, uids_tensor)))
    dataset = dataset.batch(batch_size)

    n_tces_in_shard = 0
    tces_found = []

    with tf.io.TFRecordWriter(str(tfrec_new_fp)) as writer:
        for batch in tqdm(dataset, total=(len(fold_tce_tbl) // batch_size + 1), desc=f'Fold {fold_i}'):
            uids_batch, serialized_batch = batch

            for uid, serialized_example in zip(uids_batch.numpy(), serialized_batch.numpy()):
                writer.write(serialized_example)
                n_tces_in_shard += 1
                tces_found.append(uid.decode())

                if n_tces_in_shard % 50 == 0:
                    log_info(
                        f'[Fold {fold_i}] Iterating over fold table {fold_i} {shard_tbl_fp.name} '
                        f'({n_tces_in_shard} out of {len(fold_tce_tbl)})\nNumber of TCEs in the shard: {n_tces_in_shard}...',
                        logger=logger
                    )

    log_info(f'[Fold {fold_i}] Finished iterating over fold table {fold_i} {shard_tbl_fp.name}.', logger=logger)
    
    tces_not_found = fold_tce_tbl.loc[~fold_tce_tbl['uid'].isin(tces_found)]
    
    log_info(f'[Fold {fold_i}] Written {n_tces_in_shard} examples. Missing {len(tces_not_found)}.', logger=logger)

    return tces_not_found


def create_table_shard_example_location(tfrec_dir):
    """ Create shard TCE table that makes tracking of location of TCEs in the shards easier. The table consists  of 4
    columns: target_id, tce_plnt_num, shard filename that contains the example, sequential id for example in the shard
    starting at 0.

    :param tfrec_dir: Path, TFRecord directory
    :return:
    """

    tfrec_fps = [file for file in tfrec_dir.iterdir() if 'shard' in file.stem and not file.suffix == '.csv']

    data_to_df = []
    for tfrec_fp in tfrec_fps:

        tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))
        for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
            tceIdentifierTfrec = example.features.feature['tce_plnt_num'].int64_list.value[0]

            data_to_df.append([targetIdTfrec, tceIdentifierTfrec, tfrec_fp.name, string_i])

    shards_tce_tbl = pd.DataFrame(data_to_df, columns=['target_id', 'tce_plnt_num', 'shard', 'example_i'])
    shards_tce_tbl.to_csv(tfrec_dir / 'shards_tce_tbl.csv', index=False)


if __name__ == '__main__':

    tfrec_dir = Path('')
    create_table_shard_example_location(tfrec_dir)
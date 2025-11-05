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
import math

# local
from src_preprocessing.utils_manipulate_tfrecords import parse_feature, make_filter_by_feature_fn


def create_table_norm_stats_folds(cv_dir, save_fp):
    """Create talbe with the normalization statistics for all folds in the cross-validation experiment.

    :param Path cv_dir: path to CV experiment
    :param Path save_fp: save filepath for table
    """
    
    # grab norm stats directories for all folds
    folds_norm_stats_dirs = list(cv_dir.rglob('norm_stats'))
    print(f'Found {len(folds_norm_stats_dirs)} normalization statistics directories.')
    
    norm_stats_folds_lst = []
    for fold_norm_stats_dir in folds_norm_stats_dirs:
        
        fold_id = fold_norm_stats_dir.parent.name
        
        print(f'[Fold {fold_id}] Iterating through fold normalization statistics in {str(fold_norm_stats_dir)}...')
        
        # grab norm stats CSV files for fold
        norm_stats_fps = sorted(list(fold_norm_stats_dir.glob('train_*_norm_stats.csv')))
        print(f'[Fold {fold_id}] Found {len(norm_stats_fps)} normalization tables for fold {str(fold_norm_stats_dir)}.')
        
        norm_stats_fold = pd.concat([pd.read_csv(norm_stats_fp, index_col='Unnamed: 0') for norm_stats_fp in norm_stats_fps], axis=0)
        norm_stats_fold.rename(columns={'0': fold_id}, inplace=True)
        
        norm_stats_folds_lst.append(norm_stats_fold)
    
    norm_stats_folds = pd.concat(norm_stats_folds_lst, axis=1)
    
    # add metadata and save table
    norm_stats_folds.attrs['CV experiment'] = str(cv_dir)
    norm_stats_folds.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(save_fp, "w") as f:
        for key, value in norm_stats_folds.attrs.items():
            f.write(f"# {key}: {value}\n")
        norm_stats_folds.to_csv(f, index=True)
        
    
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


def create_shard_fold(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_fps, n_shards_fold=1, log_fp=None, filter_field='uid'):
    """ Create a TFRecord fold for cross-validation based on source TFRecords and fold TCE table.

    :param shard_tbl_fp: Path, shard TCE table filepath
    :param dest_tfrec_dir: Path, destination TFRecord directory
    :param fold_i: int, fold id
    :param n_shards_fold: int, number of shards to create per CV fold. By default, it creates one shard per fold
    :param src_tfrec_fps: list, Paths to source TFRecord shards
    # :param batch_size: int, batch size for source TFRecord
    :param log_fp: Path, log file path; if None, does not create a logger
    :param filter_field: str, string feature used to filter examples in source TFRecord dataset. Must also be present in the fold table `shard_tbl_fp`
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
    n_fold_tces = len(fold_tce_tbl)
    log_info(f'[Fold {fold_i}] Number of TCEs in fold table: {n_fold_tces}', logger=logger)
    # convert list of TCE uids into a TF tensor
    uids_tensor = tf.constant(fold_tce_tbl[filter_field].to_list(), dtype=tf.string)

    log_info(f'[Fold {fold_i}] Iterating over the fold TCE table...', logger=logger)
    
    dataset = tf.data.TFRecordDataset(src_tfrec_fps)
    
    # filter examples
    dataset = dataset.map(lambda x: parse_feature(x, filter_field), num_parallel_calls=tf.data.AUTOTUNE)
    filter_fn = make_filter_by_feature_fn(uids_tensor)
    dataset = dataset.filter(lambda feature_value, _: filter_fn(feature_value))

    # estimate number of examples per shard
    expected_n_tces_in_shard = math.ceil(n_fold_tces / n_shards_fold)
    
    log_info(f'[Fold {fold_i}] Writing {n_shards_fold} shards with up to {expected_n_tces_in_shard} examples each.', logger=logger)

    # # batch examples
    # batch_size = min(batch_size, expected_n_tces_in_shard)
    # dataset = dataset.batch(batch_size)
    
    tces_found = []
    shard_index = 0
    example_index = 0
    example_shard_index = 0
    writer = None
    for uid_tensor, serialized_example in tqdm(dataset, total=n_fold_tces, desc=f'TCEs in fold {fold_i}'):
        
        if example_index % expected_n_tces_in_shard == 0:  # move to next shard
            
            if shard_index != 0:
                log_info(f'[Fold {fold_i} | Shard {shard_index}] Iterated over fold table {fold_i} {shard_tbl_fp.name} '
                        f'({example_index} out of {n_fold_tces} TCEs)\nNumber of TCEs in the shard: {example_shard_index}/{expected_n_tces_in_shard}.',
                        logger=logger)
            if writer:
                writer.close()
            
            tfrec_fp = dest_tfrec_dir / f'shard-fold-{fold_i}_{shard_index:04d}-{n_shards_fold:04d}.tfrecord'
            
            # create writer
            writer = tf.io.TFRecordWriter(str(tfrec_fp))
            log_info(f'[Fold {fold_i} | Shard {shard_index}] Writing to {tfrec_fp}', logger=logger)
            
            example_shard_index = 0
            shard_index += 1

        writer.write(serialized_example.numpy())
        tces_found.append(uid_tensor.numpy().decode())
        example_shard_index += 1
        example_index += 1
        
        if example_index % 100 == 0:
            log_info(f'[Fold {fold_i} | Shard {shard_index}] Iterating over fold table {fold_i} {shard_tbl_fp.name} '
                    f'({example_index} out of {n_fold_tces} TCEs)\nNumber of TCEs in the shard: {example_shard_index}/{expected_n_tces_in_shard}.',
                    logger=logger)

    if writer:
        writer.close()

    log_info(f'[Fold {fold_i}] Finished iterating over fold table {fold_i} {shard_tbl_fp.name}.', logger=logger)
    
    tces_not_found = fold_tce_tbl.loc[~fold_tce_tbl[filter_field].isin(tces_found)]
    
    log_info(f'[Fold {fold_i}] Written {example_index}/{len(fold_tce_tbl)} TCEs. Missing {len(tces_not_found)}.', logger=logger)

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

    # tfrec_dir = Path('')
    # create_table_shard_example_location(tfrec_dir)
    
    cv_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval_normalized')
    save_fp = cv_dir / 'train_norm_stats_all_folds.csv'
    create_table_norm_stats_folds(cv_dir, save_fp)
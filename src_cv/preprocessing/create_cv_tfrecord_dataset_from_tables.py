"""
Create CV TFRecord shards based on CV TCE tables and source TFRecord shards.
"""

# 3rd party
import multiprocessing
from pathlib import Path
import pandas as pd
import logging
import tensorflow as tf

# local
from src_cv.preprocessing.utils import create_shard_fold, create_table_shard_example_location

if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # CV destination data directory
    dest_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406')
    # TFRecord source directory
    src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406')
    # table that maps a TCE to a given TFRecord file in the source TFRecords
    src_tfrec_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/shards_tbl.csv')
    n_processes = 36
    # directory with the TCE tables for each fold in the labeled data set
    labeled_shard_tbls_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/shard_tables/eval')  # dest_dir / 'shard_tables' / 'eval'
    # destination directory for the TFRecords for all CV folds in the labeled data set
    labeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'eval'
    # directory with the TCE tables for each fold for the unlabeled data set
    unlabeled_shard_tbls_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/shard_tables/predict')  # dest_dir / 'shard_tables' / 'predict'
    # destination directory for the TFRecords for all CV folds in the unlabeled data set
    unlabeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'predict'
    log_dir = dest_dir / 'create_cv_folds_logs'
    
    # string feature used to filter examples in source TFRecord dataset into fold shards (e.g., 'uid')
    filter_field = 'uid_obs_type'
    
    # batch size for source TFRecord
    src_tfrec_batch_size = 128
    
    # pattern used to look up unlabeled dataset tables
    unlabeled_fold_tbl_pattern = 'unlabeled_dataset_tbl_fold*.csv'
    
    # get CV folds tables
    labeled_shard_tbls_fps = sorted(list(labeled_shard_tbls_dir.glob('labeled_dataset_tbl_fold*.csv')))
    
    # get source TFRecord shards
    src_tfrec_fps = list(src_tfrec_dir.glob('shard-*'))

    # create directories
    dest_dir.mkdir(exist_ok=True)
    labeled_dest_tfrec_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dest_tfrec_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'create_cv_folds_shards')
    logger_handler = logging.FileHandler(filename=log_dir / f'create_cv_folds_shard_prediction.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    logger.info(f'Source TFRecords: {str(src_tfrec_dir)}')
    logger.info(f'Found {len(src_tfrec_fps)} TFRecord shards.')

    if not src_tfrec_tbl_fp.exists():
        logger.info('Creating shard example table that tracks location of examples in the source TFRecords...')
        create_table_shard_example_location(src_tfrec_dir)
    src_tfrec_tbl = pd.read_csv(src_tfrec_tbl_fp)
    src_tfrec_tbl.set_index('uid', inplace=True)

    n_processes_used = min(n_processes, len(labeled_shard_tbls_fps))
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, labeled_dest_tfrec_dir, fold_i, src_tfrec_fps, src_tfrec_batch_size, log_dir /
             f'create_fold_evaluation_{fold_i}.log', filter_field)
            for fold_i, shard_tbl_fp in enumerate(labeled_shard_tbls_fps)]
    logger.info(f'Set {len(jobs)} jobs to create TFRecord shards.')
    logger.info(f'Started creating shards for evaluation...')
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()
    pool.join()

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([async_result.get() for async_result in async_results], ignore_index=True)
    tces_not_found_df.to_csv(dest_dir / 'tces_not_found_eval.csv', index=False)

    logger.info('Finished creating TFRecords folds for evaluation.')

    # create TFRecord for examples that are not part of the evaluation set
    if unlabeled_shard_tbls_dir.exists():
        
        unlabeled_shard_tbls_fps = sorted(list(unlabeled_shard_tbls_dir.glob(unlabeled_fold_tbl_pattern)))

        logger.info('Creating shards for examples for prediction...')

        n_processes_used = min(n_processes, len(unlabeled_shard_tbls_fps))
        pool = multiprocessing.Pool(processes=n_processes_used)
        jobs = [(shard_tbl_fp, unlabeled_dest_tfrec_dir, fold_i, src_tfrec_fps, src_tfrec_batch_size, log_dir /
                 f'create_fold_predict_{fold_i}.log', filter_field)
                for fold_i, shard_tbl_fp in enumerate(unlabeled_shard_tbls_fps)]
        async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
        pool.close()
        pool.join()

        # TCEs present in the fold TCE tables that were not present in the source TFRecords
        tces_not_found_df = pd.concat([async_result.get() for async_result in async_results], ignore_index=True)
        tces_not_found_df.to_csv(dest_dir / 'tces_not_found_predict.csv', index=False)

        logger.info('Finished creating TFRecords shards for prediction.')

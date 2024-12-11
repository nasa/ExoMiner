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
from src_cv.utils_cv import create_shard_fold, create_table_shard_example_location

if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # CV destination data directory
    dest_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1347')
    # TFRecord source directory
    src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055/')
    # table that maps a TCE to a given TFRecord file in the source TFRecords
    src_tfrec_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055/shards_tbl.csv')
    n_processes = 128
    # directory with the TCE tables for each fold in the labeled data set
    labeled_shard_tbls_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1347/shard_tables/eval')  # dest_dir / 'shard_tables' / 'eval'
    # destination directory for the TFRecords for all CV folds in the labeled data set
    labeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'eval'
    # directory with the TCE tables for each fold for the unlabeled data set
    unlabeled_shard_tbls_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1347/shard_tables/predict')  # dest_dir / 'shard_tables' / 'predict'
    # destination directory for the TFRecords for all CV folds in the unlabeled data set
    unlabeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'predict'
    log_dir = dest_dir / 'create_cv_folds_logs'

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

    if not src_tfrec_tbl_fp.exists():
        logger.info('Creating shard example table that tracks location of examples in the source TFRecords...')
        create_table_shard_example_location(src_tfrec_dir)
    src_tfrec_tbl = pd.read_csv(src_tfrec_tbl_fp)

    labeled_shard_tbls_fps = sorted(list(labeled_shard_tbls_dir.iterdir()))

    n_processes_used = min(n_processes, len(labeled_shard_tbls_fps))
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, labeled_dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, log_dir /
             f'create_fold_evaluation_{fold_i}.log')
            for fold_i, shard_tbl_fp in enumerate(labeled_shard_tbls_fps)]
    logger.info(f'Set {len(jobs)} jobs to create TFRecord shards.')
    logger.info(f'Started creating shards for evaluation...')
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()
    pool.join()

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['uid'])
                                   for async_result in async_results], ignore_index=True)
    tces_not_found_df.to_csv(dest_dir / 'tces_not_found_eval.csv', index=False)

    logger.info('Finished creating TFRecords folds for evaluation.')

    # create TFRecord for examples that are not part of the evaluation set
    if unlabeled_shard_tbls_dir.exists():
        unlabeled_shard_tbls_fps = sorted(list(unlabeled_shard_tbls_dir.iterdir()))

        logger.info('Creating shards for examples for prediction...')

        # tce_tbl_noteval_fp = Path(dest_dir / f'examples_noteval.csv')
        # create_shard_fold(tce_tbl_noteval_fp, dest_tfrec_dir_noteval, -1, src_tfrec_dir, src_tfrec_tbl, True)

        n_processes_used = min(n_processes, len(unlabeled_shard_tbls_fps))
        pool = multiprocessing.Pool(processes=n_processes_used)
        jobs = [(shard_tbl_fp, unlabeled_dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, log_dir /
                 f'create_fold_predict_{fold_i}.log')
                for fold_i, shard_tbl_fp in enumerate(unlabeled_shard_tbls_fps)]
        async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
        pool.close()
        pool.join()

        # TCEs present in the fold TCE tables that were not present in the source TFRecords
        tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['uid'])
                                       for async_result in async_results], ignore_index=True)
        tces_not_found_df.to_csv(dest_dir / 'tces_not_found_predict.csv', index=False)

        logger.info('Finished creating TFRecords shards for prediction.')

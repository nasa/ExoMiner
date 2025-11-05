"""
Create CV TFRecord shards based on CV TCE tables and source TFRecord shards.
"""

# 3rd party
import multiprocessing
from pathlib import Path
import pandas as pd
import logging
import tensorflow as tf
import yaml

# local
from src_cv.preprocessing.utils import create_shard_fold, create_table_shard_example_location

if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')
    
    # load parameters from YAML file
    with open('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_cv/preprocessing/config_create_cv_tfrecord_dataset.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    dest_dir = Path(config['dest_dir'])
    labeled_shard_tbls_dir = Path(config['labeled_shard_tbls_dir'])
    src_tfrec_dir = Path(config['src_tfrec_dir'])
    src_tfrec_tbl_fp = Path(config['src_tfrec_tbl_fp'])
    n_processes = config['n_processes']
    n_shards_fold = config['n_shards_fold']
    # src_tfrec_batch_size = config['src_tfrec_batch_size']
    filter_field = config['filter_field']
    unlabeled_shard_tbls_dir = Path(config['unlabeled_shard_tbls_dir'])
    unlabeled_fold_tbl_pattern = config['unlabeled_fold_tbl_pattern']
    
    # destination directory for the TFRecords for all CV folds in the labeled data set
    labeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'eval'

    # destination directory for the TFRecords for all CV folds in the unlabeled data set
    unlabeled_dest_tfrec_dir = dest_dir / 'tfrecords' / 'predict'
    log_dir = dest_dir / 'create_cv_folds_logs'
    
    # get CV folds tables
    labeled_shard_tbls_fps = sorted(list(labeled_shard_tbls_dir.glob('labeled_dataset_tbl_fold*.csv')))
    
    # get source TFRecord shards
    src_tfrec_fps = list(src_tfrec_dir.glob('shard-*'))

    # create directories
    dest_dir.mkdir(exist_ok=True)
    labeled_dest_tfrec_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dest_tfrec_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # save yaml config file into destination directory
    with open(dest_dir / 'config_create_cv_tfrecord_dataset.yaml', 'w') as config_file:
        yaml.dump(config, config_file)

    # set up logger
    logger = logging.getLogger(name=f'create_cv_folds_shards')
    logger_handler = logging.FileHandler(filename=log_dir / f'create_cv_folds_shard.log', mode='w')
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

    # create jobs - one per fold
    jobs = [(shard_tbl_fp, labeled_dest_tfrec_dir, fold_i, src_tfrec_fps, n_shards_fold, log_dir /
             f'create_fold_evaluation_{fold_i}.log', filter_field)
            for fold_i, shard_tbl_fp in enumerate(labeled_shard_tbls_fps)]
    logger.info(f'Set {len(jobs)} jobs to create TFRecord shards.')
    
    logger.info(f'Started creating shards for evaluation...')
    # parallel using pool
    n_processes_used = min(n_processes, len(labeled_shard_tbls_fps))
    logger.info(f'Using {n_processes_used} processes to parallelize jobs...')
    with multiprocessing.Pool(processes=n_processes) as pool:
        async_results = [pool.apply_async(create_shard_fold, args=job) for job in jobs]
        jobs_results = [res.get() for res in async_results]

    # # sequential
    # jobs_results = []
    # for job in jobs:
    #     job_result = create_shard_fold(*job)
    #     jobs_results.append(job_result)
        
    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([job_result for job_result in jobs_results], ignore_index=True)
    tces_not_found_df.to_csv(dest_dir / 'tces_not_found_eval.csv', index=False)

    logger.info('Finished creating TFRecords folds for evaluation.')

    # create TFRecord for examples that are not part of the evaluation set
    if unlabeled_shard_tbls_dir.exists():
        
        unlabeled_shard_tbls_fps = sorted(list(unlabeled_shard_tbls_dir.glob(unlabeled_fold_tbl_pattern)))

        logger.info('Creating shards for examples for prediction...')

        jobs = [(shard_tbl_fp, unlabeled_dest_tfrec_dir, fold_i, src_tfrec_fps, 1, log_dir /
                        f'create_fold_predict_{fold_i}.log', filter_field)
                        for fold_i, shard_tbl_fp in enumerate(unlabeled_shard_tbls_fps)]
        
        n_processes_used = min(n_processes, len(unlabeled_shard_tbls_fps))
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            async_results = [pool.apply_async(create_shard_fold, args=job) for job in jobs]
            jobs_results = [res.get() for res in async_results]

        # TCEs present in the fold TCE tables that were not present in the source TFRecords
        tces_not_found_df = pd.concat([job_result for job_result in jobs_results], ignore_index=True)
        tces_not_found_df.to_csv(dest_dir / 'tces_not_found_predict.csv', index=False)

        logger.info('Finished creating TFRecords shards for prediction.')

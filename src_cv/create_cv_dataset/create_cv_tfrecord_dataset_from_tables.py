"""
Create CV TFRecord shards based on CV TCE tables and source TFRecord shards.
"""

# 3rd party
import multiprocessing
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# local
from src_cv.utils_cv import create_shard_fold, create_table_shard_example_location

if __name__ == '__main__':

    # CV data directory; contains the TCE tables for each fold
    data_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_obsplanets_siminj1_2-22-2024_1115/cv_keplerq1q17dr25_obsplanets_siminj1_data_02-22-2024_1155')
    # TFRecord source directory; non-normalized examples
    src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_obsplanets_siminj1_2-22-2024_1115/tfrecords_kepler_q1q17dr25obs_planets_sim_inj1_2-22-2024_1115')
    # table that maps a TCE to a given TFRecord file in the source TFRecords
    src_tfrec_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_obsplanets_siminj1_2-22-2024_1115/tfrecords_kepler_q1q17dr25obs_planets_sim_inj1_2-22-2024_1115/shards_tbl.csv')
    n_processes = 10

    data_dir.mkdir(exist_ok=True)

    # set up logger
    log_dir = data_dir / 'create_cv_folds_logs'
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(name=f'create_cv_folds_shards')
    logger_handler = logging.FileHandler(filename=log_dir / f'create_cv_folds_shards.log', mode='w')
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

    shard_tbls_dir = data_dir / 'shard_tables' / 'eval'  # directory with the TCE tables for each fold
    shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

    # destination directory for the TFRecords for all CV folds
    dest_tfrec_dir = data_dir / 'tfrecords' / 'eval'
    dest_tfrec_dir.mkdir(exist_ok=True, parents=True)

    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, True) for fold_i, shard_tbl_fp in
            enumerate(shard_tbls_fps)]
    logger.info(f'Set {len(jobs)} jobs to create TFRecord shards.')
    logger.info(f'Started creating shards...')
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()
    pool.join()
    logger.info(f'Finished creating shards.')

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['uid', 'target_id', 'tce_plnt_num'])
                                   for async_result in async_results], ignore_index=True)
    tces_not_found_df.to_csv(data_dir / 'tces_not_found_eval.csv', index=False)

    logger.info('Finished creating TFRecords folds for evaluation.')

    # create list of CV iterations
    data_shards_fps = list(dest_tfrec_dir.iterdir())
    data_shards = [data_shards_fp.name for data_shards_fp in data_shards_fps]

    # assert len(data_dir.iterdir()) == num_folds
    num_folds = len(data_shards_fps)

    # assert num_unit_folds <= num_folds
    folds_arr = np.arange(num_folds)

    # create TFRecord for examples that are not part of the evaluation set
    shard_tbls_dir = data_dir / 'shard_tables' / 'predict'  # directory with the TCE tables for each fold
    if shard_tbls_dir.exists():
        shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

        logger.info('Creating shards for examples not used for evaluation...')
        dest_tfrec_dir_noteval = data_dir / 'tfrecords' / 'predict'
        dest_tfrec_dir_noteval.mkdir(exist_ok=True)
        tce_tbl_noteval_fp = Path(data_dir / f'examples_noteval.csv')
        # create_shard_fold(tce_tbl_noteval_fp, dest_tfrec_dir_noteval, -1, src_tfrec_dir, src_tfrec_tbl, True)

        # destination directory for the predicted shards
        dest_tfrec_dir = data_dir / 'tfrecords' / 'predict'
        dest_tfrec_dir.mkdir(exist_ok=True, parents=True)

        pool = multiprocessing.Pool(processes=n_processes)
        jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, True) for fold_i, shard_tbl_fp in
                enumerate(shard_tbls_fps)]
        async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
        pool.close()

        # TCEs present in the fold TCE tables that were not present in the source TFRecords
        tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['uid', 'target_id', 'tce_plnt_num'])
                                       for async_result in async_results], ignore_index=True)
        tces_not_found_df.to_csv(data_dir / 'tces_not_found_predict.csv', index=False)

        logger.info('Finished creating TFRecords shards for prediction.')

        data_shards_fps = list(dest_tfrec_dir.iterdir())
        data_shards = [data_shards_fp.name for data_shards_fp in data_shards_fps]

        # assert len(data_dir.iterdir()) == num_folds
        num_folds = len(data_shards_fps)
        # assert num_unit_folds <= num_folds
        folds_arr = np.arange(num_folds)

        pred_shards = {'predict': data_shards}
        np.save(data_dir / f'predict_shards.npy', pred_shards)

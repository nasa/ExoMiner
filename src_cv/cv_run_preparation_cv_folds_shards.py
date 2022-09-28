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
    data_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/tess/cv_tess-s1s40-dv_all_features_phases_09-07-2022_1547')
    data_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'create_cv_folds_shards')
    logger_handler = logging.FileHandler(filename=data_dir / f'create_cv_folds_shards.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    # TFRecord source directory; non-normalized examples
    src_tfrec_dir = Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/tess/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_data/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_updtlablesrenamedfeats_toidv')
    logger.info(f'Source TFRecords: {str(src_tfrec_dir)}')

    # table that maps a TCE to a given TFRecord file in the source TFRecords
    # src_tfrec_tbl_fp = src_tfrec_dir / 'shards_tce_tbl.csv'
    src_tfrec_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/tess/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_data/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624/merged_shards.csv')
    if not src_tfrec_tbl_fp.exists():
        logger.info('Creating shard example table that tracks location of examples in the source TFRecords...')
        create_table_shard_example_location(src_tfrec_dir)
    src_tfrec_tbl = pd.read_csv(src_tfrec_tbl_fp)

    shard_tbls_dir = data_dir / 'shard_tables' / 'eval'  # directory with the TCE tables for each fold
    shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

    # destination directory for the TFRecords for all CV folds
    dest_tfrec_dir = data_dir / 'tfrecords' / 'eval'
    dest_tfrec_dir.mkdir(exist_ok=True, parents=True)

    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, True) for fold_i, shard_tbl_fp in
            enumerate(shard_tbls_fps)]
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['target_id', 'tce_plnt_num'])
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

    cv_folds_runs = [{'train': data_shards[:fold_i] + data_shards[fold_i + 1:],
                      'test': [data_shards[fold_i]]} for fold_i in folds_arr]

    np.save(data_dir / f'cv_folds_runs.npy', cv_folds_runs)

    cv_iterations = []
    for cv_iter in cv_folds_runs:
        cv_iterations.append([cv_iter['train'], cv_iter['test']])
    cv_iterations = pd.DataFrame(cv_iterations, columns=['train', 'test'])
    cv_iterations.to_csv(data_dir / f'cv_folds_runs.csv')

    # create TFRecord for examples that are not part of the evaluation set
    logger.info('Creating shards for examples not used for evaluation...')
    dest_tfrec_dir_noteval = data_dir / 'tfrecords' / 'predict'
    dest_tfrec_dir_noteval.mkdir(exist_ok=True)
    tce_tbl_noteval_fp = Path(data_dir / f'examples_noteval.csv')
    # create_shard_fold(tce_tbl_noteval_fp, dest_tfrec_dir_noteval, -1, src_tfrec_dir, src_tfrec_tbl, True)

    shard_tbls_dir = data_dir / 'shard_tables' / 'predict'  # directory with the TCE tables for each fold
    shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

    # destination directory for the predicted shards
    dest_tfrec_dir = data_dir / 'tfrecords' / 'predict'
    dest_tfrec_dir.mkdir(exist_ok=True, parents=True)

    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, True) for fold_i, shard_tbl_fp in
            enumerate(shard_tbls_fps)]
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['target_id', 'tce_plnt_num'])
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

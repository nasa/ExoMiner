"""
Create CV TFRecord shards based on CV TCE tables and source TFRecord shards.
"""

# 3rd party
import multiprocessing
from pathlib import Path
import pandas as pd
import numpy as np

# local
from src_cv.utils_cv import create_shard_fold

if __name__ == '__main__':

    # CV data directory; contains the TCE tables for each fold
    data_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_11-24-2021_1005')
    shard_tbls_dir = data_dir / 'shard_tables'  # directory with the TCE tables for each fold

    # TFRecord source directory; non-normalized examples
    tfrec_dir_root = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data/')
    src_tfrec_dir = tfrec_dir_root / \
                    'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_newval'

    # destination directory for the TFRecords for all CV folds
    dest_tfrec_dir = data_dir / 'tfrecords'
    dest_tfrec_dir.mkdir(exist_ok=True)

    # table that maps a TCE to a given TFRecord file in the source TFRecords
    src_tfrec_tbl = pd.read_csv(src_tfrec_dir / 'shards_tce_tbl.csv')
    shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, True) for fold_i, shard_tbl_fp in
            enumerate(shard_tbls_fps)]
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()

    print('Finished creating TFRecords folds.')

    # TCEs present in the fold TCE tables that were not present in the source TFRecords
    tces_not_found_df = pd.concat([pd.DataFrame(async_result.get(), columns=['target_id', 'tce_plnt_num'])
                                   for async_result in async_results], ignore_index=True)
    tces_not_found_df.to_csv(data_dir / 'tces_not_found.csv', index=False)

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

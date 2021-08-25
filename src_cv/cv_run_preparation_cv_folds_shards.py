
# 3rd party
import multiprocessing
from pathlib import Path

import pandas as pd

# local
from src_cv.utils_cv import create_shard_fold

if __name__ == '__main__':

    data_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_08-10-2021_06-47')
    shard_tbls_dir = data_dir / 'shard_tables'

    tfrec_dir_root = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                          'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar'
                          '-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/')

    src_tfrec_dir = tfrec_dir_root / \
                    'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-' \
                    'bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_' \
                    'experiment-labels-norm_nopps_secparams_prad_period'

    dest_tfrec_dir = data_dir / 'tfrecords'
    dest_tfrec_dir.mkdir(exist_ok=True)

    src_tfrec_tbl = pd.read_csv(src_tfrec_dir / 'shards_tce_tbl.csv')
    shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl) for fold_i, shard_tbl_fp in
            enumerate(shard_tbls_fps)]
    async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
    pool.close()

    tces_not_found_df = pd.DataFrame([async_result.get() for async_result in async_results],
                                     columns=['target_id', 'tce_plnt_num'])
    tces_not_found_df.to_csv(data_dir / 'tces_not_found.csv', index=False)

"""
Get start and end timestamps for each TIC observed in each sector run using multiprocessing.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import multiprocessing

# local
from src_preprocessing.tce_tables.ephemeris_matching.get_start_end_timestamps_sector_runs import \
    get_start_end_timestamps_tics_sector_runs

if __name__ == '__main__':

    # directory used to save start/end timestamps target tables for each sector run
    res_dir = Path('')
    res_dir.mkdir(exist_ok=True)
    # lightcurve root directory for the target data of interest from where to get the timestamps
    lc_root_dir = Path('')
    n_procs = 36  # number of parallel processes to spawn

    # 2min data
    sector_dirs_fps = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('sector_')]
    # ffi data
    # sector_dirs_fps = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('s')]

    print(f'Extracting start/end timestamps for targets in {len(sector_dirs_fps)} sector runs.')

    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [([sector_dir_fp], res_dir) for sector_dir_fp in sector_dirs_fps]
    async_results = [pool.apply_async(get_start_end_timestamps_tics_sector_runs, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        _ = async_result.get()

    print(f'Aggregating start/end timestamps target lc tables.')
    # aggregate tables into a single table
    target_sector_run_timestamps_all = \
        pd.concat([pd.read_csv(fp)
                   for fp in res_dir.iterdir()], axis=0).to_csv(res_dir / f'{res_dir.name}.csv', index=False)

    print('Finished.')

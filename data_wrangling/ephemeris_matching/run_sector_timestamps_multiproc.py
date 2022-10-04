"""
Get start and end timestamps for each TIC observed in each sector run using multiprocessing.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import multiprocessing

# local
from data_wrangling.ephemeris_matching.get_start_end_timestamps_sector_runs import \
    get_start_end_timestamps_tics_sector_runs

if __name__ == '__main__':

    root_dir = Path('/home/msaragoc/Projects/exoplnt_dl/experiments/ephemeris_matching_dv')
    lc_root_dir = Path('/data5/tess_project/Data/TESS_lc_fits')
    # lc_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/lc/')
    sector_dirs = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('sector_') and fp.name.split('_')[-1] in ['10', '12', '15', '18', '52', '53', '54', '55']]
    save_dir = root_dir / 'start_end_timestamps_tics_lc'
    save_dir.mkdir(exist_ok=True)

    n_procs = 10
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [([sector_dir], save_dir) for sector_dir in sector_dirs]
    async_results = [pool.apply_async(get_start_end_timestamps_tics_sector_runs, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        _ = async_result.get()

    target_sector_run_timestamps_all = \
        pd.concat([pd.read_csv(fp)
                   for fp in save_dir.iterdir()], axis=0).to_csv(root_dir /
                                                                 'all_sectors_times_btjd_start_end.csv', index=False)

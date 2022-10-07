"""
Match transit signals using correlation coefficient tables.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import multiprocessing

# local
from data_wrangling.ephemeris_matching.resolve_matchings import solve_matches

if __name__ == '__main__':

    match_thr = 0.75
    matching_root_dir = Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/10-05-2022_1621')
    match_dir = matching_root_dir / 'sector_run_tic_tbls'
    matched_signals = []
    for tbl_fp in match_dir.iterdir():
        matched_signals.append(solve_matches(tbl_fp, match_thr))

    n_procs = 4
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(match_tbl_fp, match_thr) for match_tbl_fp in match_dir.iterdir()]
    async_results = [pool.apply_async(solve_matches, job) for job in jobs]
    pool.close()

    matched_signals = pd.concat([async_result.get() for async_result in async_results], axis=0)
    matched_signals.to_csv(matching_root_dir / 'matched_signals.csv', index=False)

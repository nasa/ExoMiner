"""
Wrapper to resolve_matchings.py that parallelizes processing.

Match transit signals using correlation coefficient tables.

A match is accepted if 1) the correlation score is higher than the defined threshold and 2) it is the highest score out
of all matches done for a given transit signal.

Each match table should be such that the first column is the ID of the transit signals that are to be matched to a
single transit signal from the set of transit signals defined in the columns after the ID column of the row transit
signal. Each match table should only contain transit signals that are related to the same target star.

e.g.

|         uid            | col_transit_signal_1 | col_transit_signal_2  |
| row_transit_signal_1   |     corr_row1_col1   |     corr_row1_col2    |
| row_transit_signal_2   |     corr_row2_col1   |    corr_row2_col2     |

where corr_rowX_colY refers to the correlation score between transit signal in row X and transit signal in row Y. Match
is accepted between these two if corr_rowX_colY > matching threshold AND max(corr_rowX_coli) = corr_rowX_colY
"""

# 3rd party
import pandas as pd
from pathlib import Path
import multiprocessing

# local
from src_preprocessing.tce_tables.ephemeris_matching.resolve_matchings import solve_matches

if __name__ == '__main__':

    match_thr = 0.75  # set matching threshold
    n_procs = 36  # number of parallel processes
    # directory with ephemeris matching results
    matching_root_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/tces_spoc_ffi_s36-s72_prsaebs_4-17-2025_2110')

    print(f'Resolving matches for {str(matching_root_dir)}')

    print(f'Set matching threshold to {match_thr}.')

    match_dir = matching_root_dir / 'sector_run_tic_tbls'

    # number of jobs; one per match table
    jobs = [(match_tbl_fp, match_thr) for match_tbl_fp in match_dir.iterdir()]

    # parallel processing
    pool = multiprocessing.Pool(processes=n_procs)
    async_results = [pool.apply_async(solve_matches, job) for job in jobs]
    pool.close()
    pool.join()
    # aggregate match results into a single file
    matched_signals = pd.concat([async_result.get() for async_result in async_results], axis=0)

    # # sequential processing
    # matched_signals = []
    # for tbl_fp in jobs:
    #     matched_signals.append(solve_matches(tbl_fp, match_thr))

    print(f'Found {len(matched_signals)} signals with a match.')

    matched_signals.to_csv(matching_root_dir / f'matched_signals_thr{match_thr}.csv', index=False)

    print('Finished resolving matches.')

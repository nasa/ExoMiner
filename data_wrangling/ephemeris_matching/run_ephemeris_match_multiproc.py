"""
Run ephemeris matching using multiprocessing.

Parameters:
    - sector_timestamps_tbl_fp: the file path to the csv file containing the start and end timestamps for each TIC in
    each sector run of interest to build the binary time series used for computing match scores.
    - plot_prob: probability of plotting the binary time series plots for a matching pair.
    - sampling interval: sampling rate used to generate the binary time series.
    - tce_tbl_fp: the file path to a table of transit signals that need to be matched to signals in `toi_tbl_fp`.
    - toi_tbl_fp: the file path to a table of transit signals to be matched in a one-to-many fashion to the signals in
    `tce_tbl_fp`.
    - n_procs: number of parallel processes to spawn for parallelization.
    - n_jobs:  number of jobs spread through the `n_procs`.

`tce_tbl` and `toi_tbl` must contain the following columns:
- tce_tbl: uid, duration (hours), period (days), epoch (days), target_id
- toi_tbl: uid, duration (hours), period (days), epoch (days), target_id
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing

# local
from data_wrangling.ephemeris_matching.ephemeris_matching import match_transit_signals_in_target

if __name__ == '__main__':

    # root directory for the ephemeris matching experiment
    root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/')
    # results directory
    exp_dir = root_dir / f'tces_spoc_dv_2mindata_s1-s68_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    # table with start and end timestamps per TIC for different sector runs
    sector_timestamps_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/all_sectors_times_btjd_start_end.csv')
    # plot probability
    plot_prob = 0.1
    sampling_interval = 2 / 60 / 24  # sampling rate for binary time series
    # table of signals of interest
    tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_09-25-2023_1608.csv')
    # table of signals to be matched
    toi_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/9-19-2023/exofop_tess_tois.csv')
    n_procs = 10  # no processes (for parallelization)
    n_jobs = 10  # no jobs

    # create experiment directory
    exp_dir.mkdir(exist_ok=True)
    print(f'Starting run {exp_dir}...')
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)

    print(f'Plot probability: {plot_prob}')
    print(f'Sampling interval for binary time series: {sampling_interval}')

    # load object catalog to be matched against TCEs
    toi_tbl = pd.read_csv(toi_tbl_fp, header=1)
    print(f'Using objects\' table {toi_tbl_fp}')

    # load TCE table
    tce_tbl = pd.read_csv(tce_tbl_fp)
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)
    tce_tbl = tce_tbl.dropna(subset=['period', 'epoch', 'duration'])
    tce_tbl['sector_run'] = tce_tbl['sector_run'].astype('str')

    # load table with start and end timestamps for each sector run for the TICs associated with the tCEs in the TCE
    # table
    sector_timestamps_tbl = pd.read_csv(sector_timestamps_tbl_fp).sort_values('sector')
    print(f'Using sector timestamps table {sector_timestamps_tbl_fp}')

    targets_arr = tce_tbl['target_id'].unique()
    print(f'Number of targets to be iterated through: {len(targets_arr)}')

    # match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
    #                                 plot_prob, plot_dir)

    print(f'Using {n_procs} processes to run {n_jobs} jobs...')
    pool = multiprocessing.Pool(processes=n_procs)
    targets_arr_jobs = [(targets_arr_job, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
                         plot_prob, plot_dir) for targets_arr_job in np.array_split(targets_arr, n_jobs)]
    async_results = [pool.apply_async(match_transit_signals_in_target, targets_arr_job)
                     for targets_arr_job in targets_arr_jobs]
    pool.close()
    pool.join()

    print('Finished ephemeris matching.')

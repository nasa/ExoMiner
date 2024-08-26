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
import multiprocessing
import argparse
import yaml

# local
from src_preprocessing.ephemeris_matching.ephemeris_matching import match_transit_signals_in_target

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default='./src_preprocessing/ephemeris_matching/config_ephem_matching.yaml')
    args = parser.parse_args()

    with(open(Path(args.config_fp).resolve(), 'r')) as file:
        config = yaml.safe_load(file)

    # results directory
    if args.output_dir is not None:
        config['exp_dir'] = args.output_dir

    exp_dir = Path(config["exp_dir"])

    # create experiment directory
    exp_dir.mkdir(exist_ok=True)
    print(f'Starting run {exp_dir}...')
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)

    # save yaml config file
    with open(exp_dir / 'config_run.yaml', 'w') as run_file:
        yaml.dump(config, run_file, sort_keys=False)

    print(f'Plot probability: {config["plot_prob"]}')
    print(f'Sampling interval for binary time series: {config["sampling_interval"]}')

    # load table of signals of interest (usually TCEs)
    tce_tbl = pd.read_csv(config['tbl_a_fp'])
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)
    tce_tbl = tce_tbl.dropna(subset=['period', 'epoch', 'duration'])
    tce_tbl['sector_run'] = tce_tbl['sector_run'].astype('str')
    print(f'Using table of signals to be matched against: {config["tbl_a_fp"]}')
    print(f'Table with {len(tce_tbl)} signals.')

    # load of signals to be matched to those in table of signals of interest (usually objects with dispositions)
    toi_tbl = pd.read_csv(config['tbl_b_fp'])
    print(f'Using objects\' table: {config["tbl_b_fp"]}')
    print(f'Table with {len(toi_tbl)} signals.')

    # load table with start and end timestamps for each sector run for the TICs associated with the tCEs in the TCE
    # table
    sector_timestamps_tbl = pd.read_csv(config["sector_timestamps_tbl_fp"]).sort_values('sector')
    print(f'Using sector timestamps table {config["sector_timestamps_tbl_fp"]}')

    targets_arr = toi_tbl['target_id'].unique()
    print(f'Number of targets to be iterated through: {len(targets_arr)}')

    print(f'Using {config["n_procs"]} processes to run {config["n_jobs"]} jobs...')
    pool = multiprocessing.Pool(processes=config["n_procs"])
    targets_arr_jobs = [(targets_arr_job, tce_tbl, toi_tbl, sector_timestamps_tbl, config["sampling_interval"],
                         save_dir, config["plot_prob"], plot_dir)
                        for job_i, targets_arr_job in enumerate(np.array_split(targets_arr, config["n_jobs"]))]
    async_results = [pool.apply_async(match_transit_signals_in_target, targets_arr_job)
                     for targets_arr_job in targets_arr_jobs]
    pool.close()
    pool.join()

    print('Finished ephemeris matching.')

"""
Run ephemeris matching using multiprocessing.

Parameters:
    - sectors_timestamps_tbl_fp: the file path to the csv file containing the start and end timestamps for each sector in 
    to build the binary time series used for computing match scores.
    - plot_prob: probability of plotting the binary time series plots for a matching pair.
    - sampling interval: sampling rate used to generate the binary time series.
    - tce_tbl_fp: the file path to a table of transit signals that need to be matched to signals in `toi_tbl_fp`.
    - toi_tbl_fp: the file path to a table of transit signals to be matched in a one-to-many fashion to the signals in
    `tce_tbl_fp`.
    - n_procs: number of parallel processes to spawn for parallelization.
    - n_jobs:  number of jobs spread through the `n_procs`.

`tce_tbl` and `toi_tbl` must contain the following columns:
- tce_tbl: uid, duration (hours), period (days), epoch (days), target_id, sector_run, sectors_observed
- toi_tbl: uid, duration (hours), period (days), epoch (days), target_id

`sectors_timestamps_tbl` must contain the following columns: sector, start_btjd, end_btjd
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing
import argparse
import yaml
import logging
from tqdm import tqdm

# local
from src_preprocessing.tce_tables.ephemeris_matching.ephemeris_matching import match_transit_signals_in_target

logger = logging.getLogger(__name__)


def get_params(config_fp, output_dir=None):
    
    with(open(Path(config_fp).resolve(), 'r')) as file:
        config = yaml.safe_load(file)
    
    # overwrite results directory
    if output_dir is not None:
        config['exp_dir'] = output_dir

    exp_dir = Path(config["exp_dir"])
    
    # save yaml config file
    with open(exp_dir / 'config_run.yaml', 'w') as run_file:
        yaml.dump(config, run_file, sort_keys=False)
    
    tbl_a_fp = config['tbl_a_fp']
    tbl_b_fp = config['tbl_b_fp']
    sectors_ts_tbl_fp = config['sectors_timestamps_tbl_fp']
    sampling_int = config['sampling_interval']
    plot_prob = config['plot_prob']
    n_procs = config['n_procs']
    n_jobs = config['n_jobs']

    return exp_dir, tbl_a_fp, tbl_b_fp, sectors_ts_tbl_fp, sampling_int, plot_prob, n_procs, n_jobs
    
def check_tables(tbl_a, tbl_b, timestamps_tbl):
    
    required_cols = ['uid', 'epoch', 'period', 'duration', 'target_id']
    required_cols_a_only = ['sector_run']
    required_cols_timestamps = ['sector', 'start_btjd', 'end_btjd']
    
    missing_cols_a = [col for col in required_cols + required_cols_a_only if col not in tbl_a]
    missing_cols_b = [col for col in required_cols if col not in tbl_b]
    missing_cols_timestamps = [col for col in required_cols_timestamps if col not in timestamps_tbl]
    
    if len(missing_cols_a):
        raise ValueError(f'Table a is missing required columns: {missing_cols_a}')

    if len(missing_cols_b):
        raise ValueError(f'Table b is missing required columns: {missing_cols_b}')

    if len(missing_cols_timestamps):
        raise ValueError(f'Table with sector timestamps is missing required columns: {missing_cols_timestamps}')

    tbl_a = tbl_a.dropna(subset=['period', 'epoch', 'duration'])
    tbl_b = tbl_b.dropna(subset=['period', 'epoch', 'duration'])
    
    tbl_a['sector_run'] = tbl_a['sector_run'].astype('str')
    
    # timestamps_tbl['sectors_observed'] = timestamps_tbl['sectors_observed'].astype('str')
    
    return tbl_a, tbl_b

    
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default='./config_ephem_matching.yaml')
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
    
    exp_dir, tbl_a_fp, tbl_b_fp, sectors_ts_tbl_fp, sampling_int, plot_prob, n_procs, n_jobs = get_params(args.config_fp, args.output_dir)
    
    # create experiment directory
    exp_dir.mkdir(exist_ok=True)
    print(f'Starting run {exp_dir}...')
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # create logger
    logging.basicConfig(filename=log_dir / f'ephem_matching_main.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a',
                        )

    logger.info(f'Plot probability: {plot_prob}')
    logger.info(f'Sampling interval for binary time series: {sampling_int}')

    # load table of signals of interest (usually TCEs)
    tce_tbl = pd.read_csv(tbl_a_fp, dtype={'sector_run': str, 'sectors_observed': str})
    
    # load table of signals to be matched to those in table of signals of interest (usually objects with dispositions)
    obj_tbl = pd.read_csv(tbl_b_fp)
    
    # # load table with start and end timestamps for each sector run for the TICs associated with the tCEs in the TCE
    # # table
    # sector_timestamps_tbl = pd.read_csv(sector_ts_tbl_fp).sort_values('sector')
    # load table with start and end timestamps for each sector
    sectors_timestamps_tbl = pd.read_csv(sectors_ts_tbl_fp)

    check_tables(tce_tbl, obj_tbl, sectors_timestamps_tbl)
    
    logger.info(f'Using table of signals to be matched against: {tbl_a_fp}')
    logger.info(f'Table with {len(tce_tbl)} signals.')
    
    logger.info(f'Using objects\' table: {tbl_b_fp}')
    logger.info(f'Table with {len(obj_tbl)} signals.')
    
    logger.info(f'Using sector timestamps table {sectors_ts_tbl_fp}')

    targets_arr = obj_tbl.loc[obj_tbl['target_id'].isin(tce_tbl['target_id']), 'target_id'].unique()
    logger.info(f'Number of targets to be iterated through: {len(targets_arr)}')

    logger.info(f'Using {n_procs} processes to run {n_jobs} jobs...')
    targets_arr_jobs = [(targets_arr_job, tce_tbl, obj_tbl, sectors_timestamps_tbl, sampling_int,
                         save_dir, plot_prob, plot_dir, job_i)
                        for job_i, targets_arr_job in enumerate(np.array_split(targets_arr, n_jobs))]
    
    with multiprocessing.Pool(processes=n_procs) as pool, tqdm(total=len(targets_arr_jobs), desc='Ephemeris matching', unit='job') as pbar:
        
        async_results = []
        for targets_arr_job in targets_arr_jobs:
            job_id = targets_arr_job[-1]
            ar = pool.apply_async(
                match_transit_signals_in_target, 
                targets_arr_job,
                callback=lambda _, j=job_id: (pbar.update(1), logger.info(f'Finished job {j}')),
                # error_callback=lambda e, j=job_id: logger.error(f'Job {j} failed with error: {e}')
                ) 
            async_results.append(ar)
        
        # propagate exceptions and dealing with them if needed
        for r in async_results:
            _ = r.get()

    logger.info('Finished ephemeris matching.')
    

if __name__ == '__main__':

    main()

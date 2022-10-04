"""
Run ephemeris matching using multiprocessing.
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
    root_dir = Path('/home/msaragoc/Projects/exoplnt_dl/experiments/ephemeris_matching_dv/')
    # root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/')

    # load table with start and end timestamps for each sector run
    sector_timestamps_tbl = pd.read_csv(root_dir / 'all_sectors_times_btjd_start_end.csv').sort_values('sector')
    # load TOI catalog
    toi_tbl = pd.read_csv(root_dir / 'exofop_tess_tois-4.csv')
    # toi_tbl['Epoch (BTJD)'] = toi_tbl['Epoch (BJD)'] - 2457000
    toi_tbl['Epoch (BTJD)'] = toi_tbl['Transit Epoch (BJD)'] - 2457000
    toi_tbl.rename(columns={'Epoch (BTJD)': 'epoch', 'Period (days)': 'period', 'Duration (hours)': 'duration', 'TOI': 'uid'}, inplace=True)
    toi_tbl = toi_tbl.dropna(subset=['period', 'epoch', 'duration'])

    # load TCE table
    tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_final.csv')
    # tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_smet.csv')
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)
    tce_tbl = tce_tbl.dropna(subset=['period', 'epoch', 'duration'])

    # create experiment directory
    exp_dir = root_dir / f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    exp_dir.mkdir(exist_ok=True)
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)

    plot_prob = 0.1
    sampling_interval = 2 / 60 / 24  # sampling rate for binary time series

    targets_arr = tce_tbl['target_id'].unique()

    match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
                                    plot_prob, plot_dir)

    n_procs = 10
    n_jobs = 10
    pool = multiprocessing.Pool(processes=n_procs)
    targets_arr_jobs = [(targets_arr_job, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
                         plot_prob, plot_dir) for targets_arr_job in np.array_split(targets_arr, n_jobs)]
    async_results = [pool.apply_async(match_transit_signals_in_target, targets_arr_job)
                     for targets_arr_job in targets_arr_jobs]
    pool.close()

    print('Finished ephemeris matching.')

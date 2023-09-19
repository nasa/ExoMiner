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
    root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_spoc_ffi_dv/')
    # root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/')

    # create experiment directory
    exp_dir = root_dir / f'tces_spoc_dv_ffidata_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    exp_dir.mkdir(exist_ok=True)
    print(f'Starting run {exp_dir}...')
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)

    plot_prob = 0.1
    print(f'Plot probability: {plot_prob}')
    sampling_interval = 2 / 60 / 24  # sampling rate for binary time series
    print(f'Sampling interval for binary time series: {sampling_interval}')

    # load table with start and end timestamps for each sector run
    sector_timestamps_tbl_fp = root_dir / 'all_sectors_times_btjd_start_end.csv'
    sector_timestamps_tbl = pd.read_csv(sector_timestamps_tbl_fp).sort_values('sector')
    print(f'Using sector timestamps table {sector_timestamps_tbl_fp}')

    # load TOI catalog
    toi_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/9-19-2023/exofop_tess_tois.csv')
    toi_tbl = pd.read_csv(toi_tbl_fp)
    # toi_tbl['Epoch (BTJD)'] = toi_tbl['Epoch (BJD)'] - 2457000
    toi_tbl['Epoch (BTJD)'] = toi_tbl['Epoch (BJD)'] - 2457000
    toi_tbl.rename(columns={'Epoch (BTJD)': 'epoch', 'Period (days)': 'period', 'Duration (hours)': 'duration', 'TOI': 'uid'}, inplace=True)
    toi_tbl = toi_tbl.dropna(subset=['period', 'epoch', 'duration'])
    toi_tbl = toi_tbl.loc[(~toi_tbl['period'].isna() & (toi_tbl['period'] > 0) & ~toi_tbl['epoch'].isna() & ~toi_tbl['duration'].isna())]

    # toi_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv')
    # toi_tbl = pd.read_csv(toi_tbl_fp)
    toi_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration', 'target_id': 'TIC ID'}, inplace=True)
    print(f'Using TOI table {toi_tbl_fp}')

    # load TCE table
    tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/dv_spoc_ffi/04-11-2023_1623/tess_spoc_ffi_tces_dv_s47-s55_04-11-2023_1623_preproc.csv')
    tce_tbl = pd.read_csv(tce_tbl_fp)
    # tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_smet.csv')
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)
    tce_tbl = tce_tbl.dropna(subset=['period', 'epoch', 'duration'])
    tce_tbl['sector_run'] = tce_tbl['sector_run'].astype('str')

    targets_arr = tce_tbl['target_id'].unique()
    print(f'Number of targets to be iterated through: {len(targets_arr)}')

    # match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
    #                                 plot_prob, plot_dir)

    n_procs = 10
    n_jobs = 10
    print(f'Using {n_procs} processes to run {n_jobs} jobs...')
    pool = multiprocessing.Pool(processes=n_procs)
    targets_arr_jobs = [(targets_arr_job, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
                         plot_prob, plot_dir) for targets_arr_job in np.array_split(targets_arr, n_jobs)]
    async_results = [pool.apply_async(match_transit_signals_in_target, targets_arr_job)
                     for targets_arr_job in targets_arr_jobs]
    pool.close()

    print('Finished ephemeris matching.')

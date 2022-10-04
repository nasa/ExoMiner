"""
Match pairs of transit signals based on the correlation between the in-transit binary time series create using their
ephemerides (orbital period, epoch, and transit duration).
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# local
from data_wrangling.ephemeris_matching.utils_ephemeris_matching import create_binary_time_series, \
    find_first_epoch_after_this_time


def compute_correlation_coeff(bin_ts_a, bin_ts_b):
    """ Compute correlation coefficient between two 1D arrays. The correlation coefficient is computed as
    rho = u.v / (||u||*||v||)

    Args:
        bin_ts_a: 1D NumPy array
        bin_ts_b: 1D NumPy array

    Returns: float, correlation coefficient between the two 1D arrays.

    """

    return np.sum(bin_ts_a * bin_ts_b) / (np.linalg.norm(bin_ts_a) * np.linalg.norm(bin_ts_b))


def match_transit_signals(transit_signal_a, transit_signal_b, sampling_interval, tstart, tend, plot_bin_ts=False,
                          plot_dir=None):
    """ Compute matching agreement between two transit signals by creating binary time series that are 1 for in-transit
    cadences and zero otherwise.
    The binary time series are created using the orbital period, epoch, and transit duration estimated for the transit
    signals. A correlation coefficient is computed between the two binary time series to quantify the match between the
    two transit signals. The pandas Series for the transit signals must contain ephemerides information
    ('epoch' in days, 'duration' in hours, 'period' in days), and the 'uid' field.

    Args:
        transit_signal_a: pandas Series, transit signal a
        transit_signal_b: pandas Series, transit signal b
        sampling_interval: float, sampling interval used to create the binary time series
        tstart: float, start time for binary time series
        tend: float, end time for binary time series
        plot_bin_ts: bool, if True plots binary time series for the two transit signals
        plot_dir: Path, directory in which to save the plots

    Returns: float, correlation coefficient between the two transit signals

    """

    transit_signal_a_bin_ts = create_binary_time_series(
        epoch=find_first_epoch_after_this_time(transit_signal_a['epoch'], transit_signal_a['period'], tstart),
        duration=transit_signal_a['duration'] / 24,
        period=transit_signal_a['period'],
        tStart=tstart,
        tEnd=tend,
        samplingInterval=sampling_interval
    )

    epoch_new = find_first_epoch_after_this_time(transit_signal_b['epoch'], transit_signal_b['period'], tstart)
    transit_signal_b_bin_ts = create_binary_time_series(
        epoch=epoch_new,
        duration=transit_signal_b['duration'] / 24,
        period=transit_signal_b['period'],
        tStart=tstart,
        tEnd=tend,
        samplingInterval=sampling_interval
    )

    corr_coeff = compute_correlation_coeff(transit_signal_a_bin_ts, transit_signal_b_bin_ts)

    if plot_bin_ts:

        time_arr = np.linspace(tstart, tend, int((tend - tstart) / sampling_interval), endpoint=True)

        f, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_arr, transit_signal_a_bin_ts, 'b', label='Signal A', zorder=1)
        ax.plot(time_arr, transit_signal_b_bin_ts, 'r--', label='Signal B', zorder=2, alpha=0.8)
        ax.legend()
        ax.set_title(f'Signal A: {transit_signal_a["uid"]} p={transit_signal_a["period"]:.4f},e={transit_signal_a["epoch"]:.4f}, d={transit_signal_a["duration"]:.4f}'
                     f'\nSignal B: {transit_signal_b["uid"]} p={transit_signal_b["period"]:.4f},e={transit_signal_b["epoch"]:.4f}, d={transit_signal_b["duration"]:.4f}, e_shift={epoch_new}'
                     f'\nCorrelation Coefficient: {corr_coeff:.4f}')
        ax.set_xlabel('Timestamps [BTJD]')
        ax.set_ylabel('In-transit Flag')
        ax.set_xlim([time_arr[0], time_arr[-1]])
        f.savefig(plot_dir / f'bin_ts_{transit_signal_a["uid"]}-{transit_signal_b["uid"]}.png')
        plt.close()

    return corr_coeff


def match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval,
                                    save_dir, plot_prob=0, plot_dir=None):
    """ Compute matching correlation coefficient between TCEs and TOIs for each TIC in each sector run.

    Args:
        targets_arr: pandas Series, transit signal a
        tce_tbl: pandas DataFrame, TCE table
        toi_tbl: pandas DataFrame, TOI catalog
        sector_timestamps_tbl: pandas DataFrame, table with start and end timestamps for each sector run
        sampling_interval: float, sampling interval used to create the binary time series
        save_dir: Path, directory used to save the matching tables
        plot_prob: float, probability to create plot with both binary time series
        plot_dir: Path, directory in which to save the plots

    Returns:

    """

    for target in targets_arr:

        print(f'Iterating over TIC {target}...')

        # get start and end timestamps for this TIC
        tic_timestamps_sector = sector_timestamps_tbl.loc[sector_timestamps_tbl['target'] == target]
        if len(tic_timestamps_sector) == 0:
            print(f'TIC {target} not found in the timestamps table.')
            continue

        # get TOIs in this TIC
        tois_in_tic = toi_tbl.loc[toi_tbl['TIC ID'] == target].reset_index()
        if len(tois_in_tic) == 0:
            print(f'No TOIs in TIC {target}.')
            continue

        for sector_run in tce_tbl.loc[tce_tbl['target_id'] == target, 'sector_run'].unique():
            print(f'Iterating over sector run {sector_run} for TIC {target}')

            # get TCEs in this TIC and sector run
            tces_in_tic_sectorun = tce_tbl.loc[(tce_tbl['target_id'] == target) &
                                               (tce_tbl['sector_run'] == sector_run)].reset_index()
            if len(tces_in_tic_sectorun) == 0:
                print(f'No TCEs in TIC {target} for sector run {sector_run}.')
                continue

            # get start and end timestamps for the sector run
            if '-' in sector_run:
                s_sector, e_sector = [int(s) for s in sector_run.split('-')]
                sector_flag = (tic_timestamps_sector['sector'] >= s_sector) & \
                              (tic_timestamps_sector['sector'] <= e_sector)
                if sector_flag.sum() == 0:
                    print(f'No start and end timestamps available for TIC {target} in sector run {sector_run}')
                    continue

                tstart = tic_timestamps_sector.loc[sector_flag, 'start'].values[0]
                tend = tic_timestamps_sector.loc[sector_flag, 'end'].values[-1]
            else:
                sector = int(sector_run)
                if (tic_timestamps_sector['sector'] == sector).sum() == 0:
                    print(f'No start and end timestamps available for TIC {target} in sector run {sector_run}')
                    continue
                tstart = tic_timestamps_sector.loc[tic_timestamps_sector['sector'] == sector, 'start'].values[0]
                tend = tic_timestamps_sector.loc[tic_timestamps_sector['sector'] == sector, 'end'].values[0]

            # initialize correlation coefficient matrix to test matching between TCEs and TOIs
            corr_coef_mat = np.nan * np.ones((len(tces_in_tic_sectorun), len(tois_in_tic)))
            plot_signals = True if np.random.uniform() < plot_prob else False
            # compute correlation coefficient
            for tce_i, tce in tces_in_tic_sectorun.iterrows():
                for toi_i, toi in tois_in_tic.iterrows():
                    try:
                        corr_coef_mat[tce_i, toi_i] = \
                            match_transit_signals(tce, toi, sampling_interval, tstart, tend,
                                                  plot_signals,
                                                  plot_dir)
                    except:
                        aaaa

            tic_match_tbl = pd.DataFrame(corr_coef_mat, index=tces_in_tic_sectorun['uid'], columns=tois_in_tic['uid'])

            tic_match_tbl.to_csv(save_dir / f'match_tbl_s{sector_run}_tic_{target}.csv')


if __name__ == '__main__':

    # root directory for the ephemeris matching experiment
    root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/')

    # load table with start and end timestamps for each sector run
    # sector_timestamps_tbl = pd.read_csv(root_dir / 'sector_times_btjd.csv')
    sector_timestamps_tbl = pd.read_csv(root_dir / 'all_sectors_times_btjd_start_end.csv')
    # load TOI catalog
    toi_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/exofop_toi_catalog.csv')
    toi_tbl['Epoch (BTJD)'] = toi_tbl['Epoch (BJD)'] - 2457000
    toi_tbl.rename(columns={'Epoch (BTJD)': 'epoch', 'Period (days)': 'period', 'Duration (hours)': 'duration', 'TOI': 'uid'}, inplace=True)
    # load TCE table
    tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)

    # create experiment directory
    exp_dir = root_dir / f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    exp_dir.mkdir(exist_ok=True)
    save_dir = exp_dir / 'sector_run_tic_tbls'
    save_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / 'bin_ts_plots'
    plot_dir.mkdir(exist_ok=True)

    plot_prob = 1  # 0.01
    sampling_interval = 2 / 60 / 24  # sampling rate for binary time series

    targets_arr = sector_timestamps_tbl['target']  # tce_tbl.loc[~tce_tbl['TOI'].isna(), 'target_id'][:5]  # tce_tbl.loc[(tce_tbl['target_id'].isin(sector_timestamps_tbl['target'])), 'target_id'].unique()  # & (tce_tbl['sector_run'].isin(sector_timestamps_tbl['sector'])
    match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, sampling_interval, save_dir,
                                    plot_prob=plot_prob, plot_dir=plot_dir)


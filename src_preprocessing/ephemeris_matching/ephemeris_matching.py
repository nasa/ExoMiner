"""
Match pairs of transit signals based on the correlation between the in-transit binary time series created using their
ephemerides (orbital period, epoch, and transit duration).

The signals in the two tables should have the columns `period`, `duration`, and `epoch`, for period (in days),
duration (in hours), and epoch (in days and same frame, e.g., TBJD), respectively. They should also have a column
`target_id` for the id of the target star they are associated with.
"""

# 3rd party
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
import logging

# local
from src_preprocessing.ephemeris_matching.utils_ephemeris_matching import create_binary_time_series, \
    find_first_epoch_after_this_time

logger = logging.getLogger(__name__)


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
        ax.set_title(f'Signal A: {transit_signal_a["uid"]} p={transit_signal_a["period"]:.4f},' 
                     f'e={transit_signal_a["epoch"]:.4f}, d={transit_signal_a["duration"]:.4f}'
                     f'\nSignal B: {transit_signal_b["uid"]} p={transit_signal_b["period"]:.4f},' 
                     f'e={transit_signal_b["epoch"]:.4f}, d={transit_signal_b["duration"]:.4f}, e_shift={epoch_new}'
                     f'\nCorrelation Coefficient: {corr_coeff:.4f}')
        ax.set_xlabel('Timestamps [BTJD]')
        ax.set_ylabel('In-transit Flag')
        ax.set_xlim([time_arr[0], time_arr[-1]])
        f.savefig(plot_dir / f'bin_ts_{transit_signal_a["uid"]}-{transit_signal_b["uid"]}.png')
        plt.close()

    return corr_coeff


def match_transit_signals_in_target(targets_arr, tce_tbl, objects_tbl, sector_timestamps_tbl, sampling_interval,
                                    save_dir, plot_prob=0, plot_dir=None):
    """ Compute matching correlation coefficient between signals/objects for each TIC in each sector run.

    Args:
        targets_arr: NumPy arrau, list of target stars for whose transit signals ephemerides matching is going to be run
        tce_tbl: pandas DataFrame, TCE table
        objects_tbl: pandas DataFrame, objects' catalog
        sector_timestamps_tbl: pandas DataFrame, table with start and end timestamps for each sector run
        sampling_interval: float, sampling interval used to create the binary time series
        save_dir: Path, directory used to save the matching tables
        plot_prob: float, probability to create plot with both binary time series
        plot_dir: Path, directory in which to save the plots

    Returns:

    """

    pid = os.getpid()
    logging.basicConfig(filename=save_dir.parent / 'logs' / f'ephem_matching_{pid}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        force=True
                        )

    for target_i, target in enumerate(targets_arr):

        logger.info(f'Iterating over target {target} ({target_i + 1}/{len(targets_arr)} targets)...')

        # get start and end timestamps for this TIC
        tic_timestamps_sector = sector_timestamps_tbl.loc[sector_timestamps_tbl['target'] == target]
        tic_timestamps_sector = tic_timestamps_sector.sort_values('sector')
        if len(tic_timestamps_sector) == 0:
            logger.info(f'Target {target} not found in the timestamps table.')
            continue

        # get objects in this TIC
        objects_in_tic = objects_tbl.loc[objects_tbl['target_id'] == target].reset_index()
        if len(objects_in_tic) == 0:
            logger.info(f'No objects in target {target} to be matched to.')
            continue

        # iterate over sector runs
        sector_runs_target = tce_tbl.loc[tce_tbl['target_id'] == target, 'sector_run'].unique()
        logger.info(f'Found {len(sector_runs_target)} sector runs for target {target}.')
        for sector_run_i, sector_run in enumerate(sector_runs_target):
            logger.info(f'Iterating over sector run {sector_run} for target {target} '
                  f'({sector_run_i + 1}/{len(sector_runs_target)} sector runs)')

            # get TCEs in this TIC and sector run
            tces_in_tic_sectorun = tce_tbl.loc[(tce_tbl['target_id'] == target) &
                                               (tce_tbl['sector_run'] == sector_run)].reset_index()
            if len(tces_in_tic_sectorun) == 0:
                logger.info(f'No TCEs in target {target} for sector run {sector_run}.')
                continue

            # get start and end timestamps for the sector run
            if '-' in sector_run:
                s_sector, e_sector = [int(s) for s in sector_run.split('-')]
                sector_flag = (tic_timestamps_sector['sector'] >= s_sector) & \
                              (tic_timestamps_sector['sector'] <= e_sector)
                if sector_flag.sum() == 0:
                    logger.info(f'No start and end timestamps available for target {target} in sector run {sector_run}')
                    continue

                tstart = tic_timestamps_sector.loc[sector_flag, 'start'].values[0]
                tend = tic_timestamps_sector.loc[sector_flag, 'end'].values[-1]
            else:
                sector = int(sector_run)
                if (tic_timestamps_sector['sector'] == sector).sum() == 0:
                    logger.info(f'No start and end timestamps available for target {target} in sector run {sector_run}')
                    continue
                tstart = tic_timestamps_sector.loc[tic_timestamps_sector['sector'] == sector, 'start'].values[0]
                tend = tic_timestamps_sector.loc[tic_timestamps_sector['sector'] == sector, 'end'].values[0]

            # initialize correlation coefficient matrix to test matching between TCEs and TOIs
            corr_coef_mat = np.nan * np.ones((len(tces_in_tic_sectorun), len(objects_in_tic)))
            plot_signals = True if np.random.uniform() < plot_prob else False
            # compute correlation coefficient
            for tce_i, tce in tces_in_tic_sectorun.iterrows():
                for obj_i, obj in objects_in_tic.iterrows():
                    # obj['duration'] = tce['duration']
                    corr_coef_mat[tce_i, obj_i] = \
                        match_transit_signals(tce, obj, sampling_interval, tstart, tend,
                                              plot_signals,
                                              plot_dir)

            tic_match_tbl = pd.DataFrame(corr_coef_mat, index=tces_in_tic_sectorun['uid'],
                                         columns=objects_in_tic['uid'])

            tic_match_tbl.to_csv(save_dir / f'match_tbl_s{sector_run}_tic_{target}.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default='./config_ephem_matching.yaml')
    args = parser.parse_args()

    with(open(Path(args.config_fp).resolve(), 'r')) as file:
        config = yaml.safe_load(file)

    # results directory
    if args.output_dir is not None:
        config['exp_dir'] = args.output_dir

    exp_dir = Path(config["exp_dir"])

    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

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

    # load TCE table
    tce_tbl = pd.read_csv(config['tbl_a_fp'])
    # tce_tbl['sector_run'] = tce_tbl['uid'].apply(lambda x: '-'.join(x.split('-')[2:])[1:])
    tce_tbl.rename(columns={'tce_period': 'period', 'tce_time0bk': 'epoch', 'tce_duration': 'duration'}, inplace=True)
    # tce_tbl['duration'] *= 24
    print(f'Using table of signals to be matched against: {config["tbl_a_fp"]}')
    print(f'Table with {len(tce_tbl)} signals.')

    # load TOI catalog
    toi_tbl = pd.read_csv(config['tbl_b_fp'])

    # load table with start and end timestamps for each sector run for the TICs associated with the TCEs in the TCE
    # table
    sector_timestamps_tbl = pd.read_csv(config["sector_timestamps_tbl_fp"]).sort_values('sector')
    print(f'Using sector timestamps table {config["sector_timestamps_tbl_fp"]}')

    print(f'Plot probability: {config["plot_prob"]}')
    print(f'Sampling interval for binary time series: {config["sampling_interval"]}')

    targets_arr = tce_tbl['target_id'].unique()
    print(f'Number of targets to be iterated through: {len(targets_arr)}')

    # sequential option
    match_transit_signals_in_target(targets_arr, tce_tbl, toi_tbl, sector_timestamps_tbl, config['sampling_interval'],
                                    save_dir, plot_prob=config['plot_prob'], plot_dir=plot_dir)

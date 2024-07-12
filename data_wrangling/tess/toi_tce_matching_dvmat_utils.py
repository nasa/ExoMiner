""" Utility functions used to match TOIs with TCEs. """

# 3d party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance

# local
from src_preprocessing.ephemeris_matching.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


def match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces,
                  prob_plot=0, res_dir=None):
    """ Match TOI to TCEs. The process consists of iterating through the sectors for which the TOI was observed. For
    each sector, the TCEs from the respective sector run that are in the same TIC are compared against the TOI. The
    matching is performed by measuring the cosine distance between the templates of the TOI and TCE. The templates are
    built based on the ephemerides information of the two. The templates have the duration of the maximum period between
    the two. The timestamp of the mid-transit of the closest TCE transit to the epoch of the TOI is estimated and used
    to create the TCE template. The sampling interval determines the number of points in the templates and the matching
    threshold is used to determine the successful match between the templates.

    :param toi: row of pandas DataFrame, TOI ephemerides and data
    :param singlesector_tce_tbls: list of pandas DataFrame, single-sector runs DV TCE tables
    :param multisector_tce_tbls: list of pandas DataFrame, multi-sector runs DV TCE tables
    :param match_thr: float, matching threshold
    :param sampling_interval: float, sampling interval for pulse template time series
    :param max_num_tces: int, maximum number of TCEs that can be matched to the TOI
    :param prob_plot: float, probability to plot ephemerides templates for both TCE and TOI
    :param  res_dir: Path, results directory
    :return:
        data_to_tbl: dict, include TIC, TOI ID, matched TCEs and respective matching distances to TOI
    """

    toi_sectors = [int(sector) for sector in toi['Sectors'].split(',')]

    matching_dist_dict = {}

    for toi_sector in toi_sectors:  # iterate through the sectors the TOI was observed

        # check the single sector run table
        if toi_sector in singlesector_tce_tbls:

            tce_tbl_aux = singlesector_tce_tbls[toi_sector]  # get TCE table for that single-sector run

            # get TCEs in the run for the same TIC
            tce_found = tce_tbl_aux.loc[tce_tbl_aux['catId'] == toi['TIC ID']]
        else:
            tce_found = []

        if len(tce_found) > 0:

            for tce_i, tce in tce_found.iterrows():  # iterate through the TCEs that belong to the same TIC

                tceid = int(tce['planetIndexNumber'])

                # size of the template is set to larger orbital period between TOI and TCE
                phase = max(toi['Period (days)'], tce['allTransitsFit_orbitalPeriodDays_value'])

                # find phase difference of TCE to the TOI
                closest_tce_epoch = find_nearest_epoch_to_this_time(
                    tce['allTransitsFit_transitEpochBtjd_value'],
                    tce['allTransitsFit_orbitalPeriodDays_value'],
                    toi['Epoch (TBJD)']
                )

                toi_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                                       duration=toi['Duration (hours)'] / 24,
                                                       period=toi['Period (days)'],
                                                       tStart=toi['Epoch (TBJD)'] - phase / 2,
                                                       tEnd=toi['Epoch (TBJD)'] + phase / 2,
                                                       samplingInterval=sampling_interval)

                tce_bin_ts = create_binary_time_series(epoch=closest_tce_epoch,
                                                       duration=tce['allTransitsFit_transitDurationHours_value'] / 24,
                                                       period=tce['allTransitsFit_orbitalPeriodDays_value'],
                                                       tStart=toi['Epoch (TBJD)'] - phase / 2,
                                                       tEnd=toi['Epoch (TBJD)'] + phase / 2,
                                                       samplingInterval=sampling_interval)

                # compute distance between TOI and TCE templates as cosine distance
                if toi_bin_ts.sum() == 0 or tce_bin_ts.sum() == 0:
                    match_distance = 1
                else:
                    match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)
                    if np.random.random() < prob_plot:  # plot templates for some matches
                        tce_aux = {'id': f'{toi["TIC ID"]}.{tceid}-S{toi_sector}',
                                   'template': tce_bin_ts,
                                   'period': tce['allTransitsFit_orbitalPeriodDays_value'],
                                   }
                        toi_aux = {'id': f'{toi["TOI"]}',
                                   'template': toi_bin_ts,
                                   'period': toi['Period (days)'],
                                   }
                        plot_templates(tce_aux, toi_aux, match_distance, res_dir / 'plots')

                # set matching distance if it is smaller than the matching threshold
                if match_distance < match_thr:
                    matching_dist_dict[f'{toi_sector}_{tceid}'] = match_distance

        # check the multi sector runs tables
        for multisector_tce_tbl in multisector_tce_tbls:

            if toi_sector >= multisector_tce_tbl[0] and toi_sector <= multisector_tce_tbl[1]:

                # get multi-sector run TCE table
                tce_tbl_aux = multisector_tce_tbls[multisector_tce_tbl]

                # get TCEs in the run for the same TIC
                tce_found = tce_tbl_aux.loc[tce_tbl_aux['catId'] == toi['TIC ID']]

                if len(tce_found) > 0:

                    for tce_i, tce in tce_found.iterrows():

                        tceid = int(tce['planetIndexNumber'])

                        phase = max(toi['Period (days)'], tce['allTransitsFit_orbitalPeriodDays_value'])

                        # find phase difference of TCE to the TOI
                        closest_tce_epoch = find_nearest_epoch_to_this_time(
                            tce['allTransitsFit_transitEpochBtjd_value'],
                            tce['allTransitsFit_orbitalPeriodDays_value'],
                            toi['Epoch (TBJD)'])

                        toi_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                                               duration=toi['Duration (hours)'] / 24,
                                                               period=toi['Period (days)'],
                                                               tStart=toi['Epoch (TBJD)'] - phase / 2,
                                                               tEnd=toi['Epoch (TBJD)'] + phase / 2,
                                                               samplingInterval=sampling_interval)

                        tce_bin_ts = create_binary_time_series(epoch=closest_tce_epoch,
                                                               duration=tce['allTransitsFit_transitDurationHours_value']
                                                                        / 24,
                                                               period=tce['allTransitsFit_orbitalPeriodDays_value'],
                                                               tStart=toi['Epoch (TBJD)'] - phase / 2,
                                                               tEnd=toi['Epoch (TBJD)'] + phase / 2,
                                                               samplingInterval=sampling_interval)

                        if toi_bin_ts.sum() == 0 or tce_bin_ts.sum() == 0:
                            match_distance = 1
                        else:
                            match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)
                            if np.random.random() < prob_plot:
                                tce_aux = {'id': f'{toi["TIC ID"]}.{tceid}-S{multisector_tce_tbl[0]}-{multisector_tce_tbl[1]}',
                                           'template': tce_bin_ts,
                                           'period': tce['allTransitsFit_orbitalPeriodDays_value'],
                                           }
                                toi_aux = {'id': f'{toi["TOI"]}',
                                           'template': toi_bin_ts,
                                           'period': toi['Period (days)'],
                                           }
                                plot_templates(tce_aux, toi_aux, match_distance, res_dir / 'plots')

                        if match_distance < match_thr:
                            matching_dist_dict[f'{multisector_tce_tbl[0]}-{multisector_tce_tbl[1]}_{tceid}'] = \
                                match_distance

    # sort TCEs based on matching distance
    matching_dist_dict = {k: v for k, v in sorted(matching_dist_dict.items(), key=lambda x: x[1])}

    # add TOI row to the csv matching file
    data_to_tbl = {'TOI ID': toi['TOI'],
                   'TIC': toi['TIC ID'],
                   'Matched TCEs': ' '.join(list(matching_dist_dict.keys()))}
    matching_dist_arr = list(matching_dist_dict.values())
    data_to_tbl.update({f'matching_dist_{i}': matching_dist_arr[i] if i < len(matching_dist_arr) else np.nan
                        for i in range(max_num_tces)})

    return data_to_tbl


def match_set_tois_tces(toi_tbl, tbl_i, match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr,
                        sampling_interval, max_num_tces, res_dir, prob_plot=0, logger=None):
    """ Match TOIs to TCEs.

    :param toi_tbl: pandas DataFrame, TOIs
    :param tbl_i: int, subset table id
    :param match_tbl_cols: list, columns for matching table
    :param singlesector_tce_tbls: list of pandas DataFrame, single-sector runs DV TCE tables
    :param multisector_tce_tbls: list of pandas DataFrame, multi-sector runs DV TCE tables
    :param match_thr: float, matching threshold
    :param sampling_interval: float, sampling interval for pulse template time series
    :param max_num_tces: int, maximum number of TCEs that can be matched to the TOI
    :param res_dir: Path, results directory
    :param prob_plot: float, probability to plot ephemerides templates for both TCE and TOI
    :param logger: logger
    :return:
        matching_tbl: pandas DataFrame, TOI matching table
    """

    matching_tbl = pd.DataFrame(columns=match_tbl_cols, data=np.zeros((len(toi_tbl), len(match_tbl_cols))))

    for toi_i, toi in toi_tbl.iterrows():
        prt_str = f'[Matching Subtable {tbl_i}] Matching TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})'
        if logger is None:
            print(prt_str)
        else:
            logger.info(prt_str)

        match_toi_row = match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval,
                                      max_num_tces, prob_plot=prob_plot, res_dir=res_dir)

        matching_tbl.loc[toi_i] = pd.Series(match_toi_row)

        prt_str = f'[Matching Subtable {tbl_i}] Matched TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})'
        if logger is None:
            print(prt_str)
        else:
            logger.info(prt_str)

    prt_str = f'[Matching Subtable {tbl_i}] Finished matching {len(toi_tbl)} TOIs'
    if logger is None:
        print(prt_str)
    else:
        logger.info(prt_str)

    matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval:.4f}_'
                                  f'{tbl_i}.csv', index=False)

    return matching_tbl


def plot_templates(tce, toi, match_dist, plot_dir):
    """ Plot TCE and TOI templates.

    :param tce: dict, TCE includes `id` and `template` time series
    :param toi: dict, TOI includes `id` and `template` time series
    :param plot_dir: Path, directory used to save plots
    :return:
    """

    f, ax = plt.subplots()
    ax.plot(tce['template'], label=f'TIC {tce["id"]}', zorder=2, linestyle='dashed')
    ax.plot(toi['template'], label=f'TOI {toi["id"]}', zorder=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Single-phase Pulse Transit\nMatch distance={match_dist:.2f}\nTCE/TOI Period (days): {tce["period"]:.3f}/{toi["period"]:.3f}')
    ax.legend()
    f.savefig(plot_dir / f'toi_{toi["id"]}-tic_{tce["id"]}.png')
    plt.close()

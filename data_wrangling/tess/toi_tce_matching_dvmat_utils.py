# 3d party
import numpy as np
import pandas as pd
from scipy.spatial import distance

# local
from data_wrangling.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


def match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces):
    """ Match TOI to TCEs. The process consists of iterating through the sectors for which the TOI was observed. For
    each sector, the TCEs from the respective sector run that are in the same TIC are compared against the TOI. The
    matching is performed by measuring the cosine distance between the templates of the TOI and TCE. The templates are
    built based on the ephemerides information of the two. The templates have the duration of the maximum period between
    the two. The timestamp of the mid-transit of the closest TCE transit to the epoch of the TOI is estimated and used
    to create the TCE template. The sampling interval determines the number of points in the templates and the matching
    threshold is used to determined the successful match between the templates.

    :param toi: row of pandas DataFrame, TOI ephemerides and data
    :param singlesector_tce_tbls: list of pandas DataFrame, single-sector runs DV TCE tables
    :param multisector_tce_tbls: list of pandas DataFrame, multi-sector runs DV TCE tables
    :param match_thr: float, matching threshold
    :param sampling_interval: float, sampling interval for pulse template time series
    :param max_num_tces: int, maximum number of TCEs that can be matched to the TOI
    :return:
        data_to_tbl: dict, include TIC, TOI ID, matched TCEs and respective matching distances to TOI
    """

    toi_sectors = [int(sector) for sector in toi['Sectors'].split(',')]

    matching_dist_dict = {}

    for toi_sector in toi_sectors:  # iterate through the sectors the TOI was observed

        # check the single sector run table
        if toi_sector in singlesector_tce_tbls:
            tce_tbl_aux = singlesector_tce_tbls[toi_sector]

            # get TCEs in the run for the same TIC
            tce_found = tce_tbl_aux.loc[tce_tbl_aux['catId'] == toi['TIC ID']]
        else:
            tce_found = []

        if len(tce_found) > 0:

            for tce_i, tce in tce_found.iterrows():

                tceid = tce['planetIndexNumber']

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

                        tceid = tce['planetIndexNumber']

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
                        sampling_interval, max_num_tces, res_dir):
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
    :return:
        matching_tbl: pandas DataFrame, TOI matching table
    """

    matching_tbl = pd.DataFrame(columns=match_tbl_cols, data=np.zeros((len(toi_tbl), len(match_tbl_cols))))

    for toi_i, toi in toi_tbl.iterrows():
        print(f'[Matching Subtable {tbl_i}] Matching TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})')

        match_toi_row = match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval,
                                      max_num_tces)

        matching_tbl.loc[toi_i] = pd.Series(match_toi_row)

        print(f'[Matching Subtable {tbl_i}] Matched TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})')

    print(f'[Matching Subtable {tbl_i}] Finished matching {len(toi_tbl)} TOIs')

    matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}_'
                                  f'{tbl_i}.csv', index=False)

    return matching_tbl

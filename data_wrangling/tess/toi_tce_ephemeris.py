# 3rd party
import numpy as np
import pandas as pd
from scipy.spatial import distance

# local
from src_preprocessing.ephemeris_matching.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


def get_bin_ts_toi_tce(toi, tce, sampling_interval):

    phase = max(toi['period'], tce['period'])

    # find phase difference of TCE to the TOI
    phase_diff = find_nearest_epoch_to_this_time(
        tce['epoch'],
        tce['period'],
        toi['epoch']) - toi['epoch']


    toi_bin_ts = create_binary_time_series(
        epoch=toi['epoch'],
        duration=toi['duration'] / 24,
        period=toi['period'],
        tStart=toi['epoch'] - phase / 2,
        tEnd=toi['epoch'] + phase / 2,
        samplingInterval=sampling_interval
    )

    tce_bin_ts = create_binary_time_series(
        epoch=toi['epoch'] + phase_diff,
        duration=tce['duration'] / 24,
        period=tce['period'],
        tStart=toi['epoch'] - phase / 2,
        tEnd=toi['epoch'] + phase / 2,
        samplingInterval=sampling_interval
    )

    # compute distance between TOI and TCE templates as cosine distance
    match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

    return toi_bin_ts, tce_bin_ts, match_distance



def match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces):

    toi_sectors = [int(sector) for sector in toi['Sectors'].split(',')]

    matching_dist_dict = {}

    # create template for TOI
    toi_bin_ts_max = create_binary_time_series(
        epoch=toi['epoch'],
        duration=toi['duration'] / 24,
        period=toi['period'],
        tStart=toi['epoch'] - toi['period'] / 2,
        tEnd=toi['epoch'] + toi['period'] / 2,
        samplingInterval=sampling_interval
    )

    for toi_sector in toi_sectors:  # iterate through the sectors the TOI was observed in

        # check the single sector run table
        if toi_sector in singlesector_tce_tbls:
            tce_tbl_aux = singlesector_tce_tbls[toi_sector]

            # get TCEs in the run for the same TIC
            tce_found = tce_tbl_aux.loc[tce_tbl_aux['target_id'] == toi['target_id']]
        else:
            tce_found = []

        if len(tce_found) > 0:

            for tce_i, tce in tce_found.iterrows():

                # size of the template is set to larger orbital period between TOI and TCE
                phase = max(toi['period'], tce['period'])

                # find phase difference of TCE to the TOI
                phase_diff = find_nearest_epoch_to_this_time(
                    tce['epoch'],
                    tce['period'],
                    toi['period'])

                if phase == toi['period']:  # if TOI orbital period is larger, keep the TOI template
                    toi_bin_ts = toi_bin_ts_max
                else:
                    toi_bin_ts = create_binary_time_series(
                        epoch=toi['epoch'],
                                                           duration=toi['duration'] / 24,
                                                           period=toi['period'],
                                                           tStart=toi['epoch'] - phase / 2,
                                                           tEnd=toi['epoch'] + phase / 2,
                                                           samplingInterval=sampling_interval
                                                           )

                tce_bin_ts = create_binary_time_series(
                    epoch=toi['epoch'] + phase_diff,
                                                       duration=tce['duration'] / 24,
                                                       period=tce['period'],
                                                       tStart=toi['epoch'] - phase / 2,
                                                       tEnd=toi['epoch'] + phase / 2,
                                                       samplingInterval=sampling_interval
                                                       )

                # compute distance between TOI and TCE templates as cosine distance
                match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                # set matching distance if it is smaller than the matching threshold
                if match_distance < match_thr:
                    matching_dist_dict[f'{toi_sector}_{tce["tce_plnt_num"]}'] = match_distance

        # check the multi sector runs tables
        for multisector_tce_tbl in multisector_tce_tbls:

            if toi_sector >= multisector_tce_tbl[0] and toi_sector <= multisector_tce_tbl[1]:

                # get multi-sector run TCE table
                tce_tbl_aux = multisector_tce_tbls[multisector_tce_tbl]

                # get TCEs in the run for the same TIC
                tce_found = tce_tbl_aux.loc[tce_tbl_aux['target_id'] == toi['target_id']]

                if len(tce_found) > 0:

                    for tce_i, tce in tce_found.iterrows():

                        phase = max(toi['period'], tce['period'])

                        # find phase difference of TCE to the TOI
                        phase_diff = find_nearest_epoch_to_this_time(tce['epoch'],
                                                                     tce['period'],
                                                                     toi['epoch'])

                        if phase == toi['period']:
                            toi_bin_ts = toi_bin_ts_max
                        else:
                            toi_bin_ts = create_binary_time_series(epoch=toi['epoch'],
                                                                   duration=toi['duration'] / 24,
                                                                   period=toi['period'],
                                                                   tStart=toi['epoch'] - phase / 2,
                                                                   tEnd=toi['epoch'] + phase / 2,
                                                                   samplingInterval=sampling_interval)

                        tce_bin_ts = create_binary_time_series(epoch=toi['epoch)'] + phase_diff,
                                                               duration=tce['duration'] / 24,
                                                               period=tce['period'],
                                                               tStart=toi['epoch'] - phase / 2,
                                                               tEnd=toi['epoch'] + phase / 2,
                                                               samplingInterval=sampling_interval)

                        # TODO: add epsilon to avoid nan value when one of the vectors is zero?
                        match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                        if match_distance < match_thr:
                            matching_dist_dict[f'{multisector_tce_tbl[0]}-{multisector_tce_tbl[1]}_{tce["tce_plnt_num"]}'] = \
                                match_distance

    # sort TCEs based on matching distance
    matching_dist_dict = {k: v for k, v in sorted(matching_dist_dict.items(), key=lambda x: x[1])}

    # add TOI row to the csv matching file
    data_to_tbl = {'TOI ID': toi['TOI'],
                   'TIC': toi['target_id'],
                   'Matched TCEs': ' '.join(list(matching_dist_dict.keys()))}
    matching_dist_arr = list(matching_dist_dict.values())
    data_to_tbl.update({f'matching_dist_{i}': matching_dist_arr[i] if i < len(matching_dist_arr) else np.nan
                        for i in range(max_num_tces)})

    return data_to_tbl


def match_set_tois_tces(toi_tbl, tbl_i, match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr,
                        sampling_interval, max_num_tces, res_dir):

    matching_tbl = pd.DataFrame(columns=match_tbl_cols, data=np.zeros((len(toi_tbl), len(match_tbl_cols))))

    for toi_i, toi in toi_tbl.iterrows():

        print(f'[Matching Subtable {tbl_i}] Matching TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})')

        match_toi_row = match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval,
                                      max_num_tces)

        matching_tbl.loc[toi_i] = pd.Series(match_toi_row)

    print(f'[Matching Subtable {tbl_i}] Finished matching {len(toi_tbl)} TOIs')

    matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}_'
                                  f'{tbl_i}.csv', index=False)

    return matching_tbl

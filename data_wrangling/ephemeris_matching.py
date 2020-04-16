"""
Script created to perform ephemeris matching across TESS sectors.
"""

import logging
import numpy as np
# import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import itertools
# import os

#%%


def find_first_epoch_after_this_time(epoch, period, reference_time):
    """ Finds the first epoch after a certain reference time.

    :param epoch: float initial epoch value (TJD/KJD)
    :param period: float, period (d)
    :param reference_time: float, start reference time
    :return:
        tepoch: float, new epoch value

    # Code is ported from Jeff's Matlab ephemeris matching code
    """

    if epoch < reference_time:
        tepoch = epoch + period * np.ceil((reference_time - epoch) / period)
    else:
        tepoch = epoch - period * np.floor((epoch - reference_time) / period)

    return tepoch


def create_binary_time_series(epoch, duration, period, tStart, tEnd, samplingInterval):
    """ Creates a binary time series based on the the ephemeris.

    :param epoch: float, epoch in days
    :param duration: float, transit duration in days
    :param period: float, orbital period in days
    :param tStart: float, reference start time
    :param tEnd: float, reference end time
    :param samplingInterval: float, sampling interval in days
    :return:
        binary_time_series: binary array with 1's for in-transit points and 0's otherwise

    # Code is ported from Jeff's Matlab ephemeris matching code
    """

    # sampleTimes = np.linspace(tStart / samplingInterval, tEnd / samplingInterval, (tEnd - tStart) / samplingInterval,
    #                           endpoint=True)
    sampleTimes = np.linspace(tStart / samplingInterval, tEnd / samplingInterval,
                              int((tEnd - tStart) / samplingInterval),
                              endpoint=True)

    # mid-transit timestamps between epoch and reference end time
    midTransitTimes = np.arange(epoch, tEnd, period)

    # get transits whose mid-point is before and after the reference start and end times
    if len(midTransitTimes) != 0:  # epoch before reference end time
        midTransitTimeBefore = midTransitTimes[0] - period
        midTransitTimeAfter = midTransitTimes[-1] + period
    else:  # epoch after reference end time
        midTransitTimeBefore = epoch - period
        midTransitTimeAfter = epoch

    # concatenate the mid-transit timestamps before, inside and after the time array
    extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

    # get beginning and end timestamps of the transits
    # convert to units of sampling interval
    startTransitTimes = (extendedMidTransitTimes - 0.5 * duration) / samplingInterval
    endTransitTimes = (extendedMidTransitTimes + 0.5 * duration) / samplingInterval

    # initialize binary time series - points with 1's belong to the transit
    binary_time_series = np.zeros(len(sampleTimes), dtype='uint8')
    # binary_time_series = np.zeros(len(time), dtype='bool')

    # set to 1 the in-transit timestamps
    for sTransit, eTransit in zip(startTransitTimes, endTransitTimes):

        transit_idxs = np.where(sampleTimes >= sTransit)[0]
        transit_idxs = np.intersect1d(transit_idxs, np.where(sampleTimes < eTransit)[0])
        binary_time_series[transit_idxs] = 1
        # binary_time_series[transit_idxs] = True

    return binary_time_series


#%% Pair TCEs across sectors based on their ephemeris

# read TCE table
ephem_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv')

# convert to days
ephem_tbl['transitDurationHours'] /= 24

# get unique TIC IDs
tic_ids = ephem_tbl['tic'].unique()

# match score threshold
matchscore_thr = 0.28

# {(sectori, sectorj): {'scores': MxN, 'tces': [[row_tce1, ...], [col_tce1, ...]]}, ...}
matchscore_data = {}
num_tces_only1sector = 0  # 3651

# matched TCEs {tic_id1: {new_tce_plnt_num1: [(sectori, old_tce_plnt_numi, old_dispositioni),
# (sectorj, old_tce_plnt_numj, old_dispositionj), ...]}, tic_id2:...}
matched_tces_table = {}
for tic_i, tic_id in enumerate(tic_ids):

    print('TIC ID = {:.0f} - {}/{}'.format(tic_id, tic_i, len(tic_ids)))

    matched_tces_table[tic_id] = {}

    # get sectors for the target star
    tic_sectors = np.sort(ephem_tbl.loc[ephem_tbl['tic'] == tic_id]['sector'].unique())
    print('TIC sectors for target {:.0f}: {}'.format(tic_id, tic_sectors))

    if len(tic_sectors) == 1:
        print('TCEs only found in 1 sector (sector: {})'.format(tic_sectors))

        tces_sector1 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
                                     (ephem_tbl['sector'] == tic_sectors[0])][['tce_plnt_num', 'disposition']].values
        matched_tces_table[tic_id] = {i + 1: [(tic_sectors[0], tces_sector1[i, 0], tces_sector1[i, 1])]
                                      for i in range(len(tces_sector1))}
        num_tces_only1sector += len(tces_sector1)
        continue

    # auxiliary dictionary for book-keeping of the tces already matched for this target star
    # {sector1: [old_tce_plnt_numi, old_tce_plnt_numj, ...], sector2:...}
    matched_tces = {sector: [] for sector in tic_sectors}

    # get all pairwise combinations between the sectors
    paired_sectors = itertools.combinations(tic_sectors, 2)

    for paired_sector in paired_sectors:

        print('Trying to pair TCEs from TIC ID {:.0f} - sectors {} and {}'.format(tic_id, paired_sector[0],
                                                                                  paired_sector[1]))

        # get TCEs for the two sectors (let's call them first and sector sectors); original TCE planet number
        tces_sector1 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
                                     (ephem_tbl['sector'] == paired_sector[0])][['tce_plnt_num', 'disposition']].values
        tces_sector2 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
                                     (ephem_tbl['sector'] == paired_sector[1])][['tce_plnt_num', 'disposition']].values

        # remove TCEs that were paired already with a TCE from a previous sector
        tces_sector1 = [tce for tce in tces_sector1 if tce[:-1] not in matched_tces[paired_sector[0]]]
        tces_sector2 = [tce for tce in tces_sector2 if tce[:-1] not in matched_tces[paired_sector[1]]]

        if len(tces_sector1) == 0 or len(tces_sector2) == 0:
            print('TCEs in at least one of the sectors were already paired: {}|{}'.format(len(tces_sector1) == 0,
                                                                                          len(tces_sector2) == 0))
            continue

        # create match score matrix
        matchscore_mat = 2 * np.ones((len(tces_sector1), len(tces_sector2)))
        # datapoints = {}

        # compute match scores for each pair between TCEs in the first and second sectors
        for tce1_i in range(len(tces_sector1)):

            # method - 3D ephemeris data points
            p1 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
                               (ephem_tbl['tce_plnt_num'] == tces_sector1[tce1_i][0]) &
                               (ephem_tbl['sector'] == paired_sector[0])][['transitDurationHours',
                                                                           'orbitalPeriodDays',
                                                                           'transitEpochBtjd']].values[0]
            #
            # per1, e1 = p1[1], p1[2]
            #
            # p1 = (p1 - min_ephem) / (max_ephem - min_ephem)
            #
            # p1[-1] = 0

            # method - template matching
            bintseries1 = create_binary_time_series(epoch=0,
                                                    duration=p1[0],
                                                    period=p1[1],
                                                    tStart=0,
                                                    tEnd=p1[1],
                                                    samplingInterval=0.00001)

            for tce2_i in range(len(tces_sector2)):

                # method - 3D ephemeris data points
                p2 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
                                   (ephem_tbl['tce_plnt_num'] == tces_sector2[tce2_i][0]) &
                                   (ephem_tbl['sector'] == paired_sector[1])][['transitDurationHours',
                                                                               'orbitalPeriodDays',
                                                                               'transitEpochBtjd']].values[0]
                #
                # e2 = p2[2]
                #
                # p2 = (p2 - min_ephem) / (max_ephem - min_ephem)
                #
                # epoch_shift = np.abs(e1 - e2) / per1
                # p2[-1] = np.abs(round(epoch_shift) - epoch_shift)
                #
                # matchscore_mat[tce1_i, tce2_i] = distance.cosine(p1, p2)
                #
                # datapoints[(tces_sector1[tce1_i], paired_sector[0], tces_sector2[tce2_i], paired_sector[1])] = (p1, p2)

                # method - template matching
                epoch_shift = np.abs(p1[2] - p2[2]) / p1[1]
                e2 = np.abs(round(epoch_shift) - epoch_shift) * p1[1]
                bintseries2 = create_binary_time_series(epoch=e2,
                                                        duration=p2[0],
                                                        period=p2[1],
                                                        tStart=0,
                                                        tEnd=p1[1],
                                                        samplingInterval=0.00001)

                matchscore_mat[tce1_i, tce2_i] = distance.cosine(bintseries1, bintseries2)

                # method - binary time series matching
                # matchscore_mat[tce1_i, tce2_i] = distance.cosine(bintseries[tic_id][(tces_sector1[tce1_i],
                #                                                                      paired_sector[0])],
                #                                                  bintseries[tic_id][(tces_sector2[tce2_i],
                #                                                                      paired_sector[1])])
                # matchscore_mat[tce1_i, tce2_i] = distance.jaccard(bintseries[tic_id][(tces_sector1[tce1_i],
                #                                                                       paired_sector[0])],
                #                                                   bintseries[tic_id][(tces_sector2[tce2_i],
                #                                                                       paired_sector[1])])
                # matchscore_mat[tce1_i, tce2_i] = distance.hamming(bintseries[tic_id][(tces_sector1[tce1_i],
                #                                                                       paired_sector[0])],
                #                                                   bintseries[tic_id][(tces_sector2[tce2_i],
                #                                                                       paired_sector[1])])

        matchscore_data[paired_sector] = {'scores': matchscore_mat, 'tces': [tces_sector1, tces_sector2]}

        # greedily pair TCEs
        # sort indexes in ascending order
        sorted_idxs = np.dstack(np.unravel_index(np.argsort(matchscore_mat.ravel()),
                                                 (len(tces_sector1), len(tces_sector2))))[0]

        # iterate through the indexes of the matrix in ascending order of match score
        tces_idxs_visited = [[], []]
        for idx_pair in sorted_idxs:

            # check if at least one of the TCEs was already matched
            if idx_pair[0] in tces_idxs_visited[0] or idx_pair[1] in tces_idxs_visited[1]:

                # print('One of the TCEs was already visited: '
                #       '{} {:.0f} s{} {}|{} {:.0f} s{} {}; '
                #       'Match score = {}, Data points {}'.format(idx_pair[0] in tces_idxs_visited[0],
                #                                                 tic_id, paired_sector[0],
                #                                                 tces_sector1[idx_pair[0]],
                #                                                 idx_pair[1] in tces_idxs_visited[1],
                #                                                 tic_id, paired_sector[1],
                #                                                 tces_sector2[idx_pair[1]],
                #                                                 matchscore_mat[idx_pair[0], idx_pair[1]],
                #                                                 datapoints[(tces_sector1[idx_pair[0]], paired_sector[0],
                #                                                            tces_sector2[idx_pair[1]], paired_sector[1])]
                #                                                 )
                #       )
                print('One of the TCEs was already visited: '
                      '{} {:.0f} s{} {}|{} {:.0f} s{} {}; '
                      'Match score = {}'.format(idx_pair[0] in tces_idxs_visited[0],
                                                                tic_id, paired_sector[0],
                                                                tces_sector1[idx_pair[0]],
                                                                idx_pair[1] in tces_idxs_visited[1],
                                                                tic_id, paired_sector[1],
                                                                tces_sector2[idx_pair[1]],
                                                                matchscore_mat[idx_pair[0], idx_pair[1]]
                                                                )
                      )

                if matchscore_mat[idx_pair[0], idx_pair[1]] < matchscore_thr:
                    print('Below threshold!!!!!!')
                continue

            # check if match score is above threshold
            if matchscore_mat[idx_pair[0], idx_pair[1]] >= matchscore_thr:

                # print('TCEs not paired: {:.0f} s{} {} - {:.0f} s{} {}; Match score = {}, Data points {}'.format(
                #     tic_id, paired_sector[0], tces_sector1[idx_pair[0]],
                #     tic_id, paired_sector[1], tces_sector2[idx_pair[1]],
                #     matchscore_mat[idx_pair[0], idx_pair[1]],
                #     datapoints[(tces_sector1[idx_pair[0]], paired_sector[0],
                #                 tces_sector2[idx_pair[1]], paired_sector[1])]))

                print('TCEs not paired: {:.0f} s{} {} - {:.0f} s{} {}; Match score = {}'.format(
                    tic_id, paired_sector[0], tces_sector1[idx_pair[0]],
                    tic_id, paired_sector[1], tces_sector2[idx_pair[1]],
                    matchscore_mat[idx_pair[0], idx_pair[1]]))

                # add non-paired TCE from first sector to the lookup table of matched TCEs only the first time it is
                # visited
                if tic_sectors[np.where(tic_sectors == paired_sector[0])[0][0] + 1] == paired_sector[1]:
                    # if len(matched_tces_table[tic_id]) + 1 not in matched_tces_table[tic_id]:
                    matched_tces_table[tic_id][len(matched_tces_table[tic_id]) + 1] = [(paired_sector[0],
                                                                                        tces_sector1[idx_pair[0]][0],
                                                                                        tces_sector1[idx_pair[0]][1])]
                    # else:
                    #     matched_tces_table[tic_id][len(matched_tces_table[tic_id]) + 1].append((paired_sector[0],
                    #                                                                             tces_sector1[
                    #                                                                                 idx_pair[0]]))

                    # mark this TCE as visited for this sector pairing
                    tces_idxs_visited[0].append(idx_pair[0])

            else:  # add paired TCEs to the lookup table of matched TCEs

                # print('TCEs PAIRED: {:.0f} s{} {} - {:.0f} s{} {}; Match score = {}, Data points {}'.format(
                #     tic_id, paired_sector[0], tces_sector1[idx_pair[0]],
                #     tic_id, paired_sector[1], tces_sector2[idx_pair[1]],
                #     matchscore_mat[idx_pair[0], idx_pair[1]],
                #     datapoints[(tces_sector1[idx_pair[0]], paired_sector[0],
                #                 tces_sector2[idx_pair[1]], paired_sector[1])]
                # ))

                print('TCEs PAIRED: {:.0f} s{} {} - {:.0f} s{} {}; Match score = {}'.format(
                    tic_id, paired_sector[0], tces_sector1[idx_pair[0]],
                    tic_id, paired_sector[1], tces_sector2[idx_pair[1]],
                    matchscore_mat[idx_pair[0], idx_pair[1]]
                ))

                # check if first sector TCE is already in the table
                tce_in_table = False
                for new_tceid in matched_tces_table[tic_id]:
                    for old_tceid in matched_tces_table[tic_id][new_tceid]:
                        if old_tceid[:-1] == (paired_sector[0], tces_sector1[idx_pair[0]][0]):
                            tce_in_table = True
                            break
                    if old_tceid[:-1] == (paired_sector[0], tces_sector1[idx_pair[0]][0]):
                        break

                # TODO: an alternative to this check would be to add all TCEs not matched as new TCEs when doing the
                #  first sector comparison
                if tce_in_table:  # if first sector TCE is already in the table (either alone or not)
                    matched_tces_table[tic_id][new_tceid].append((paired_sector[1], tces_sector2[idx_pair[1]][0],
                                                                  tces_sector2[idx_pair[1]][1]))
                else:  # create new TCE
                    matched_tces_table[tic_id][len(matched_tces_table[tic_id]) + 1] = \
                        [(paired_sector[0], tces_sector1[idx_pair[0]][0], tces_sector1[idx_pair[0]][1]),
                         (paired_sector[1], tces_sector2[idx_pair[1]][0], tces_sector2[idx_pair[1]][1])]

                    # if len(matched_tces_table[tic_id]) + 1 not in matched_tces_table[tic_id]:
                    #     matched_tces_table[tic_id][len(matched_tces_table[tic_id]) + 1] = \
                    #         [(paired_sector[0], tces_sector1[idx_pair[0]]), (paired_sector[1], tces_sector2[idx_pair[1]])]
                    # else:
                    #     matched_tces_table[tic_id][len(matched_tces_table[tic_id]) + 1].extend(
                    #         [(paired_sector[0], tces_sector1[idx_pair[0]]), (paired_sector[1], tces_sector2[idx_pair[1]])])

                # add paired TCEs from sector 2
                # as a greedy approach, these TCEs were already paired to TCE(s) from previous sectors and are not
                # checked in next iterations
                matched_tces[paired_sector[1]].append(tces_sector2[idx_pair[1]][0])

                # mark these TCEs as visited for this sector pairing
                tces_idxs_visited[0].append(idx_pair[0])
                tces_idxs_visited[1].append(idx_pair[1])

print('Number of unique TCEs: {}'.format(sum([len(matched_tces_table[tic_id]) for tic_id in matched_tces_table])))
np.save('/home/msaragoc/Downloads/matched_tces_tbl_TESS_ss1-13.npy', matched_tces_table)

#%% Solve conflicts between matched TCEs
# Priority-based rule: KP/CP, FP (IS, V, EB, O), PC

new_eph_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv')

for tic_i, tic_id in enumerate(tic_ids):

    # print('TIC ID = {:.0f} - {}/{}'.format(tic_id, tic_i + 1, len(tic_ids)))

    for tce_i, tce_id in enumerate(matched_tces_table[tic_id]):

        # print('TCE {}/{} - old TCEs: {}'.format(tce_i + 1, len(matched_tces_table[tic_id]), new_tce_df[[0, 1]].values))

        if len(matched_tces_table[tic_id][tce_id]) == 1:
            # print('Only one TCE.')
            continue

        new_tce_df = pd.DataFrame(matched_tces_table[tic_id][tce_id])

        if len(set(new_tce_df[2].values) & {'KP', 'CP'}) > 0:  # at least one KP/CP in the matched TCEs
            if len(np.unique(new_tce_df[2].values)) > 1:
                print('TIC ID = {:.0f} - old TCEs: {}'.format(tic_id, new_tce_df[[0, 1]].values))
            label = 'KP'
        elif len(set(new_tce_df[2].values) & {'EB', 'IS', 'V', 'O'}) > 0:  # at least one FP in the matched TCEs
            label = 'FP'
        else:  # all TCEs have 'PC' as disposition
            label = 'PC'

        # print('TCE {}/{} - old TCEs: {}'.format(tce_i + 1, len(matched_tces_table[tic_id]), new_tce_df[[0, 1]].values))
        # print('Old labels: {} -> new label: {}'.format(new_tce_df[2].values, label))
        # print('-' * 50)

        # update dispositions
        for old_tce in new_tce_df[[0, 1]].values:

            new_eph_tbl.loc[(new_eph_tbl['tic'] == tic_id) &
                            (new_eph_tbl['sector'] == old_tce[0]) &
                            (new_eph_tbl['tce_plnt_num'] == old_tce[1]), 'disposition'] = label

new_eph_tbl.to_csv('/home/msaragoc/Downloads/tcetable_matched.csv', index=False)

print(new_eph_tbl['disposition'].value_counts())

old_eph_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv')

print(old_eph_tbl['disposition'].value_counts())

#%% Matching KOI with TCEs to add KOI fields such as FP flags - sequential match

logging.basicConfig(filename='/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                             'koi_ephemeris_matching/koi_ephemeris_matching.log', filemode='a', format='%(message)s',
                    level=logging.INFO)

# Q1-Q17 DR25 TCE list
keplerTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_processed.csv',
                             header=0)

# Cumulative KOI list
koiCumTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/koi_table/'
                          'cumulative_2020.02.21_10.29.22.csv', header=90)

# filter KOIs that do not come from Q1-Q17 DR25 TCE list
koiCumTable = koiCumTable.loc[koiCumTable['koi_tce_delivname'] == 'q1_q17_dr25_tce']

koiColumnNames = np.array(koiCumTable.columns.values.tolist())[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -25, -24, -21,
                                                                -20]]

# initialize new columns
keplerTceTable = pd.concat([keplerTceTable, pd.DataFrame(columns=koiColumnNames)])

# matching threshold
matchThreshold = 0.25

numKoi = len(koiCumTable)
samplingInterval = 1e-4
koiNotMatched, koiNoTceInTarget = [], []
for koi_i, koi in koiCumTable.iterrows():

    # if koi_i % 500 == 0:
    #     print('Analyzed {} out of {} ({} %)'.format(koi_i, numKoi, koi_i / numKoi * 100))
    #     logging.info('Analyzed {} out of {} ({} %)'.format(koi_i, numKoi, koi_i / numKoi * 100))

    # get TCEs that are from the same target star and that were not already matched with a KOI
    candidateTcesTable = keplerTceTable.loc[(keplerTceTable['target_id'] == koi.kepid) &
                                            (keplerTceTable['kepoi_name'] != np.nan)]

    if len(candidateTcesTable) == 0:
        print('No candidate TCE found for KOI {}'.format(koi.kepoi_name))
        logging.info('No candidate TCE found for KOI {}'.format(koi.kepoi_name))
        koiNotMatched.append(koi.kepoi_name)
        koiNoTceInTarget.append(koi.kepoi_name)
        continue

    # create binary time-series for the KOI
    binTseriesKoi = create_binary_time_series(epoch=0,
                                              duration=koi.koi_duration / 24,
                                              period=koi.koi_period,
                                              tStart=0,
                                              tEnd=koi.koi_period,
                                              samplingInterval=samplingInterval)

    matchScore = {candidate.tce_plnt_num: np.nan for candidate_i, candidate in candidateTcesTable.iterrows()}
    binTseriesCandidateList = []
    ephemerisCandidateList = []
    # iterate through the candidate TCEs
    for candidate_i, candidate in candidateTcesTable.iterrows():

        ephemerisCandidateList.append([candidate.tce_time0bk, candidate.tce_period, candidate.tce_duration])
        # # method - 3D ephemeris data points
        # p2 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
        #                    (ephem_tbl['tce_plnt_num'] == tces_sector2[tce2_i][0]) &
        #                    (ephem_tbl['sector'] == paired_sector[1])][['transitDurationHours',
        #                                                                'orbitalPeriodDays',
        #                                                                'transitEpochBtjd']].values[0]
        #
        # e2 = p2[2]
        #
        # p2 = (p2 - min_ephem) / (max_ephem - min_ephem)
        #
        # epoch_shift = np.abs(e1 - e2) / per1
        # p2[-1] = np.abs(round(epoch_shift) - epoch_shift)
        #
        # matchscore_mat[tce1_i, tce2_i] = distance.cosine(p1, p2)
        #
        # datapoints[(tces_sector1[tce1_i], paired_sector[0], tces_sector2[tce2_i], paired_sector[1])] = (p1, p2)

        # method - template matching
        # epoch_shift = np.abs(koi.koi_time0bk - candidate.tce_time0bk) / koi.koi_period
        # e2 = np.abs(round(epoch_shift) - epoch_shift) * koi.koi_period
        e2 = find_first_epoch_after_this_time(candidate.tce_time0bk, candidate.tce_period, koi.koi_time0bk)
        e2 -= koi.koi_time0bk
        binTseriesCandidate = create_binary_time_series(epoch=0,  #e2,  # set epoch to 0 so that it does not take into account the epoch shift
                                                        duration=candidate.tce_duration / 24,
                                                        period=candidate.tce_period,
                                                        tStart=0,
                                                        tEnd=koi.koi_period,
                                                        samplingInterval=samplingInterval)
        binTseriesCandidateList.append(binTseriesCandidate)

        matchScore[(candidate.tce_plnt_num)] = distance.cosine(binTseriesKoi, binTseriesCandidate)

    matchedCandidate = min(matchScore, key=matchScore.get)

    # test if the best candidate has a score lower than the matching threshold
    if matchScore[matchedCandidate] >= matchThreshold:
        print('No candidate was matched to KOI {}'.format(koi.kepoi_name))
        logging.info('No candidate was matched to KOI {} - {}: {}'.format(koi.kepoi_name, koi.koi_tce_plnt_num,
                                                                          matchScore))
        logging.info('KOI ephemeris: {}, {}, {}'.format(koi.koi_time0bk, koi.koi_period, koi.koi_duration))
        for eph_i in range(len(ephemerisCandidateList)):
            logging.info('Candidate {}: {}'.format(eph_i, ephemerisCandidateList[eph_i]))
        matched = False
        koiNotMatched.append(koi.kepoi_name)
    else:
        matched = True

    # # plot candidates binary time-series against KOI
    # f, ax = plt.subplots()
    # ax.plot(binTseriesKoi, label='KOI {}|{}'.format(koi.kepoi_name, koi.koi_tce_plnt_num))
    # for i, binTseriesCandidate in enumerate(binTseriesCandidateList):
    #     ax.plot(binTseriesCandidate, label='Candidate {}'.format(list(matchScore.keys())[i]))
    # ax.set_title('Kepler ID {}\n{}'.format(koi.kepid, list(matchScore.values())))
    # ax.legend()
    # if koi.koi_tce_plnt_num != matchedCandidate:
    #     f.suptitle('Matched {}| different tce_plnt_num'.format(matched))
    # f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
    #           'koi_ephemeris_matching/plots/{}.png'.format(koi.kepoi_name))
    # plt.close()

    if not matched:
        continue

    # add KOI parameters to the matched TCE
    keplerTceTable.loc[(keplerTceTable['target_id'] == koi.kepid) &
                       (keplerTceTable['tce_plnt_num'] == matchedCandidate), koiColumnNames] = \
        koi[koiColumnNames].values

# print('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
logging.info('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
logging.info('Number of KOI not matched to any TCE in the same Kepler ID = {}'.format(len(koiNotMatched) -
                                                                                      len(koiNoTceInTarget)))
logging.info('Number of KOI without any TCE in the same Kepler ID = {}'.format(len(koiNoTceInTarget)))

# save updated TCE table with KOI parameters
keplerTceTable.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_koinoepochmatch_processed.csv',
                      index=False)

#%% Matching KOI with TCEs to add KOI fields such as FP flags using the ephemerides - optimal match

logging.basicConfig(filename='/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                             'koi_ephemeris_matching/koi_ephemeris_matching.log', filemode='a', format='%(message)s',
                    level=logging.INFO)

# Q1-Q17 DR25 TCE list
keplerTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_processed.csv',
                             header=0)

# Cumulative KOI list
# koiCumTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/koi_table/'
#                           'cumulative_2020.02.21_10.29.22.csv', header=90)
koiCumTable = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                          'koi_ephemeris_matching/oldvsnewkoidispositions.csv', header=0)

# # filter KOIs that do not come from Q1-Q17 DR25 TCE list
# koiCumTable = koiCumTable.loc[koiCumTable['koi_tce_delivname'] == 'q1_q17_dr25_tce']
#
# koiColumnNames = np.array(koiCumTable.columns.values.tolist())[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -25, -24, -21,
#                                                                 -20]]
koiColumnNames = koiCumTable.columns.values.tolist()

# initialize new columns
keplerTceTable = pd.concat([keplerTceTable, pd.DataFrame(columns=koiColumnNames)])

# matching threshold
matchThreshold = np.inf  # 0.25

# get Kepler IDs in the KOI table
keplerIdsinKoiTbl = np.unique(koiCumTable['kepid'].values)
numKepidsinKoiTbl = len(keplerIdsinKoiTbl)

samplingInterval = 1e-4

koiNotMatched, koiNoTceInTarget = [], []

# iterate through the Kepler IDs in the KOI table
for kepid_i, kepid in enumerate(keplerIdsinKoiTbl):

    if kepid_i % 100 == 0:
        print('Analyzed {} out of {} ({} %)'.format(kepid_i, numKepidsinKoiTbl, kepid_i / numKepidsinKoiTbl * 100))
        logging.info('Analyzed {} out of {} ({} %)'.format(kepid_i, numKepidsinKoiTbl,
                                                           numKepidsinKoiTbl / numKepidsinKoiTbl * 100))

    # get TCEs for this target star
    candidateTcesTable = keplerTceTable.loc[keplerTceTable['target_id'] == kepid]
    tcesList = list(candidateTcesTable['tce_plnt_num'].values)

    # get KOIs for this target star
    koisForKepid = koiCumTable.loc[koiCumTable['kepid'] == kepid]
    koisList = list(koisForKepid['kepoi_name'].values)

    # no TCEs found for this target star - no match for the KOI(s)
    if len(candidateTcesTable) == 0:
        print('No TCEs found for Kepler ID {}'.format(kepid))
        logging.info('No TCEs found for Kepler ID {}'.format(kepid))
        koiNotMatched += koisList
        koiNoTceInTarget += koisList
        continue

    # instantiate match score dictionary between KOIs and TCEs
    matchScores = {}
    for koi_i, koi in koisForKepid.iterrows():
        # create binary time-series for the KOI
        binTseriesKoi = create_binary_time_series(epoch=0,
                                                  duration=koi.koi_duration / 24,
                                                  period=koi.koi_period,
                                                  tStart=0,
                                                  tEnd=koi.koi_period,
                                                  samplingInterval=samplingInterval)
        ephemerisCandidateList = []
        binTseriesCandidateList = []
        for candidate_i, candidate in candidateTcesTable.iterrows():
            ephemerisCandidateList.append([candidate.tce_time0bk, candidate.tce_period, candidate.tce_duration])
            # # method - 3D ephemeris data points
            # p2 = ephem_tbl.loc[(ephem_tbl['tic'] == tic_id) &
            #                    (ephem_tbl['tce_plnt_num'] == tces_sector2[tce2_i][0]) &
            #                    (ephem_tbl['sector'] == paired_sector[1])][['transitDurationHours',
            #                                                                'orbitalPeriodDays',
            #                                                                'transitEpochBtjd']].values[0]
            #
            # e2 = p2[2]
            #
            # p2 = (p2 - min_ephem) / (max_ephem - min_ephem)
            #
            # epoch_shift = np.abs(e1 - e2) / per1
            # p2[-1] = np.abs(round(epoch_shift) - epoch_shift)
            #
            # matchscore_mat[tce1_i, tce2_i] = distance.cosine(p1, p2)
            #
            # datapoints[(tces_sector1[tce1_i], paired_sector[0], tces_sector2[tce2_i], paired_sector[1])] = (p1, p2)

            # method - template matching
            # epoch_shift = np.abs(koi.koi_time0bk - candidate.tce_time0bk) / koi.koi_period
            # e2 = np.abs(round(epoch_shift) - epoch_shift) * koi.koi_period
            e2 = find_first_epoch_after_this_time(candidate.tce_time0bk, candidate.tce_period, koi.koi_time0bk)
            e2 -= koi.koi_time0bk
            binTseriesCandidate = create_binary_time_series(epoch=e2,  # set epoch to 0 so that it does not take into account the epoch shift
                                                            duration=candidate.tce_duration / 24,
                                                            period=candidate.tce_period,
                                                            tStart=0,
                                                            tEnd=koi.koi_period,
                                                            samplingInterval=samplingInterval)

            matchScore = distance.cosine(binTseriesKoi, binTseriesCandidate)
            if np.isnan(matchScore):
                matchScore = 1
            matchScores[koi.kepoi_name, candidate.tce_plnt_num] = matchScore

            binTseriesCandidateList.append(binTseriesCandidate)

        # # plot candidates binary time-series against KOI
        # f, ax = plt.subplots()
        # ax.plot(binTseriesKoi, label='KOI {}'.format(koi.kepoi_name))
        # for i, binTseriesCandidate in enumerate(binTseriesCandidateList):
        #     ax.plot(binTseriesCandidate, label='Candidate {}'.format(tcesList[i]))
        # ax.set_title('Kepler ID {}\n{}'.format(kepid, [matchScores[koi.kepoi_name, tce] for tce in tcesList]))
        # ax.legend(loc=8)
        # f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
        #           'koi_ephemeris_matching/plots/{}.png'.format(koi.kepoi_name))
        # plt.close()

    # sort scores in increasing order
    sortedPairsbyMatchScore = sorted(matchScores, key=matchScores.get)

    # filter scores based on the matching threshold
    sortedPairsbyMatchScore = [pair for pair in sortedPairsbyMatchScore if matchScores[pair] <= matchThreshold]

    # pair KOIs and TCEs in increasing order of match score
    for pairKoi, pairTce in sortedPairsbyMatchScore:
        if pairKoi in koisList and pairTce in tcesList:  # check if pair is still in the unmatched lists
            # add KOI parameters to the matched TCE
            keplerTceTable.loc[(keplerTceTable['target_id'] == kepid) &
                               (keplerTceTable['tce_plnt_num'] == pairTce), koiColumnNames] = \
                koisForKepid.loc[koisForKepid['kepoi_name'] == pairKoi][koiColumnNames].values

            # remove pair from unmatched lists
            koisList.remove(pairKoi)
            tcesList.remove(pairTce)

            print('KOI {} matched to TCE {} for Kepler ID {}: match score = {}'.format(pairKoi, pairTce, kepid,
                                                                                       matchScores[(pairKoi, pairTce)]))
            logging.info('KOI {} matched to TCE {} for Kepler ID {}: match score = {}'.format(pairKoi,pairTce, kepid,
                                                                                              matchScores[(pairKoi,
                                                                                                           pairTce)]))
            logging.info('KOI ephemeris: \n{}'.format(
                koisForKepid.loc[koisForKepid['kepoi_name'] == pairKoi][['koi_time0bk',
                                                                         'koi_period',
                                                                         'koi_duration']].to_string(index=False)))
            logging.info('TCE ephemeris: \n{}'.format(
                candidateTcesTable.loc[(candidateTcesTable['target_id'] == kepid) &
                                       (candidateTcesTable['tce_plnt_num'] == pairTce)][['tce_time0bk',
                                                                                         'tce_period',
                                                                                         'tce_duration']].to_string(
                    index=False)))

    # non-matched KOIs
    koiNotMatched += koisList
    for nonmatchedKoi in koisList:
        print('No candidate TCE ({}) was matched to KOI {} in Kepler ID {}'.format(tcesList, nonmatchedKoi, kepid))
        logging.info('No candidate TCE ({}) was matched to KOI {} in Kepler ID {}'.format(tcesList, nonmatchedKoi,
                                                                                          kepid))
        logging.info('KOI ephemeris: \n{}'.format(
            koisForKepid.loc[koisForKepid['kepoi_name'] ==
                             nonmatchedKoi][['koi_time0bk', 'koi_period', 'koi_duration']].to_string(index=False)))
        logging.info('Non-matched TCE ephemeris: \n{}'.format(
            candidateTcesTable.loc[(candidateTcesTable['target_id'] == kepid) &
                                   (candidateTcesTable['tce_plnt_num'].isin(tcesList))][['tce_time0bk',
                                                                                         'tce_period',
                                                                                         'tce_duration']].to_string(
                index=False)))

# print('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
logging.info('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
logging.info('Number of KOI not matched to any TCE in the same Kepler ID = {}'.format(len(koiNotMatched) -
                                                                                      len(koiNoTceInTarget)))
logging.info('Number of KOI without any TCE in the same Kepler ID = {}'.format(len(koiNoTceInTarget)))

# save updated TCE table with KOI parameters
keplerTceTable.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                      'koioldKOIlist_processed.csv',
                      index=False)

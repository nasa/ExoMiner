"""
Script created to perform ephemeris matching related tests.
"""

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.spatial import distance
# import itertools
import os
from scipy.spatial import distance

from data_wrangling.tess.ephemeris_matching import create_binary_time_series, find_first_epoch_after_this_time

#%%
tic_id = 19271382

sector_list = [8, 9]

# # look for the fits files for this target TIC ID and for these sectors

# # light curve filepath for the target TIC ID
# light_curve_fp = '/data5/tess_project/Data/TESS_TOI_fits(MAST)/sector_8/' \
#                  'tess2019032160000-s0008-0000000019271382-0136-s_lc.fits'
# lc_data = fits.getdata(light_curve_fp, 1, header=True, memmap=False)[0]
#
# light_curve_fp2 = '/data5/tess_project/Data/TESS_TOI_fits(MAST)/sector_9/' \
#                   'tess2019058134432-s0009-0000000019271382-0139-s_lc.fits'
# lc_data2 = fits.getdata(light_curve_fp2, 1, header=True, memmap=False)[0]

# hdu_list[2].data['TIMECORR'], hdu_list[2].data['TIME']

tStart, tEnd = 0, 28.625  # 1500, 2500  # start and end times in TJD
deltaT = tEnd - tStart
# numSamplIntervals = 20000  # number of sampling intervals
numCadencesDeltaT = deltaT * 24 * 60 / 2
samplingInterval = deltaT / numCadencesDeltaT
# samplingInterval = (tEnd - tStart) / numSamplIntervals

# # get time array
# num_points = min(len(lc_data['TIME']), len(lc_data2['TIME']))
# time_arr = np.linspace(lc_data['TIME'][0] - lc_data['TIMECORR'][0], lc_data['TIME'][-1] - lc_data['TIMECORR'][-1],
#                        num=num_points)
# time_arr2 = np.linspace(lc_data2['TIME'][0] - lc_data2['TIMECORR'][0], lc_data2['TIME'][-1] - lc_data2['TIMECORR'][-1],
#                         num=num_points)
#
# # time_arr = lc_data['TIME']  # - lc_data['TIMECORR']
# # time_arr2 = lc_data2['TIME']  # - lc_data['TIMECORR']
#
# # f, ax = plt.subplots()
# # # ax.plot(time_arr - lc_data['TIMECORR'], c='r', label='Slice correction')
# # # ax.plot(lc_data['TIME'], label='Non-corrected')
# # ax.plot(lc_data['TIMECORR'])
# # ax.set_ylabel('Time [TJD]')
# # ax.set_xlabel('Sample number')

# get all TCEs for the target TIC ID in the two sectors
ephemeris_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv'
ephemeris_tbl = pd.read_csv(ephemeris_tbl_fp)

# get ephemeris for the paired objects
ephem_tblcmmn = ephemeris_tbl.loc[(ephemeris_tbl['tic'] == tic_id) &
                                  (ephemeris_tbl['sector'].isin(sector_list))][['sector',
                                                                                'transitDurationHours',
                                                                                'orbitalPeriodDays',
                                                                                'transitEpochBtjd']]

# convert period from hours to days
ephem_tblcmmn['transitDurationHours'] = ephem_tblcmmn['transitDurationHours'] / 24

# convert DataFrame to dictionary
ephem1, ephem2 = ephem_tblcmmn.to_dict(orient='records')
# ephem2 = ephem1

# adjust epoch so that it is the first mid-transit timestamp after the reference
ephem1['transitEpochBtjd'] = find_first_epoch_after_this_time(ephem1['transitEpochBtjd'], ephem1['orbitalPeriodDays'],
                                                              reference_time=tStart)
ephem2['transitEpochBtjd'] = find_first_epoch_after_this_time(ephem2['transitEpochBtjd'], ephem2['orbitalPeriodDays'],
                                                              reference_time=tStart)

# create binary time series
bintseries = create_binary_time_series(ephem1['transitEpochBtjd'], ephem1['transitDurationHours'],
                                       ephem1['orbitalPeriodDays'], tStart, tEnd, samplingInterval)
bintseries2 = create_binary_time_series(ephem2['transitEpochBtjd'], ephem2['transitDurationHours'],
                                        ephem2['orbitalPeriodDays'], tStart, tEnd, samplingInterval)

# # weight binary time series by number of total duration of transits
# bintseries = bintseries / np.sum(bintseries)
# bintseries2 = bintseries2 / np.sum(bintseries2)

# # compute inner product (xcorrelation at tau=0)
# xcorr0 = np.inner(bintseries, bintseries2)  # np.correlate(bintseries, bintseries2, mode='same')
# cmmn_idxs_fact = len(np.nonzero(bintseries * bintseries2)[0])
# xcorr0_fact = xcorr0 * cmmn_idxs_fact  # max correlation at one
match_score = distance.cosine(bintseries, bintseries2)

sampleTimes = np.linspace(tStart / samplingInterval, tEnd / samplingInterval, (tEnd - tStart) / samplingInterval,
                          endpoint=True)
f, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True, sharey=True)
ax.plot(bintseries, c='b')
ax.scatter(np.arange(len(bintseries)), bintseries, c='b')
# ax.set_title('Cross-correlation {}|{}'.format(xcorr0, xcorr0_fact))
ax.set_title('Cosine distance {}'.format(match_score))
ax.plot(np.arange(len(bintseries2)), bintseries2, c='r')
ax.scatter(np.arange(len(bintseries2)), bintseries2, c='r')
ax.set_xlabel('Number samples')

#%%

# TODO: plot transit duration histogram and see what is the minimum transit duration in days
#       plot time interval of the time arrays for each light curve
#       decide on a threshold

root_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

sect_dirs = [os.path.join(root_dir, sect_dir) for sect_dir in os.listdir(root_dir) if sect_dir.startswith('sector_')]

maxmin_time = {'min': [], 'max': []}
tinterval = {'timeTJD': 27.41, 'numSamples': 19737}

for sect_dir in sect_dirs:

    print('Sector: {}'.format(sect_dir))

    lc_fps = [os.path.join(sect_dir, lc_fp) for lc_fp in os.listdir(sect_dir)]

    for i_lc, lc_fp in enumerate(lc_fps):

        print('Reading light curve fits file {}/{} (Sector {})'.format(i_lc, len(lc_fps), sect_dir))

        try:
            lc_time = fits.getdata(lc_fp, 1, header=True, memmap=False)[0]['TIME']
            print(lc_time[0], lc_time[-1])
            maxmin_time['min'].append(lc_time[0])
            maxmin_time['max'].append(lc_time[-1])
        except:
            print('Could not read the FITS file...')
            continue

        # tinterval['timeTJD'].append(lc_time[-1] - lc_time[0])
        # tinterval['numSamples'].append(len(lc_time))

# f, ax = plt.subplots()
# ax.hist(tinterval['timeTJD'], bins='auto')
# ax.set_xlabel('Time interval [days]')
# ax.set_ylabel('Target counts')
#
# f, ax = plt.subplots()
# ax.hist(tinterval['numSamples'], bins='auto')
# ax.set_xlabel('Number of samples')
# ax.set_ylabel('Target counts')

#%%

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv')
tce_tbl['transitDurationHours'] /= 24  # convert to days

ephem_tbl = tce_tbl[['transitDurationHours', 'orbitalPeriodDays', 'transitEpochBtjd']]

print('Minimum values')
print(ephem_tbl.min())
print('Maximum values')
print(ephem_tbl.max())

ephem_tbl[['transitDurationHours', 'orbitalPeriodDays']].hist(bins=1000)
ephem_tbl['transitEpochBtjd'].hist(bins=100)

minTransitDuration = 0.007762  # days
maxTransitDuration = 5.757333  # days

tStart, tEnd = 1325.3, 1637.68 + 28.625  # 0, 28.625  # 1300, 1700  # BTJD
deltaTDesired = tEnd - tStart  # JD
print('Time interval desired = {}'.format(deltaTDesired))
# numCadencesSector = 20610
numCadencesDeltaT = deltaTDesired * 24 * 60 / 2
print('Number of cadences = {}'.format(numCadencesDeltaT))
samplingInterval = deltaTDesired / numCadencesDeltaT
print('Sampling interval = {}'.format(samplingInterval))
print('Number of samples for a minimum transit duration = {}'.format(minTransitDuration / samplingInterval))
print('Number of transits for a minimum transit duration = {}'.format(deltaTDesired / minTransitDuration))
print('Number of samples for a maximum transit duration = {}'.format(maxTransitDuration / samplingInterval))
print('Number of transits for a maximum transit duration = {}'.format(deltaTDesired / maxTransitDuration))

# # the sampling interval should be much smaller than the transit duration
# samplingIntervalDesired = minTransitDuration / 50
# print('Sampling interval desired = {}'.format(samplingIntervalDesired))
# deltaT = samplingInterval * numCadencesSector
# print('Time interval = {}'.format(deltaT))
# print('Number of samples for a minimum transit duration = {}'.format(minTransitDuration / samplingIntervalDesired))
# print('Number of transits for a maximum transit duration = {}'.format(deltaT / maxTransitDuration))

#%% Build binary time series for the TCEs

max_ephem = ephem_tbl[['transitDurationHours', 'orbitalPeriodDays', 'transitEpochBtjd']].max().values
min_ephem = ephem_tbl[['transitDurationHours', 'orbitalPeriodDays', 'transitEpochBtjd']].min().values

tStart, tEnd = 0, 27.4  # 1325.3, 1637.68 + 27.4  # 28.625  # start and end times in days
deltaT = tEnd - tStart
# number of cadences in the time interval based on TESS's sampling rate (2 mins)
numCadencesDeltaT = int(deltaT * 24 * 60 / 2)
samplingInterval = deltaT / numCadencesDeltaT

# get unique TIC IDs
tic_ids = ephem_tbl['tic'].unique()

# {tic_id1: {(old_tce_plnt_numi, sectori): [0-1] array, (old_tce_plnt_numj, sectorj):...}, tic_id2:...}
bintseries = {}
for tic_id in tic_ids:  # [150100106]:  # tic_ids:

    # if tic_id == 150100106:
    #     aaaaaa

    print('TIC ID = {}'.format(tic_id))

    bintseries[tic_id] = {}

    tic_tces = ephem_tbl.loc[ephem_tbl['tic'] == tic_id][['tce_plnt_num', 'sector', 'transitDurationHours',
                                                          'orbitalPeriodDays', 'transitEpochBtjd']]

    print('Found {} TCEs for this target'.format(len(tic_tces)))

    for tce_i, tce in tic_tces.iterrows():

        # print('TCE ', tce)
        # print('before {}'.format(tce['transitEpochBtjd']))
        # get epoch of the first transit after the start time
        tce['transitEpochBtjd'] = find_first_epoch_after_this_time(tce['transitEpochBtjd'],
                                                                   tce['orbitalPeriodDays'],
                                                                   reference_time=tStart)
        # print('after {}'.format(tce['transitEpochBtjd']))

        # create binary time series
        bintseries_tce = create_binary_time_series(tce['transitEpochBtjd'], tce['transitDurationHours'],
                                                   tce['orbitalPeriodDays'], tStart, tEnd, samplingInterval)
        # bintseries_tce = create_binary_time_series(0, tce['transitDurationHours'],
        #                                            tce['orbitalPeriodDays'], tStart, tEnd, samplingInterval)

        bintseries[tic_id][(tce['tce_plnt_num'], tce['sector'])] = bintseries_tce


# f, ax = plt.subplots()
# ax.plot(bintseries[150100106][(1, 4)], label='1s4')
# ax.plot(bintseries[150100106][(1, 7)], label='1s7')
# ax.plot(bintseries[150100106][(2, 7)], label='2s7')
# ax.legend()
# ax.set_title('TIC ID {}'.format(150100106))
# ax.set_xlabel('Sample number')
# ax.set_ylabel('Binary amplitude')

# f, ax = plt.subplots()
# for tce in bintseries[7624182]:
#     ax.plot(bintseries[7624182][tce], label='{}'.format(tce))
# ax.legend()
# ax.set_title('TIC ID {}'.format(7624182))
# ax.set_xlabel('Sample number')
# ax.set_ylabel('Binary amplitude')

# # plot some binary time series
# tic_ids_plot = np.random.choice(tic_ids, 100)
# for tic_id in tic_ids_plot:
#
#     tces_tic = list(bintseries[tic_id].keys())
#     tces_tic_plot = np.random.choice(len(tces_tic), min(len(tces_tic), 13))
#
#     f, ax = plt.subplots()
#     for tce_i in tces_tic_plot:
#         ax.plot(bintseries[tic_id][tces_tic[tce_i]])
#     f.suptitle('Number of tces {}'.format(len(tces_tic_plot)))
#     plt.savefig('/home/msaragoc/Downloads/binary_time_series_ephem/{}.svg'.format(tic_id))
#     plt.close()

#%% plot template binary time series for TCEs to be compared

sampinterv = 0.00001
tic_test = 382626661
tces_test = ((1, 1), (5, 1), (7, 1), (12, 1))  # first one is considered as the one to be compared against

f, ax = plt.subplots()
ax.set_title('TIC {}'.format(tic_test))
for tce_i in range(len(tces_test)):

    p = ephem_tbl.loc[(ephem_tbl['tic'] == tic_test) &
                      (ephem_tbl['sector'] == tces_test[tce_i][0]) &
                      (ephem_tbl['tce_plnt_num'] == tces_test[tce_i][1])][['transitDurationHours',
                                                                           'orbitalPeriodDays',
                                                                           'transitEpochBtjd']].values[0]

    if tce_i == 0:
        bintseries = create_binary_time_series(epoch=0,
                                               duration=p[0],
                                               period=p[1],
                                               tStart=0,
                                               tEnd=p[1],
                                               samplingInterval=sampinterv)
        bintseries1 = bintseries
        per1 = p[1]
        epoch1 = p[2]
    else:
        epoch_shift = np.abs(epoch1 - p[2]) / per1
        e2 = np.abs(round(epoch_shift) - epoch_shift) * per1
        bintseries = create_binary_time_series(epoch=e2,
                                           duration=p[0],
                                           period=p[1],
                                           tStart=0,
                                           tEnd=per1,
                                           samplingInterval=sampinterv)

        matchscore = distance.cosine(bintseries1, bintseries)
        print('Match score for {}: {}'.format(tces_test[tce_i], matchscore))

        print('Jaccardi: {}'.format(distance.jaccard(bintseries1, bintseries)))
        print('Dice: {}'.format(distance.dice(bintseries1, bintseries)))
        print('Hamming: {}'.format(distance.hamming(bintseries1, bintseries)))

    ax.plot(bintseries, label=tces_test[tce_i])

ax.set_ylabel('Binary amplitude')
ax.set_xlabel('Sample number \n(Sampling interval (d) = {})'.format(sampinterv))
ax.legend()

#%%

dr25Koi = pd.read_csv('/home/msaragoc/Downloads/cumulative_2020.02.06_16.03.16.csv', header=85)
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors.csv')

addFields = ['koi_pdisposition', 'koi_score', 'koi_fpflag', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
tceTbl[addFields] = \
    pd.DataFrame(np.nan * np.ones((len(tceTbl), 6)), index=tceTbl.index)
for i, row in dr25Koi.iterrow():

    tcesTarget = tceTbl.loc[tceTbl['kepid'] == row.kepid][['tce_plnt_num']]

    if len(tcesTarget) == 0:
        print('Kepid {} not found in TCE table.'.format(row.kepid))
    elif len(tcesTarget) == 1:

    else:
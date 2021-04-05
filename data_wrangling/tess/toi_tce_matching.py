""" Script used to match TOIs to TCEs from different single- and multi-sector runs. """

# 3rd party
from pathlib import Path
import pandas as pd
from scipy.spatial import distance
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# local
from data_wrangling.old.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time

#%% Matching between TOIs and TCEs using the cosine distance between period templates

res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
res_dir.mkdir(exist_ok=True)

toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/3-11-2021/')

toi_tbl = pd.read_csv(toi_dir / f'exofop_toilists_spoc_nomissingpephem.csv')

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'}
singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

map_tce_tbl_names = {'<=30': {'period': 'orbitalPeriodDays', 'epoch': 'transitEpochBtjd', 'duration': 'transitDurationHours'},
                     '>30': {'period': 'tce_period', 'epoch': 'tce_time0bt', 'duration': 'tce_duration'},
                     }

match_thr = np.inf  # np.inf  # 0.25
sampling_interval = 0.00001  # approximately 1 min

# toi_tce_match = {}
# cols = [
    # 'orbitalPeriodDays',
    # 'transitDepthPpm',
    # 'transitDurationHours',
    # 'transitEpochBtjd',
    # 'ws_mes',
    # 'ws_mesphase',
    # 'mes'
# ]
max_num_tces = len(singlesector_tce_tbls) + len(multisector_tce_tbls)
matching_tbl = pd.DataFrame(columns=['Full TOI ID', 'TIC', 'Matched TCEs'] + [f'matching_dist_{i}'
                                                                              for i in range(max_num_tces)])
for toi_i, toi in toi_tbl.iterrows():

    if toi_i % 50 == 0:
        print(f'Matched {toi_i} ouf of {len(toi_tbl)}...')

    # if toi['Full TOI ID'] == 137.01:
    #     aaaa
    # else:
    #     continue

    # toi_tce_match[toi['Full TOI ID']] = {}

    # toi_tceid = '{}-{}'.format(f'{toi["TIC"]}'.zfill(11), f'{toi["Signal ID"]}'.zfill(2))

    # toi_sectors = [int(sector) for sector in toi['Sectors'].split(' ')]
    toi_sectors = [int(sector) for sector in toi['Sectors'].split(',')]

    # toi_bin_ts = create_binary_time_series(epoch=toi['Epoch Value'],
    #                                        duration=toi['Transit Duration Value'] / 24,
    #                                        period=toi['Orbital Period Value'],
    #                                        tStart=toi['Epoch Value'],
    #                                        tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
    #                                        samplingInterval=sampling_interval)
    toi_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                           duration=toi['Duration (hours)'] / 24,
                                           period=toi['Period (days)'],
                                           tStart=toi['Epoch (TBJD)'],
                                           tEnd=toi['Epoch (TBJD)'] + toi['Period (days)'],
                                           samplingInterval=sampling_interval)

    matching_dist_dict = {}

    for toi_sector in toi_sectors:

        if toi_sector <= 30:
            tce_ephem_name = map_tce_tbl_names['<=30']
        else:
            tce_ephem_name = map_tce_tbl_names['>30']

        # check the single sector run table
        tce_tbl_aux = singlesector_tce_tbls[toi_sector]
        # tce_found = tce_tbl_aux.loc[tce_tbl_aux['tceid'] == toi_tceid]
        # tce_found = tce_tbl_aux.loc[tce_tbl_aux['ticid'] == toi['TIC']]
        tce_found = tce_tbl_aux.loc[tce_tbl_aux['ticid'] == toi['TIC ID']]

        if len(tce_found) > 0:

            for tce_i, tce in tce_found.iterrows():
                tceid = int(tce['tceid'].split('-')[1])

                # epoch_aux = find_nearest_epoch_to_this_time(tce['transitEpochBtjd'],
                #                                             tce['orbitalPeriodDays'],
                #                                             toi['Epoch Value'])
                epoch_aux = find_nearest_epoch_to_this_time(tce[tce_ephem_name['epoch']],
                                                            tce[tce_ephem_name['period']],
                                                            toi['Epoch (TBJD)'])
                # epoch_shift = np.abs(p1[2] - p2[2]) / p1[1]
                # e2 = np.abs(round(epoch_shift) - epoch_shift) * p1[1]
                # tce_bin_ts = create_binary_time_series(epoch=epoch_aux,
                #                                        duration=tce['transitDurationHours'] / 24,
                #                                        period=tce['orbitalPeriodDays'],
                #                                        tStart=toi['Epoch Value'],
                #                                        tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
                #                                        samplingInterval=sampling_interval)
                tce_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                                       duration=tce[tce_ephem_name['duration']] / 24,
                                                       period=tce[tce_ephem_name['period']],
                                                       tStart=toi['Epoch (TBJD)'],
                                                       tEnd=toi['Epoch (TBJD)'] + toi['Period (days)'],
                                                       samplingInterval=sampling_interval)

                match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                if match_distance < match_thr:
                    # toi_tce_match[toi['Full TOI ID']][toi_sector] = tce_found
                    matching_dist_dict[f'{toi_sector}_{tceid}'] = match_distance

        # check the multi sector runs tables
        for multisector_tce_tbl in multisector_tce_tbls:
            if toi_sector >= multisector_tce_tbl[0] and toi_sector <= multisector_tce_tbl[1]:
                tce_tbl_aux = multisector_tce_tbls[multisector_tce_tbl]
                # tce_found = tce_tbl_aux.loc[tce_tbl_aux['tceid'] == toi_tceid]
                # tce_found = tce_tbl_aux.loc[tce_tbl_aux['ticid'] == toi['TIC']]
                tce_found = tce_tbl_aux.loc[tce_tbl_aux['ticid'] == toi['TIC ID']]

                if len(tce_found) > 0:
                    for tce_i, tce in tce_found.iterrows():

                        tceid = int(tce['tceid'].split('-')[1])

                        # epoch_aux = find_nearest_epoch_to_this_time(tce['transitEpochBtjd'],
                        #                                             tce['orbitalPeriodDays'],
                        #                                             toi['Epoch Value'])
                        epoch_aux = find_nearest_epoch_to_this_time(tce['transitEpochBtjd'],
                                                                    tce['orbitalPeriodDays'],
                                                                    toi['Epoch (TBJD)'])
                        # epoch_shift = np.abs(p1[2] - p2[2]) / p1[1]
                        # e2 = np.abs(round(epoch_shift) - epoch_shift) * p1[1]
                        # tce_bin_ts = create_binary_time_series(epoch=epoch_aux,
                        #                                        duration=tce['transitDurationHours'] / 24,
                        #                                        period=tce['orbitalPeriodDays'],
                        #                                        tStart=toi['Epoch Value'],
                        #                                        tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
                        #                                        samplingInterval=sampling_interval)
                        tce_bin_ts = create_binary_time_series(epoch=epoch_aux,
                                                               duration=tce['transitDurationHours'] / 24,
                                                               period=tce['orbitalPeriodDays'],
                                                               tStart=toi['Epoch (TBJD)'],
                                                               tEnd=toi['Epoch (TBJD)'] + toi['Period (days)'],
                                                               samplingInterval=sampling_interval)

                        # TODO: add epsilon to avoid nan value when one of the vectors is zero?
                        match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                        if match_distance < match_thr:
                            # toi_tce_match[toi['Full TOI ID']][(multisector_tce_tbl[0], multisector_tce_tbl[1])] = \
                            #     tce_found[cols]
                            matching_dist_dict[f'{multisector_tce_tbl[0]}-{multisector_tce_tbl[1]}_{tceid}'] = \
                                match_distance

    matching_dist_dict = {k: v for k, v in sorted(matching_dist_dict.items(), key=lambda x: x[1])}
    # data_to_tbl = {'Full TOI ID': [toi['Full TOI ID']], 'TIC': [toi['TIC']],
    #                'Matched TCEs': ' '.join(list(matching_dist_dict.keys()))}
    data_to_tbl = {'Full TOI ID': [toi['TOI']], 'TIC': [toi['TIC ID']],
                   'Matched TCEs': ' '.join(list(matching_dist_dict.keys()))}
    matching_dist_arr = list(matching_dist_dict.values())
    for i in range(max_num_tces):
        if i < len(matching_dist_arr):
            data_to_tbl[f'matching_dist_{i}'] = [matching_dist_arr[i]]
        else:
            data_to_tbl[f'matching_dist_{i}'] = [np.nan]
    matching_tbl = pd.concat([matching_tbl, pd.DataFrame(data=data_to_tbl)], axis=0)

matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
                    index=False)

#%% Plot histograms for matching distance

bins = np.linspace(0, 1, 21, endpoint=True)
f, ax = plt.subplots()
ax.hist(matching_tbl['matching_dist_0'], bins, edgecolor='k')
ax.set_yscale('log')
ax.set_xlabel('Matching distance')
ax.set_ylabel('Counts')
ax.set_xlim([0, 1])
f.savefig(res_dir / 'hist_matching_dist_0.png')

f, ax = plt.subplots()
ax.hist(matching_tbl['matching_dist_0'], bins, edgecolor='k', cumulative=True, density=True)
# ax.set_yscale('log')
ax.set_xlabel('Matching distance')
ax.set_ylabel('Normalized Cumulative Counts')
ax.set_xlim([0, 1])
f.savefig(res_dir / 'cumhist_norm_matching_dist_0.png')

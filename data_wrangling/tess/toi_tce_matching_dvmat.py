""" Script used to match TOIs to TCEs from different single- and multi-sector runs DV mat files. """

# 3rd party
import multiprocessing
from pathlib import Path
import pandas as pd
from scipy.spatial import distance
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# local
from data_wrangling.old.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time

#%%


def match_toi_tce(toi, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces):

    toi_sectors = [int(sector) for sector in toi['Sectors'].split(',')]

    matching_dist_dict = {}

    toi_bin_ts_max = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                           duration=toi['Duration (hours)'] / 24,
                                           period=toi['Period (days)'],
                                           tStart=toi['Epoch (TBJD)'],
                                           tEnd=toi['Epoch (TBJD)'] + toi['Period (days)'],
                                           samplingInterval=sampling_interval)

    for toi_sector in toi_sectors:

        # check the single sector run table
        tce_tbl_aux = singlesector_tce_tbls[toi_sector]

        tce_found = tce_tbl_aux.loc[tce_tbl_aux['catId'] == toi['TIC ID']]

        if len(tce_found) > 0:

            for tce_i, tce in tce_found.iterrows():

                tceid = tce['planetIndexNumber']

                phase = max(toi['Period (days)'], tce['allTransitsFit_orbitalPeriodDays_value'])

                # find phase difference of TCE to the TOI
                phase_diff = 0  # find_nearest_epoch_to_this_time(tce['allTransitsFit_transitEpochBtjd_value'],
                # tce['allTransitsFit_orbitalPeriodDays_value'],
                # toi['Epoch (TBJD)'])

                if phase == toi['Period (days)']:
                    toi_bin_ts = toi_bin_ts_max
                else:
                    toi_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                                           duration=toi['Duration (hours)'] / 24,
                                                           period=toi['Period (days)'],
                                                           tStart=toi['Epoch (TBJD)'],
                                                           tEnd=toi['Epoch (TBJD)'] + phase,
                                                           samplingInterval=sampling_interval)

                tce_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'] + phase_diff,
                                                       duration=tce['allTransitsFit_transitDurationHours_value'] / 24,
                                                       period=tce['allTransitsFit_orbitalPeriodDays_value'],
                                                       tStart=toi['Epoch (TBJD)'],
                                                       tEnd=toi['Epoch (TBJD)'] + phase,
                                                       samplingInterval=sampling_interval)

                match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                if match_distance < match_thr:
                    matching_dist_dict[f'{toi_sector}_{tceid}'] = match_distance

        # check the multi sector runs tables
        for multisector_tce_tbl in multisector_tce_tbls:
            if toi_sector >= multisector_tce_tbl[0] and toi_sector <= multisector_tce_tbl[1]:

                tce_tbl_aux = multisector_tce_tbls[multisector_tce_tbl]

                tce_found = tce_tbl_aux.loc[tce_tbl_aux['catId'] == toi['TIC ID']]

                if len(tce_found) > 0:
                    for tce_i, tce in tce_found.iterrows():

                        tceid = tce['planetIndexNumber']

                        phase = max(toi['Period (days)'], tce['allTransitsFit_orbitalPeriodDays_value'])

                        # find phase difference of TCE to the TOI
                        phase_diff = 0  # find_nearest_epoch_to_this_time(tce['allTransitsFit_transitEpochBtjd_value'],
                        # tce['allTransitsFit_orbitalPeriodDays_value'],
                        # toi['Epoch (TBJD)'])

                        if phase == toi['Period (days)']:
                            toi_bin_ts = toi_bin_ts_max
                        else:
                            toi_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'],
                                                               duration=toi['Duration (hours)'] / 24,
                                                               period=toi['Period (days)'],
                                                               tStart=toi['Epoch (TBJD)'],
                                                               tEnd=toi['Epoch (TBJD)'] + phase,
                                                               samplingInterval=sampling_interval)

                        tce_bin_ts = create_binary_time_series(epoch=toi['Epoch (TBJD)'] + phase_diff,
                                                               duration=tce['allTransitsFit_transitDurationHours_value']
                                                                        / 24,
                                                               period=tce['allTransitsFit_orbitalPeriodDays_value'],
                                                               tStart=toi['Epoch (TBJD)'],
                                                               tEnd=toi['Epoch (TBJD)'] + phase,
                                                               samplingInterval=sampling_interval)

                        # TODO: add epsilon to avoid nan value when one of the vectors is zero?
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
                        sampling_interval, max_num_tces):

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

# %% Matching between TOIs and TCEs using the cosine distance between period templates

if __name__ == '__main__':

    # set results directory
    res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                   f'{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    res_dir.mkdir(exist_ok=True)

    # get TOI table
    toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-12-2021/')

    toi_cols = ['TOI', 'TIC ID', 'Sectors', 'Period (days)', 'Duration (hours)', 'Epoch (TBJD)']
    toi_tbl = pd.read_csv(toi_dir / f'exofop_toilists_nomissingpephem.csv', usecols=toi_cols)

    # get DV TCE tables for single- and multi-sector runs
    tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

    multisector_tce_dir = tce_root_dir / 'multi-sector' / 'csv_tables'
    singlesector_tce_dir = tce_root_dir / 'single-sector' / 'csv_tables'

    tce_cols = ['catId', 'planetIndexNumber', 'allTransitsFit_orbitalPeriodDays_value',
                'allTransitsFit_transitDurationHours_value', 'allTransitsFit_transitEpochBtjd_value']
    multisector_tce_tbls = {(int(file.stem[14:16]), int(file.stem[16:18])): pd.read_csv(file, usecols=tce_cols)
                            for file in multisector_tce_dir.iterdir()}
    singlesector_tce_tbls = {int(file.stem[14:16]): pd.read_csv(file, usecols=tce_cols)
                             for file in singlesector_tce_dir.iterdir()}

    # set matching threshold and sampling interval
    match_thr = np.inf  # np.inf  # 0.25
    sampling_interval = 0.00001  # approximately 1 min

    max_num_tces = len(singlesector_tce_tbls) + len(multisector_tce_tbls)

    match_tbl_cols = ['TOI ID', 'TIC', 'Matched TCEs'] + [f'matching_dist_{i}' for i in range(max_num_tces)]
    n_processes = 15
    tbl_jobs = np.array_split(toi_tbl, n_processes)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tbl_job.reset_index(inplace=False), tbl_job_i) +
            (match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces)
            for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
    async_results = [pool.apply_async(match_set_tois_tces, job) for job in jobs]
    pool.close()

    matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)

    matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
                        index=False)

    # %% Plot histograms for matching distance

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

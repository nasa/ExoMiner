""" Using TESS EB catalog provided by Jon to match to SPOC DV TCEs. """

# 3rd party
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
import numpy as np
import multiprocessing
import os
from datetime import datetime
import logging
import time

# local
from data_wrangling.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


# #%%
#
# def match_event_a_to_events_b(event_a,  events_b_tbl, match_thr, sampling_interval, max_num_events_matched):
#
#     event_a_bin_ts_max = create_binary_time_series(epoch=event_a['epoch'],
#                                                duration=event_a['duration'] / 24,
#                                                period=event_a['period'],
#                                                tStart=event_a['epoch'],
#                                                tEnd=event_a['epoch'] + event_a['period'],
#                                                samplingInterval=sampling_interval)
#
#     events_b_found = events_b_tbl.loc[events_b_tbl['target_id'] == event_a['target_id']]
#
#     if len(events_b_found) > 0:
#
#         for event_b_i, event_b in events_b_found.iterrows():
#
#             event_b_id = f'{event_b["tce_plnt_num"]}_{event_b["sector_run"]}'
#
#             # size of the template is set to larger orbital period between TOI and TCE
#             phase = max(event_a['period'], event_b_i['period'])
#
#             # find phase difference of TCE to the TOI
#             closest_tce_epoch = find_nearest_epoch_to_this_time(
#                 event_b['epoch'],
#                 event_b['period'],
#                 event_a['epoch']
#             )
#
#             if phase == event_a['period']:  # if TOI orbital period is larger, keep the TOI template
#                 event_a_bin_ts = event_a_bin_ts_max
#             else:
#                 event_a_bin_ts = create_binary_time_series(epoch=event_a['epoch'],
#                                                        duration=event_a['duration'] / 24,
#                                                        period=event_a['period'],
#                                                        tStart=event_a['epoch'],
#                                                        tEnd=event_a['epoch'] + phase,
#                                                        samplingInterval=sampling_interval)
#
#             event_b_bin_ts = create_binary_time_series(epoch=closest_tce_epoch,
#                                                    duration=event_b['duration'] / 24,
#                                                    period=event_b['period'],
#                                                    tStart=event_a['epoch'],
#                                                    tEnd=event_a['epoch'] + phase,
#                                                    samplingInterval=sampling_interval)
#
#             # compute distance between TOI and TCE templates as cosine distance
#             if event_a_bin_ts.sum() == 0 or event_b_bin_ts.sum() == 0:
#                 match_distance = 1
#             else:
#                 match_distance = distance.cosine(event_a_bin_ts, event_b_bin_ts)
#
#             # set matching distance if it is smaller than the matching threshold
#             if match_distance < match_thr:
#                 matching_dist_dict[f'{event_b_id}'] = match_distance
#
#     # sort TCEs based on matching distance
#     matching_dist_dict = {k: v for k, v in sorted(matching_dist_dict.items(), key=lambda x: x[1])}
#
#     # add TOI row to the csv matching file
#     data_to_tbl = {
#                    'target_id': event_a['target_id'],
#                    'signal_id': event_a['signal_id'],
#                    'matched_events': ' '.join(list(matching_dist_dict.keys()))
#     }
#     matching_dist_arr = list(matching_dist_dict.values())
#     data_to_tbl.update({f'matching_dist_{i}': matching_dist_arr[i] if i < len(matching_dist_arr) else np.nan
#                         for i in range(max_num_events_matched)})
#
#     return data_to_tbl
#
#
# def match_events(events_a_tbl, tbl_i, match_tbl_cols, events_b_tbl, match_thr, sampling_interval, max_num_tces, res_dir):
#
#     matching_tbl = pd.DataFrame(columns=match_tbl_cols, data=np.zeros((len(events_a_tbl), len(match_tbl_cols))))
#
#     for event_a_i, event_a in events_a_tbl.iterrows():
#         print(f'[Matching Subtable {tbl_i}] Matching TOI {toi["TOI"]} ({toi_i + 1}/{len(toi_tbl)})')
#
#         match_toi_row = match_event_a_to_events_b(event_a, events_b_tbl, match_thr, sampling_interval, max_num_tces)
#
#         matching_tbl.loc[toi_i] = pd.Series(match_toi_row)
#
#         print(f'[Matching Subtable {tbl_i}] Matched TOI {event_a["signal_id"]} ({event_a_i + 1}/{len(events_a_tbl)})')
#
#     print(f'[Matching Subtable {tbl_i}] Finished matching {len(event_a)} events')
#
#     # matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}_'
#     #                               f'{tbl_i}.csv', index=False)
#
#     return matching_tbl
#
#
# #%%
#
# tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/9-14-2021/tess_tces_s1-s40_09-14-2021_1754_stellarparams_updated_tfopwg_disp.csv')
# tce_tbl = pd.read_csv(tce_tbl_fp)
#
# eb_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TESS_EB_Catalog_1.csv')
#
# tce_tbl['signal_id'] = tce_tbl[['tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["tce_plnt_num"]}_{x["sector_run"]}')
# tce_tbl.rename(columns={'tce_period': 'period', 'tce_duration': 'duration', 'tce_epoch': 'epoch'}, inplace=True)
#
# # set matching threshold and sampling interval
# # match_thr = np.inf  # np.inf  # 0.25
# sampling_interval = 0.00001  # approximately 1 min
#
# # maximum number of TCEs associated with a TOI
# max_num_matched_events = 50
#
# match_tbl_cols = ['target_id', 'signal_id', 'matched_events'] + [f'matching_dist_{i}' for i in range(max_num_matched_events)]
# n_processes = 15
# tbl_jobs = np.array_split(toi_tbl, n_processes)
# pool = multiprocessing.Pool(processes=n_processes)
# jobs = [(tbl_job.reset_index(inplace=False), tbl_job_i) +
#         (match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces,
#          res_dir)
#         for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
# async_results = [pool.apply_async(match_set_tois_tces, job) for job in jobs]
# pool.close()
#
# matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)
#
# # matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
# #                     index=False)

# %%


def match_tces_to_ebs(matching_tbl, eb_tbl, sampling_interval, res_dir):
    proc_id = os.getpid()

    logger = logging.getLogger(name=f'match_tces_to_ebs_{proc_id}')
    logger_handler = logging.FileHandler(filename=res_dir / f'match_tces_to_ebs_{proc_id}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{proc_id}, {datetime.now().strftime("%m-%d-%Y_%H%M")}] Starting processing of {len(matching_tbl)} '
                f'TCEs...')

    for tce_i, tce in matching_tbl.iterrows():

        logger.info(f'[{proc_id}, {datetime.now().strftime("%m-%d-%Y_%H%M")}] '
                    f'Processing TCE {tce["target_id"]}-{tce["tce_plnt_num"]} run {tce["sector_run"]}.')

        if tce_i % 50 == 0:
            print(f'[{proc_id}, {datetime.now().strftime("%m-%d-%Y_%H%M")}] Processed {tce_i + 1} out of '
                  f'{len(matching_tbl)} TCEs.')

        ebs_in_tic = eb_tbl.loc[eb_tbl['tess_id'] == tce['target_id']].reset_index()

        if len(ebs_in_tic) == 0:
            continue

        for eb_i, eb in ebs_in_tic.iterrows():

            phase = max(tce['tce_period'], eb['period'])

            # find phase difference of EB to the TCE
            closest_tce_epoch = find_nearest_epoch_to_this_time(
                eb['bjd0'],
                eb['period'],
                tce['tce_time0bk']
            )

            tce_bin_ts = create_binary_time_series(epoch=tce['tce_time0bk'],
                                                   duration=tce['tce_duration'] / 24,
                                                   period=tce['tce_period'],
                                                   tStart=tce['tce_time0bk'] - phase / 2,
                                                   tEnd=tce['tce_time0bk'] + phase / 2,
                                                   samplingInterval=sampling_interval)

            eb_bin_ts = create_binary_time_series(epoch=closest_tce_epoch,
                                                  duration=tce['tce_duration'] / 24,
                                                  period=eb['period'],
                                                  tStart=tce['tce_time0bk'] - phase / 2,
                                                  tEnd=tce['tce_time0bk'] + phase / 2,
                                                  samplingInterval=sampling_interval)

            # compute distance between TOI and TCE templates as cosine distance
            if tce_bin_ts.sum() == 0 or eb_bin_ts.sum() == 0:
                match_distance = 1
            else:
                match_distance = distance.cosine(tce_bin_ts, eb_bin_ts)

            matching_tbl.loc[tce_i, [f'{el}_{eb_i}' for el in eb_tbl_cols[1:]]] = eb[eb_tbl_cols[1:]].values
            matching_tbl.loc[tce_i, [f'match_dist_eb_{eb_i}', f'tce_eb_period_ratio_{eb_i}']] = [match_distance,
                                                                                                 tce['tce_period'] / eb[
                                                                                                     'period']]

            if tce['tce_period'] >= eb['period']:
                # tce_eb_per_rat = tce['tce_period'] / eb['period']
                # tce_eb_per_k = np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat)
                tce_eb_per_k = np.round(tce['tce_period'] / eb['period'])
            else:
                # tce_eb_per_rat = eb['period'] / tce['tce_period']
                # tce_eb_per_k = -1 * (np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat))
                tce_eb_per_k = -1 * np.round(eb['period'] / tce['tce_period'])

            matching_tbl.loc[tce_i, f'tce_eb_period_multiple_int_{eb_i}'] = tce_eb_per_k
            # aaa

    print(f'[{proc_id}, {datetime.now().strftime("%m-%d-%Y_%H%M")}] Finished processing {len(matching_tbl)} TCEs.')

    matching_tbl.to_csv(res_dir / f'matching_tbl_{proc_id}.csv', index=False)

    # return matching_tbl


if __name__ == "__main__":

    res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/'
                   f'tess_tce_eb_match/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    res_dir.mkdir(exist_ok=True)

    sampling_interval = 0.00001  # approximately 1 min

    tce_tbl_fp = Path(
        '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage.csv')
    tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration',
                    'match_dist',
                    'TFOPWG Disposition', 'TESS Disposition', 'TOI']
    tce_tbl = pd.read_csv(tce_tbl_fp)[tce_tbl_cols]
    # tce_tbl = tce_tbl.loc[tce_tbl['target_id'] == 1129033]
    # tce_tbl = tce_tbl.loc[(tce_tbl['target_id'] == 49899799) & (tce_tbl['tce_plnt_num'] == 1) & (tce_tbl['sector_run'] == '31')]
    tce_tbl.to_csv(res_dir / tce_tbl_fp.name, index=False)

    eb_tbl_cols = ['tess_id', 'signal_id', 'bjd0', 'period']
    eb_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TESS_EB_Catalog_1.csv')[eb_tbl_cols]
    for col in ['period', 'bjd0']:
        eb_tbl = eb_tbl.loc[eb_tbl[col] != 'None']
    eb_tbl = eb_tbl.astype(dtype={'period': np.float, 'bjd0': 'float'})
    eb_tbl.to_csv(res_dir / 'eb_tbl.csv', index=False)

    matching_tbl = tce_tbl.copy(deep=True)

    max_num_matched_ebs = 2
    match_tbl_eb_cols_pat = eb_tbl_cols[1:] + ['match_dist_eb', 'tce_eb_period_ratio', 'tce_eb_period_multiple_int']
    match_tbl_eb_cols = []
    for i in range(max_num_matched_ebs):
        match_tbl_eb_cols.extend([f'{el}_{i}' for el in match_tbl_eb_cols_pat])
    matching_tbl = pd.concat([matching_tbl.reset_index(drop=True),
                              pd.DataFrame(data=np.nan * np.zeros((len(matching_tbl), len(match_tbl_eb_cols))),
                                           columns=match_tbl_eb_cols)], axis=1)

    n_processes = 15
    tbl_jobs = np.array_split(matching_tbl, n_processes)
    tbl_jobs = [tbl_job.reset_index(drop=True) for tbl_job in tbl_jobs if len(tbl_job) > 0]
    n_processes = len(tbl_jobs)
    print(f'Setting number of processes to {n_processes}')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tbl_job,) +
            (eb_tbl, sampling_interval, res_dir)
            for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
    async_results = [pool.apply_async(match_tces_to_ebs, job) for job in jobs]
    pool.close()

    print('Finished matching TCEs to EBs.')

    # time.sleep(5)
    _ = [match_tbl_job.get() for match_tbl_job in async_results]
    # matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)
    matching_tbl = pd.concat([pd.read_csv(fp) for fp in res_dir.iterdir() if fp.stem.startswith('matching_tbl_')],
                             axis=0)
    matching_tbl.to_csv(res_dir / 'matching_tbl.csv', index=False)

    # consider only matches below threshold, when the period is similar (no multiple integer) and for the smallest
    # planet number

    matching_eb_thr = 0.2  # matching threshold

    matching_tbl['target_sector_run'] = matching_tbl['target_id'].astype(str)
    matching_tbl['target_sector_run'] = matching_tbl[['target_sector_run', 'sector_run']].agg('_'.join, axis=1)

    matching_tbl['match_dist_eb_0_sngl'] = 2
    matching_tbl['match_dist_eb_1_sngl'] = 2
    for target_sector_run in matching_tbl['target_sector_run'].unique():

        for eb_i in [0, 1]:
            # get TCEs for a given TIC and sector run that have same period as EB and are below the matching threshold
            tces_in_target_sector_run = (matching_tbl['target_sector_run'] == target_sector_run) & \
                                        (matching_tbl[f'tce_eb_period_multiple_int_{eb_i}'] == 1) & \
                                        (matching_tbl[f'match_dist_eb_{eb_i}'] <= matching_eb_thr)
            tces_found = matching_tbl.loc[tces_in_target_sector_run]

            if len(tces_found) == 0:
                continue
            elif len(tces_found) == 1:
                matching_tbl.loc[tces_in_target_sector_run, f'match_dist_eb_{eb_i}_sngl'] = \
                    matching_tbl.loc[tces_in_target_sector_run, f'match_dist_eb_{eb_i}']
            else:  # only match to EB the TCE with smallest planet number
                idx_tce_chosen = tces_found['tce_plnt_num'].idxmin()
                matching_tbl.loc[idx_tce_chosen, f'match_dist_eb_{eb_i}_sngl'] = \
                    matching_tbl.loc[idx_tce_chosen, f'match_dist_eb_{eb_i}']

    # select the closest eb
    eb_final_cols = ['eb_signal_id', 'eb_bjd0', 'eb_period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
                     'eb_match_dist']
    eb_chosen_cols = ['signal_id', 'bjd0', 'period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
                      'match_dist_eb']
    matching_tbl = pd.concat([matching_tbl,
                              pd.DataFrame(data=np.nan * np.ones((len(matching_tbl), len(eb_final_cols))),
                                           columns=eb_final_cols)], axis=1)
    eb_i_match = matching_tbl[[f'match_dist_eb_{eb_i}_sngl' for eb_i in [0, 1]]].idxmin(axis=1)
    for eb_i, eb in eb_i_match.items():
        if matching_tbl.loc[eb_i, eb] == 2:
            continue

        eb_chosen = eb.split('_')[3]
        matching_tbl.loc[eb_i, eb_final_cols] = \
            matching_tbl.loc[eb_i, [f'{col}_{eb_chosen}' for col in eb_chosen_cols]].values
        # aa

    matching_tbl.to_csv(res_dir / f'matching_tbl_thr_{matching_eb_thr}_sameperiod_smallestplntnum.csv', index=False)

# # %% Add TEC flux triage and labels
#
# res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tess_tce_eb_match/11-03-2021_1731')
# matching_tbl = pd.read_csv(res_dir / 'matching_tbl.csv')
# tce_tbl = pd.read_csv(
#     '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/9-14-2021/tess_tces_s1-s40_09-14-2021_1754_stellarparams_updated_tfopwg_disp_tecfluxtriage.csv')
#
# matching_tbl = matching_tbl.merge(
#     tce_tbl[['target_id', 'tce_plnt_num', 'sector_run', 'label', 'tec_fulltriage_pass', 'tec_fulltriage_comment']],
#     on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')
#
# matching_tbl.to_csv(res_dir / 'matching_tbl_tecfluxtriage.csv', index=False)
#
# # %%
#
# tce_tbl_fp = Path(
#     '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/9-14-2021/tess_tces_s1-s40_09-14-2021_1754_stellarparams_updated_tfopwg_disp_tecfluxtriage.csv')
# # tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration', 'match_dist',
# #                 'TFOPWG Disposition', 'TESS Disposition']
# tce_tbl = pd.read_csv(tce_tbl_fp)  # [tce_tbl_cols]
# # tce_tbl.to_csv(res_dir / tce_tbl_fp.name, index=False)
#
# matching_tbl = pd.read_csv(
#     '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tess_tce_eb_match/11-03-2021_1731/matching_tbl_thr_0.2_sameperiod_smallestplntnum.csv')
# eb_final_cols = ['eb_signal_id', 'eb_bjd0', 'eb_period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
#                  'eb_match_dist']
# matching_tbl = matching_tbl[['target_id', 'tce_plnt_num', 'sector_run'] + eb_final_cols]
#
# tce_tbl_eb = tce_tbl.merge(matching_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left',
#                            validate='one_to_one')
#
# # # assignment rule: 1) TCE did not pass flux triage; AND 2) the matching distance is larger than 0.3
# # tce_tbl_.loc[(tce_tbl_tec['tec_fluxtriage_pass'] == 0) &
# #                 ((tce_tbl_tec['match_dist'] > 0.3) | (tce_tbl_tec['match_dist'].isna())), 'label'] = 'NTP'
#
# tce_tbl_eb.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_eb.csv', index=False)

""" Using TESS EB catalog provided by Jon to match to SPOC DV TCEs. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime
import logging

# local
from data_wrangling.tess.match_TESS_EBs_TCEs_utils import match_tces_to_ebs


if __name__ == "__main__":

    res_dir = Path(f'/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/'
                   f'tess_tce_eb_match/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    res_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name=f'match_tces_to_ebs_main')
    logger_handler = logging.FileHandler(filename=res_dir / f'match_tces_to_ebs_main.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting matching between TCEs and EBs in {str(res_dir)}')

    sampling_interval = 0.00001  # approximately 1 min to build ephemerides templates
    logger.info(f'Using sampling interval: {sampling_interval}')

    # load TCE table
    tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                      'ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv')
    tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration',
                    'match_dist', 'TFOPWG Disposition', 'TESS Disposition', 'TOI']
    tce_tbl = pd.read_csv(tce_tbl_fp)[tce_tbl_cols]
    logger.info(f'Using TCE table ({len(tce_tbl)} TCEs): {str(tce_tbl_fp)}')
    # tce_tbl.to_csv(res_dir / tce_tbl_fp.name, index=False)

    # load EB table
    eb_tbl_cols = ['tess_id', 'signal_id', 'bjd0', 'period']
    eb_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                         'ephemeris_tables/tess/eb_catalogs/eb_catalog/tess_eb_catalog_short_12-16-2021.txt')
    eb_tbl = pd.read_csv(eb_tbl_fp)[eb_tbl_cols]
    logger.info(f'Using EB catalog ({len(eb_tbl)} EBs): {str(eb_tbl_fp)}')
    for col in ['period', 'bjd0']:   # filter EBs without ephemerides
        eb_tbl = eb_tbl.loc[eb_tbl[col] != 'None']
    eb_tbl = eb_tbl.astype(dtype={'period': float, 'bjd0': float})
    # eb_tbl.to_csv(res_dir / 'eb_tbl.csv', index=False)

    matching_tbl = tce_tbl.copy(deep=True)

    # maximum number of EBs that can be matched to a TCE
    # since the catalog does not say which sector run they belong to, the maximum number of EBs per TCE is defined by
    # the maximum number of EBs per TIC
    max_num_matched_ebs = 2
    logger.info(f'Maximum number of possible matched EBs: {max_num_matched_ebs}')
    # columns computed for each EB
    match_tbl_eb_cols_pat = eb_tbl_cols[1:] + ['match_dist_eb', 'tce_eb_period_ratio', 'tce_eb_period_multiple_int',
                                               'tce_eb_per_rel_diff']
    match_tbl_eb_cols = []
    for i in range(max_num_matched_ebs):
        match_tbl_eb_cols.extend([f'{el}_{i}' for el in match_tbl_eb_cols_pat])
    matching_tbl = pd.concat([matching_tbl.reset_index(drop=True),
                              pd.DataFrame(data=np.nan * np.zeros((len(matching_tbl), len(match_tbl_eb_cols))),
                                           columns=match_tbl_eb_cols)], axis=1)

    # perform candidate matching between each TCE and EBs
    logger.info(f'Starting candidate matching between TCEs and EBs...')
    n_processes = 11
    tbl_jobs = np.array_split(matching_tbl, n_processes)
    tbl_jobs = [tbl_job.reset_index(drop=True) for tbl_job in tbl_jobs if len(tbl_job) > 0]
    n_processes = len(tbl_jobs)
    logger.info(f'Setting number of processes to {n_processes}')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tbl_job,) + (eb_tbl, sampling_interval, eb_tbl_cols, res_dir)
            for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
    async_results = [pool.apply_async(match_tces_to_ebs, job) for job in jobs]
    pool.close()

    # time.sleep(5)
    _ = [match_tbl_job.get() for match_tbl_job in async_results]
    # matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)
    matching_tbl = pd.concat([pd.read_csv(fp) for fp in res_dir.iterdir() if fp.stem.startswith('matching_tbl_')],
                             axis=0).reset_index()
    candidate_matching_tbl_fp = res_dir / 'matching_candidates_tbl.csv'
    matching_tbl.to_csv(candidate_matching_tbl_fp, index=False)
    logger.info(f'Finished candidate matching between TCEs and EBs: {str(candidate_matching_tbl_fp)}')

    logger.info(f'Choosing between matching candidates...')

    # consider only candidate matches below threshold, when the period is similar and for the smallest
    # planet number
    matching_eb_thr = np.inf  # matching threshold
    matching_per_rel_diff_thr = 0.05
    logger.info(f'Using as ephemerides template threshold: {matching_eb_thr}')
    logger.info(f'Using as relative period difference threshold: {matching_per_rel_diff_thr}')

    matching_tbl['target_sector_run'] = matching_tbl['target_id'].astype(str)
    matching_tbl['target_sector_run'] = matching_tbl[['target_sector_run', 'sector_run']].agg('_'.join, axis=1)

    for eb_i in range(max_num_matched_ebs):  # create fields for EB that is matched to TCE
        matching_tbl[f'match_dist_eb_{eb_i}_sngl'] = 2  # default value that means no match to an EB
    # let's look at each set of TCEs in the same TIC and sector run to see which ones are matched candidates to the EBs
    # only one TCE can be matched to an EB
    target_sector_runs = matching_tbl['target_sector_run'].unique()
    logger.info(f'Started iterating over {len(target_sector_runs)} target sector runs...')
    for target_sector_run_i, target_sector_run in enumerate(target_sector_runs):

        if (target_sector_run_i + 1) % 1000 == 0:
            logger.info(f'Iterated over {target_sector_run_i + 1} out of {len(target_sector_runs)} target sector runs.')

        for eb_i in range(max_num_matched_ebs):
            # get TCEs for a given TIC and sector run that have similar period as EB and are below the matching
            # threshold
            tces_in_target_sector_run = (matching_tbl['target_sector_run'] == target_sector_run) & \
                                        (matching_tbl[f'tce_eb_per_rel_diff_{eb_i}'] <= matching_per_rel_diff_thr) & \
                                        (matching_tbl[f'match_dist_eb_{eb_i}'] <= matching_eb_thr)
            tces_found = matching_tbl.loc[tces_in_target_sector_run]

            if len(tces_found) == 0:
                continue
            elif len(tces_found) == 1:  # only a single TCE in this sector run for the EB
                matching_tbl.loc[tces_in_target_sector_run, f'match_dist_eb_{eb_i}_sngl'] = \
                    matching_tbl.loc[tces_in_target_sector_run, f'match_dist_eb_{eb_i}']
            else:  # only match to EB the TCE with smallest planet number
                idx_tce_chosen = tces_found['tce_plnt_num'].idxmin()
                matching_tbl.loc[idx_tce_chosen, f'match_dist_eb_{eb_i}_sngl'] = \
                    matching_tbl.loc[idx_tce_chosen, f'match_dist_eb_{eb_i}']

    # select the closest eb
    logger.info(f'Selecting the closest EB for each TCE...')
    eb_final_cols = ['eb_signal_id', 'eb_bjd0', 'eb_period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
                     'eb_match_dist', 'tce_eb_per_rel_diff']
    eb_chosen_cols = ['signal_id', 'bjd0', 'period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
                      'match_dist_eb', 'tce_eb_per_rel_diff']
    matching_tbl = pd.concat([matching_tbl,
                              pd.DataFrame(data=np.nan * np.ones((len(matching_tbl), len(eb_final_cols))),
                                           columns=eb_final_cols)], axis=1)
    eb_i_match = matching_tbl[[f'match_dist_eb_{eb_i}_sngl' for eb_i in [0, 1]]].idxmin(axis=1)
    for eb_i, eb in eb_i_match.items():
        if matching_tbl.loc[eb_i, eb] == 2:  # TCE was not matched against any EB
            continue

        eb_chosen = eb.split('_')[3]
        matching_tbl.loc[eb_i, eb_final_cols] = \
            matching_tbl.loc[eb_i, [f'{col}_{eb_chosen}' for col in eb_chosen_cols]].values
        # aa

    matching_tbl_fp = res_dir / f'matching_tbl_ephemthr_{matching_eb_thr}_similarperiodthr_{matching_per_rel_diff_thr}_smallestplntnum.csv'
    matching_tbl.to_csv(matching_tbl_fp, index=False)
    logger.info(f'Finished matching between TCEs and EBs: {str(matching_tbl_fp)}')

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

""" Utility functions used in match_TESS_EBs_TCEs.py """

# 3rd party
from scipy.spatial import distance
import numpy as np
import os
from datetime import datetime
import logging

# local
from src_preprocessing.ephemeris_matching.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


def match_tces_to_ebs(matching_tbl, eb_tbl, sampling_interval, eb_tbl_cols, res_dir):
    """ Compute matching scores between TCEs and EBs using ephemerides templates and cosine distance. For each TCE, gets
    EBs that belong to the same TIC and computes matching distance between the TCE and each possible EB.

    :param matching_tbl: pandas DataFrame, TCE table
    :param eb_tbl: pandas DataFrame, EB table
    :param sampling_interval: float, sampling interval to build ephemerides pulse train time series
    :param eb_tbl_cols: list, columns in EB table to add to the matching table
    :param res_dir: Path, results directory
    :return:
    """

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

        logger.info(f'[{proc_id}] '
                    f'Processing TCE {tce["target_id"]}-{tce["tce_plnt_num"]} run {tce["sector_run"]}.')

        if (tce_i + 1) % 1000 == 0:
            print(f'[{proc_id}, {datetime.now().strftime("%m-%d-%Y_%H%M")}] Processed {tce_i + 1} out of '
                  f'{len(matching_tbl)} TCEs.')

        ebs_in_tic = eb_tbl.loc[eb_tbl['tess_id'] == tce['target_id']].reset_index()

        if len(ebs_in_tic) == 0:
            continue

        # iterate over EBs in the same TIC
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
            # relative period difference between TCE and EB
            matching_tbl.loc[tce_i, [f'tce_eb_per_rel_diff_{eb_i}']] = np.abs(1 - tce['tce_period'] / eb['period'])

            # compute closest multiple for period ratios
            if tce['tce_period'] >= eb['period']:
                # tce_eb_per_rat = tce['tce_period'] / eb['period']
                # tce_eb_per_k = np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat)
                tce_eb_per_k = np.round(tce['tce_period'] / eb['period'])
            else:  # use minus sign when TCE period is larger than EB period
                # tce_eb_per_rat = eb['period'] / tce['tce_period']
                # tce_eb_per_k = -1 * (np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat))
                tce_eb_per_k = -1 * np.round(eb['period'] / tce['tce_period'])

            matching_tbl.loc[tce_i, f'tce_eb_period_multiple_int_{eb_i}'] = tce_eb_per_k
            # aaa

    logger.info(
        f'[{proc_id}] Finished processing {len(matching_tbl)} TCEs.')

    matching_tbl.to_csv(res_dir / f'matching_tbl_{proc_id}.csv', index=False)

    # return matching_tbl

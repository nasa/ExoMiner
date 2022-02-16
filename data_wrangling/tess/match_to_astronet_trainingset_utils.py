""" Utility functions used in match_to_astronet_trainingset.py """

# 3rd party
from scipy.spatial import distance
import numpy as np
import os
from datetime import datetime
import logging

# local
from data_wrangling.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time


def match_tces_to_astronet_traininset(matching_tbl, astronet_tbl, sampling_interval, astronet_tbl_cols, res_dir):
    """ Compute matching scores between TCEs and astronet training set objects using ephemerides templates and cosine
    distance. For each TCE, gets objects that belong to the same TIC and whose sector is greater or equal to the TCE
     sector and computes matching distance between the TCE and each possible object in astronet training set.

    :param matching_tbl: pandas DataFrame, TCE table
    :param astronet_tbl: pandas DataFrame, astronet training set table
    :param sampling_interval: float, sampling interval to build ephemerides pulse train time series
    :param astronet_tbl_cols: list, columns in astronet training set table to add to the matching table
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

        # grab objects for the same target star
        objs_for_search = astronet_tbl.loc[astronet_tbl['tic'] == tce['target_id']].reset_index()

        # grab objects whose sector is greater or equal to TCE sector run
        if '-' in tce['sector_run']:
            tce_sector = tce['sector_run'].split('-')[0]  # get the first sector in multi-sector runs
        else:
            tce_sector = tce['sector_run']
        tce_sector = int(tce_sector)
        objs_for_search = objs_for_search.loc[objs_for_search['sector'] <= tce_sector]

        if len(objs_for_search) == 0:  # no objects found to be compared against the TCE
            continue

        # iterate over object
        objs_for_search['match_dist_astronet'] = 2
        for obj_i, obj in objs_for_search.iterrows():

            phase = max(tce['tce_period'], obj['astronet_period'])

            # find phase difference of object to the TCE
            closest_tce_epoch = find_nearest_epoch_to_this_time(
                obj['astronet_epoch'],
                obj['astronet_period'],
                tce['tce_time0bk']
            )

            tce_bin_ts = create_binary_time_series(epoch=tce['tce_time0bk'],
                                                   duration=tce['tce_duration'] / 24,
                                                   period=tce['tce_period'],
                                                   tStart=tce['tce_time0bk'] - phase / 2,
                                                   tEnd=tce['tce_time0bk'] + phase / 2,
                                                   samplingInterval=sampling_interval)

            obj_bin_ts = create_binary_time_series(epoch=closest_tce_epoch,
                                                   duration=obj['astronet_duration'] / 24,
                                                   period=obj['astronet_period'],
                                                   tStart=tce['tce_time0bk'] - phase / 2,
                                                   tEnd=tce['tce_time0bk'] + phase / 2,
                                                   samplingInterval=sampling_interval)

            # compute distance between TOI and TCE templates as cosine distance
            if tce_bin_ts.sum() == 0 or obj_bin_ts.sum() == 0:
                objs_for_search['matching_dist'] = 1
            else:
                objs_for_search['matching_dist'] = distance.cosine(tce_bin_ts, obj_bin_ts)

            # matching_tbl.loc[tce_i, [f'{el}_{obj_i}' for el in astronet_tbl_cols[1:]]] = obj[
            #     astronet_tbl_cols[1:]].values
            # matching_tbl.loc[tce_i, [f'match_dist_eb_{obj_i}', f'tce_eb_period_ratio_{obj_i}']] = \
            #     [match_distance, tce['tce_period'] / obj['period']]

        # choose object that closest matches to TCE
        obj_matched = objs_for_search.loc[
            objs_for_search['match_dist_astronet'] == objs_for_search['match_dist_astronet'].min()]
        matching_tbl.loc[tce_i, astronet_tbl_cols] = obj_matched
        matching_tbl.loc[tce_i, 'match_dist_astronet'] = obj_matched['match_dist_astronet']

        # # relative period difference between TCE and EB
        # matching_tbl.loc[tce_i, [f'tce_eb_per_rel_diff_{obj_i}']] = np.abs(1 - tce['tce_period'] / obj['period'])
        #
        # # compute closest multiple for period ratios
        # if tce['tce_period'] >= obj['period']:
        #     # tce_eb_per_rat = tce['tce_period'] / eb['period']
        #     # tce_eb_per_k = np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat)
        #     tce_eb_per_k = np.round(tce['tce_period'] / obj['period'])
        # else:  # use minus sign when TCE period is larger than EB period
        #     # tce_eb_per_rat = eb['period'] / tce['tce_period']
        #     # tce_eb_per_k = -1 * (np.abs(np.round(tce_eb_per_rat) - tce_eb_per_rat))
        #     tce_eb_per_k = -1 * np.round(obj['period'] / tce['tce_period'])
        #
        # matching_tbl.loc[tce_i, f'tce_eb_period_multiple_int_{obj_i}'] = tce_eb_per_k

    logger.info(f'[{proc_id}] Finished processing {len(matching_tbl)} TCEs.')

    matching_tbl.to_csv(res_dir / f'matching_tbl_{proc_id}.csv', index=False)

    # return matching_tbl

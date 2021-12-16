""" Match TCEs with their secondary event that was also detected as a TCE. """

import logging
# 3rd party
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# %% Auxiliary functions


def _compute_phase_diff(epoch1, phase1, epoch2, period2):
    """ Computes the  absolute relative difference between weak secondary phase for a given TCE (primary) and the
    estimated phase between this TCE and another one (test TCE).

    :param epoch1: float, epoch of primary TCE
    :param phase1: float, phase of weak secondary for the primary TCE
    :param epoch2: float, epoch of TCE being matched against the primary TCE
    :param period2: float, orbital period of the TCE being matched against the primary TCE
    :return:
        k_mult_dec: float, decimal part of the multiple factor between the two TCEs
    """

    # if epoch2 < epoch1:
    #     if phase1 > 0:
    #         tEpoch = epoch2 + period2 * np.ceil((epoch1 - epoch2) / period2)
    #     else:
    #         tEpoch = epoch2 + period2 * np.floor((epoch1 - epoch2) / period2)
    # else:
    #     if phase1 > 0:
    #         tEpoch = epoch2 - period2 * np.floor((epoch2 - epoch1) / period2)
    #     else:
    #         tEpoch = epoch2 - period2 * np.ceil((epoch2 - epoch1) / period2)

    k_mult = (epoch2 - epoch1 - phase1) / period2

    k_mult_dec = np.abs(k_mult - np.round(k_mult))

    # phaseEst = tEpoch - epoch1

    # phaseDiffRel = np.abs(phaseEst - phase1) / np.abs(phase1)

    # return phaseDiffRel, phaseEst
    return k_mult_dec


def _compute_rel_period_diff(period1, period2):
    """ Computes the  absolute relative difference between the orbital periods of a given TCE (primary) and test TCE
    which is being matched against the former as its secondary.

    :param period1: float, orbital period of primary TCE
    :param period2: float, orbital period of test TCE
    :return:
        float, absolute relative orbital period difference
    """

    return np.abs(period1 - period2) / period1


def _phase_match(row, epochReference, wksphase):
    return _compute_phase_diff(epochReference, wksphase, row['tce_time0bk'], row['tce_period'])


def _compute_period_test(period1, period2):
    """ Computes the ratio between periods of primary and secondary TCEs and returns the absolute value of the decimal
    part.

    :param period1: float, primary period
    :param period2: float, primary period
    :return:
        r_period: float, decimal part of the multiple factor between the two TCEs

    """

    period_ratio = period1 / period2

    k_period = np.round(period_ratio)

    if k_period in [1, 2]:
        r_period = np.abs(period_ratio - np.round(period_ratio))
    else:
        r_period = np.inf

    return r_period


def _period_match(row, period1):
    return _compute_period_test(period1, row['tce_period'])


def _match_tces(tce, tceTbl, cond):
    """ Match TCE to other TCEs with higher tce planet number in the same target star.

    :param tce: pandas Series, TCE parameters
    :param tceTbl: pandas DataFrame, TCE table
    :param cond: dict, contains matching thresholds
    :return:
        results: pandas Series, matching result for the given TCE
    """

    result = pd.Series(data={
        'matched_tces': '',
        'num_matched_tces': 0,
        'match_phase_value_min': np.inf,
        'match_period_value_min': np.inf
    })

    # get TCEs in the same target star with higher tce_plnt_num
    tcesInTarget = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) &
                              (tceTbl['tce_plnt_num'] > tce['tce_plnt_num'])].reset_index()

    if len(tcesInTarget) == 0:  # no TCEs found
        return result

    tcesInTarget = pd.concat([tcesInTarget,
                              pd.DataFrame(data={
                                  'match_period_value': np.inf * np.ones(len(tcesInTarget)),
                                  'match_phase_value': np.inf * np.ones(len(tcesInTarget))
                              })], axis=1)

    # period match
    tcesInTarget['match_period_value'] = tcesInTarget[['tce_period']].apply(_period_match,
                                                                            args=(tce['tce_period'],
                                                                                  ),
                                                                            axis=1)

    # phase match
    tcesInTarget['match_phase_value'] = tcesInTarget[['tce_period', 'tce_time0bk']].apply(_phase_match,
                                                                                          args=(tce['tce_time0bk'],
                                                                                                tce['tce_maxmesd']),
                                                                                          axis=1)

    tcesInTargetMatched = tcesInTarget.loc[(tcesInTarget['match_phase_value'] < cond['phase_match_thr']) &
                                           (tcesInTarget['match_period_value'] < cond[
                                               'period_match_thr'])].reset_index()

    # sort by match value
    tcesInTargetMatched.sort_values('match_phase_value', ascending=False, inplace=True)

    result['num_matched_tces'] = len(tcesInTargetMatched)
    if result['num_matched_tces'] > 0:
        result['matched_tces'] = str(tcesInTargetMatched['tce_plnt_num'].values)[1:-1]

    # get the match values for the TCE with the lowest period match value
    idxMin = tcesInTarget['match_period_value'].idxmin()

    result['match_phase_value_min'] = tcesInTarget.loc[idxMin, 'match_phase_value']
    result['match_period_value_min'] = tcesInTarget.loc[idxMin, 'match_period_value']

    return result


# %%

if __name__ == '__main__':

    matchConditions = {
        'phase_match_thr': 5e-2,
        'period_match_thr': 1e-3
    }

    timestamp = datetime.now().strftime('%d-%m-%y_%H:%M')

    workRootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/')
    workDir = workRootDir / timestamp
    workDir.mkdir(exist_ok=True)

    logging.basicConfig(filename=os.path.join(workDir,
                                              f'results_phasematchthr_{matchConditions["phase_match_thr"]}_periodmatchthr_'
                                              f'{matchConditions["period_match_thr"]}_{timestamp}.log'),
                        level=logging.INFO,
                        format='%(message)s',
                        filemode='w')

    tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                         'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                         'nomissingval.csv')

    print('Total number of TCEs in the dataset: {}'.format(len(tceTbl)))
    logging.info('Total number of TCEs in the dataset: {}'.format(len(tceTbl)))

    # count number of TCEs per target star
    numTcesPerTarget = tceTbl['target_id'].value_counts()
    # get targets with more than one TCE
    multiTceTargets = numTcesPerTarget.loc[numTcesPerTarget > 1]

    # remove TCEs from target stars with only one TCE
    tceTblMultiTce = tceTbl.loc[tceTbl['target_id'].isin(multiTceTargets.index)].reset_index()

    print('Number of TCEs in multi-TCE target stars: {}'.format(len(tceTblMultiTce)))
    logging.info('Number of TCEs in multi-TCE target stars: {}'.format(len(tceTblMultiTce)))

    # sort TCEs by ascending Kepler ID and tce_plnt_num
    tceTblMultiTce.sort_values(['target_id', 'tce_plnt_num'], axis=0, ascending=True, inplace=True, ignore_index=True)

    # initialize columns
    secondaryMatchedTbl = pd.concat([tceTblMultiTce,
                                     pd.DataFrame(data={
                                         'matched_tces': np.nan * np.ones(len(tceTblMultiTce)),
                                         'num_matched_tces': np.zeros(len(tceTblMultiTce)),
                                         'match_phase_value_min': np.inf * np.ones(len(tceTblMultiTce)),
                                         'match_period_value_min': np.inf * np.ones(len(tceTblMultiTce))
                                     })], axis=1, ignore_index=False)

    secondaryMatchedTbl[['matched_tces',
                         'num_matched_tces',
                         'match_phase_value_min',
                         'match_period_value_min']] = secondaryMatchedTbl.apply(_match_tces,
                                                                                args=(tceTbl, matchConditions),
                                                                                axis=1)

    countMatches = secondaryMatchedTbl['num_matched_tces'].value_counts()
    logging.info('Number of matches:')
    for count_i, count in enumerate(countMatches):
        matchStr = '{} matches per TCE: {}'.format(count_i, count)
        print(matchStr)
        logging.info(matchStr)

    fpTbl = os.path.join(workDir,
                         f'matched_tces_phasematchthr_{matchConditions["phase_match_thr"]}_periodmatchthr_'
                         f'{matchConditions["period_match_thr"]}_{timestamp}.csv')

    secondaryMatchedTbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_maxmesd', 'matched_tces',
                         'num_matched_tces', 'match_phase_value_min', 'match_period_value_min', 'label',
                         'koi_disposition', 'fpwg_disp_status']].to_csv(fpTbl, index=False)

    logging.info('Preparing matching table...')

    # get only TCEs with at least one match
    secondaryMatchedTbl = secondaryMatchedTbl.loc[secondaryMatchedTbl['num_matched_tces'] > 0]

    finalMatchedTbl = pd.DataFrame(columns=['target_id', 'primary_tce', 'secondary_tce', 'tce_maxmesd_primary',
                                            'tce_maxmesd_secondary', 'tce_maxmesd_secondary_new',
                                            'koi_disposition_primary', 'koi_disposition_secondary',
                                            'fpwg_disp_status_primary', 'fpwg_disp_status_secondary',
                                            'label_primary', 'label_secondary'])

    matchedTcesInTarget = {'target_id': -1, 'tces_matched': []}
    for tce_i, tce in secondaryMatchedTbl.iterrows():
        if tce['target_id'] != matchedTcesInTarget['target_id']:  # checking a new target star
            matchedTcesInTarget = {'target_id': tce['target_id'],
                                   'tces_matched': []
                                   }

        primaryTce = tce['tce_plnt_num']
        secondaryTce = int(tce['matched_tces'].split(' ')[0])

        # add matched pair if both were not already matched
        if primaryTce not in matchedTcesInTarget['tces_matched'] and \
                secondaryTce not in matchedTcesInTarget['tces_matched']:
            matchedTcesInTarget['tces_matched'].extend([primaryTce, secondaryTce])

            secondaryTceLoc = (tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == secondaryTce)

            finalMatchedTbl = pd.concat(
                [finalMatchedTbl,
                 pd.DataFrame(data={'target_id': [tce['target_id']],
                                    'primary_tce': [tce['tce_plnt_num']],
                                    'secondary_tce': [secondaryTce],
                                    'tce_maxmesd_primary': [tce['tce_maxmesd']],
                                    'tce_maxmesd_secondary': tceTbl.loc[secondaryTceLoc,
                                                                        ['tce_maxmesd']].values[
                                        0],
                                    'tce_maxmesd_secondary_new': [-tce['tce_maxmesd']],
                                    'koi_disposition_primary': tce['koi_disposition'],
                                    'koi_disposition_secondary': tceTbl.loc[secondaryTceLoc,
                                                                            [
                                                                                'koi_disposition']].values[
                                        0],
                                    'fpwg_disp_status_primary': tce['fpwg_disp_status'],
                                    'fpwg_disp_status_secondary': tceTbl.loc[secondaryTceLoc,
                                                                             [
                                                                                 'fpwg_disp_status']].values[
                                        0],
                                    'label_primary': tce['label'],
                                    'label_secondary': tceTbl.loc[secondaryTceLoc,
                                                                  ['label']].values[0],
                                    })])

    finalMatchedTbl.to_csv('{}_final.csv'.format(fpTbl[:-4]), index=False)

    logging.shutdown()

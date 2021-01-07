""" Match TCEs with their secondary event that was also detected as a TCE. """

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

#%% Auxiliary functions


def _compute_phase_diff(epoch1, phase1, epoch2, period2):
    """ Computes the  absolute relative difference between weak secondary phase for a given TCE (primary) and the
    estimated phase between this TCE and another one (test TCE).

    :param epoch1: float, epoch of primary TCE
    :param phase1: float, phase of weak secondary for the primary TCE
    :param epoch2: float, epoch of TCE being matched against the primary TCE
    :param period2: float, orbital period of the TCE being matched against the primary TCE
    :return:
        # phaseDiffRel: float, relative absolute difference between the phase of the weak secondary of the primary TCE
        # and the estimated phase for the test TCE
        # phaseEst: float, estimated phase
        k_mult_dec: decimal part of the multiple factor between the two TCEs
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


def _match_tces(tce, tceTbl, cond):

    result = pd.Series(data={
        'matched_tces': '',
        'num_matched_tces': 0,
        'match_phase_value_min': np.inf,
        'match_period_value_min': np.inf
    })

    # get TCEs in the same target star with higher tce_plnt_num
    tcesInTarget = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) &
                              (tceTbl['tce_plnt_num'] > tce['tce_plnt_num'])].reset_index()

    if len(tcesInTarget) == 0:
        return result

    tcesInTarget = pd.concat([tcesInTarget,
                              pd.DataFrame(data={
                                  'match_period_value': np.inf * np.ones(len(tcesInTarget)),
                                  'match_phase_value': np.inf * np.ones(len(tcesInTarget))
                              })], axis=1)

    # period difference thresholding
    # tcesInTarget['match_value'] = np.abs(tcesInTarget['tce_period'] - tce['tce_period'])
    tcesInTarget['match_period_value'] = np.abs(tcesInTarget['tce_period'] - tce['tce_period'])

    # relative period difference thresholding
    # tcesInTarget['match_period_value'] = np.abs(tcesInTarget['tce_period'] - tce['tce_period']) / \
    #                                      tcesInTarget['tce_period'].apply(lambda x: max(x, tce['tce_period']))
    # tcesInTarget['match_period_value'] = np.abs(tcesInTarget['tce_period'] - tce['tce_period']) / tce['tce_period']

    # phase match
    def _phase_match(row, epochReference, wksphase):

        # phaseDiffRel, phaseEst = _compute_phase_diff(epochReference, wksphase, row['tce_time0bk'], row['tce_period'])
        k_mult_dec = _compute_phase_diff(epochReference, wksphase, row['tce_time0bk'], row['tce_period'])

        return k_mult_dec

    tcesInTarget['match_phase_value'] = tcesInTarget[['tce_period', 'tce_time0bk']].apply(_phase_match,
                                                                                          args=(tce['tce_time0bk'],
                                                                                                tce['tce_maxmesd']),
                                                                                          axis=1)

    # tcesInTargetMatched = tcesInTarget.loc[(tcesInTarget['match_phase_value'] < cond['max_phase_diff_rel']) &
    #                                        (tcesInTarget['match_period_value'] < cond['max_period_diff_rel'])].reset_index()
    tcesInTargetMatched = tcesInTarget.loc[(tcesInTarget['match_phase_value'] < cond['max_phase_diff_rel']) &
                                           (tcesInTarget['match_period_value'] < cond['max_period_diff'])].reset_index()

    # sort by match value
    tcesInTargetMatched.sort_values('match_phase_value', ascending=False, inplace=True)

    result['num_matched_tces'] = len(tcesInTargetMatched)
    if result['num_matched_tces'] > 0:
        result['matched_tces'] = str(tcesInTargetMatched['tce_plnt_num'].values)[1:-1]

    # get the match values for the TCE with the lowest period match value
    idxMin = tcesInTarget['match_period_value'].idxmin()
    # if tce['target_id'] == 8243804 and tce['tce_plnt_num'] == 3:
    #     tcesInTarget[['target_id', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_maxmesd', 'match_phase_value',
    #                   'match_period_value']].to_csv('/home/msaragoc/Downloads/index.csv')

    result['match_phase_value_min'] = tcesInTarget.loc[idxMin, 'match_phase_value']
    result['match_period_value_min'] = tcesInTarget.loc[idxMin, 'match_period_value']

    return result


# %%

workDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/'

matchConditions = {
    'max_period_diff': 5e-3,
    # 'max_period_diff_rel': 1e-3,
    'max_phase_diff_rel': 1e-2  # 1e-1
}

timestamp = datetime.now().strftime('%d-%m-%y_%H:%M')

logging.basicConfig(filename=os.path.join(workDir, 'results_{}_{}_{}.log'.format(list(matchConditions.keys()),
                                                                                 list(matchConditions.values()),
                                                                                 timestamp)),
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
tceTblMultiTce = pd.concat([tceTblMultiTce,
                            pd.DataFrame(data={
                                'matched_tces': np.nan * np.ones(len(tceTblMultiTce)),
                                'num_matched_tces': np.zeros(len(tceTblMultiTce)),
                                'match_phase_value_min': np.inf * np.ones(len(tceTblMultiTce)),
                                'match_period_value_min': np.inf * np.ones(len(tceTblMultiTce))
                            })], axis=1, ignore_index=False)

tceTblMultiTce[['matched_tces',
                'num_matched_tces',
                'match_phase_value_min',
                'match_period_value_min']] = tceTblMultiTce.apply(_match_tces, args=(tceTbl, matchConditions), axis=1)

countMatches = tceTblMultiTce['num_matched_tces'].value_counts()
logging.info('Number of matches:')
for count_i, count in enumerate(countMatches):
    matchStr = '{} matches per TCE: {}'.format(count_i, count)
    print(matchStr)
    logging.info(matchStr)

fpTbl = os.path.join(workDir, 'match_tces_{}_{}_{}.csv'.format(list(matchConditions.keys()),
                                                               list(matchConditions.values()),
                                                               timestamp))

tceTblMultiTce[['target_id', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_maxmesd', 'matched_tces',
                'num_matched_tces', 'match_phase_value_min', 'match_period_value_min', 'label', 'koi_disposition',
                'fpwg_disp_status']].to_csv(fpTbl, index=False)

logging.shutdown()

#%% Compute match values for TCEs in a given target star

targetId = 8243804
tcesTarget = tceTbl.loc[tceTbl['target_id'] == targetId].reset_index()
numTcesInTarget = len(tcesTarget)

# matchMat = {'relative_period_difference': np.inf * np.ones((numTcesInTarget, numTcesInTarget), dtype='float'),
#             'relative_phase_difference': np.inf * np.ones((numTcesInTarget, numTcesInTarget), dtype='float')
#             }
matchMat = {'period_difference': np.inf * np.ones((numTcesInTarget, numTcesInTarget), dtype='float'),
            'relative_phase_difference': np.inf * np.ones((numTcesInTarget, numTcesInTarget), dtype='float')
            }

fp = '/home/msaragoc/Downloads/match_{}.csv'.format(targetId)

for tce_i, tce in tcesTarget.iterrows():

    testTces = tcesTarget.loc[tcesTarget['tce_plnt_num'] > tce['tce_plnt_num']]

    for test_tce_i, test_tce in testTces.iterrows():

        print('#' * 10)

        print('Testing TCE {} against TCE {}'.format(tce['tce_plnt_num'], test_tce['tce_plnt_num']))

        phaseDiffRel, phaseEst = _compute_phase_diff(tce['tce_time0bk'], tce['tce_maxmesd'], test_tce['tce_time0bk'],
                                                     test_tce['tce_period'])

        print('Estimated vs wks phase (day): {} | {}'.format(phaseEst, tce['tce_maxmesd']))
        print('Relative phase difference: {}'.format(phaseDiffRel))

        # perDiffRel = _compute_rel_period_diff(tce['tce_period'], test_tce['tce_period'])
        perDiff = np.abs(tce['tce_period'] - test_tce['tce_period'])

        print('Period difference (day): {}'.format(perDiff))
        # print('Relative period difference: {}'.format(perDiffRel))

        # matchMat['relative_period_difference'][tce_i, test_tce_i] = perDiffRel
        matchMat['period_difference'][tce_i, test_tce_i] = perDiff
        matchMat['relative_phase_difference'][tce_i, test_tce_i] = phaseDiffRel

# perDiffRelDf = pd.DataFrame(data=matchMat['relative_period_difference'], index=tcesTarget['tce_plnt_num'],
#                             columns=tcesTarget['tce_plnt_num'])
perDiffDf = pd.DataFrame(data=matchMat['period_difference'], index=tcesTarget['tce_plnt_num'],
                         columns=tcesTarget['tce_plnt_num'])
phaseDiffRelDf = pd.DataFrame(data=matchMat['relative_phase_difference'], index=tcesTarget['tce_plnt_num'],
                              columns=tcesTarget['tce_plnt_num'])

# matchTargetDf = pd.concat([perDiffRelDf, phaseDiffRelDf], axis=0,
#                           keys=['Relative period difference', 'Relative phase difference'],
#                           names=['Match value parameter', 'tce_plnt_num'])
matchTargetDf = pd.concat([perDiff, phaseDiffRelDf], axis=0,
                          keys=['Period difference', 'Relative phase difference'],
                          names=['Match value parameter', 'tce_plnt_num'])
matchTargetDf.to_csv(fp)


#%% Change phase of matched secondaries

tceTblFp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'\
           'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval.csv'
tceTbl = pd.read_csv(tceTblFp)

secondaryMatchedTblFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/match_tces_' \
                        '[\'max_period_diff\', \'max_phase_diff_rel\']_[0.005, 0.01]_23-10-20_12:54.csv'
secondaryMatchedTbl = pd.read_csv(secondaryMatchedTblFp)

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

        finalMatchedTbl = pd.concat([finalMatchedTbl,
                                     pd.DataFrame(data={'target_id': [tce['target_id']],
                                                        'primary_tce': [tce['tce_plnt_num']],
                                                        'secondary_tce': [secondaryTce],
                                                        'tce_maxmesd_primary': [tce['tce_maxmesd']],
                                                        'tce_maxmesd_secondary': tceTbl.loc[(tceTbl['target_id'] ==
                                                                                             tce['target_id']) &
                                                                                            (tceTbl['tce_plnt_num'] ==
                                                                                             secondaryTce),
                                                                                            ['tce_maxmesd']].values[0],
                                                        'tce_maxmesd_secondary_new': [-tce['tce_maxmesd']],
                                                        'koi_disposition_primary': tce['koi_disposition'],
                                                        'koi_disposition_secondary': tceTbl.loc[(tceTbl['target_id'] ==
                                                                                             tce['target_id']) &
                                                                                            (tceTbl['tce_plnt_num'] ==
                                                                                             secondaryTce),
                                                                                            ['koi_disposition']].values[0],
                                                        'fpwg_disp_status_primary': tce['fpwg_disp_status'],
                                                        'fpwg_disp_status_secondary': tceTbl.loc[(tceTbl['target_id'] ==
                                                                                             tce['target_id']) &
                                                                                            (tceTbl['tce_plnt_num'] ==
                                                                                             secondaryTce),
                                                                                            ['fpwg_disp_status']].values[0],
                                                        'label_primary': tce['label'],
                                                        'label_secondary': tceTbl.loc[(tceTbl['target_id'] ==
                                                                                             tce['target_id']) &
                                                                                            (tceTbl['tce_plnt_num'] ==
                                                                                             secondaryTce),
                                                                                            ['label']].values[0],
                                                        })])

        # change phase of the matched secondary to the symmetric of the phase of the matched primary
        tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) &
                   (tceTbl['tce_plnt_num'] == secondaryTce), ['tce_maxmesd']] = - tce['tce_maxmesd']

finalMatchedTbl.to_csv('{}_final.csv'.format(secondaryMatchedTblFp[:-4]), index=False)

tceTbl.to_csv('{}_symsecphase.csv'.format(tceTblFp[:-4]), index=False)

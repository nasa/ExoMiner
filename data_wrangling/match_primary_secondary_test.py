# %% Compute match values for TCEs in a given target star

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

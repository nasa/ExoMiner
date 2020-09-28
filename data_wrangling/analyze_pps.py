"""
Script used to analyze the performance of models trained with and without PPs.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Get PPs for each model ranking

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/reliability_pps'
experimentRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'

experimentNoPPs = 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar_datasetwithpps'
experimentWithPPs = 'keplerdr25_g2001-l201_spline_gapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

# get only PPs
# tceTbl = tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'POSSIBLE PLANET') & (tceTbl['koi_disposition'] != 'CONFIRMED')]
# tceTbl = tceTbl.loc[tceTbl['koi_disposition'] == 'CONFIRMED']
# tceTbl = tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'CERTIFIED FP') & (tceTbl['koi_disposition'] != 'CONFIRMED')]
# tceTbl = tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'CERTIFIED FA') & (tceTbl['koi_disposition'] != 'CONFIRMED')]
# tceTbl = tceTbl.loc[tceTbl['koi_disposition'] == 'CANDIDATE']
tceTbl = tceTbl.loc[tceTbl['koi_disposition'].isna()]
print('Total number of PPs in the dataset: {}'.format(len(tceTbl)))

tceTbl['UID'] = tceTbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']),
                                                            axis=1)

datasets = ['train', 'test', 'val']
datasetsRankTbl = {dataset: None for dataset in datasets}
for dataset in datasets:

    rankTblNoPPs = pd.read_csv(os.path.join(experimentRootDir, experimentNoPPs,
                                            'ensemble_ranked_predictions_{}set.csv'.format(dataset)))
    rankTblNoPPs['UID'] = rankTblNoPPs[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                     x['tce_plnt_num']),
                                                                            axis=1)
    rankTblWithPPs = pd.read_csv(os.path.join(experimentRootDir, experimentWithPPs,
                                              'ensemble_ranked_predictions_{}set.csv'.format(dataset)))
    rankTblWithPPs['UID'] = rankTblWithPPs[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                         x['tce_plnt_num']),
                                                                                axis=1)

    # get only PPs in the ranking
    rankTblNoPPs = rankTblNoPPs[rankTblNoPPs['UID'].isin(tceTbl['UID'])]
    rankTblWithPPs = rankTblWithPPs[rankTblWithPPs['UID'].isin(tceTbl['UID'])]

    # sort ranking by the order of the other one
    rankTblNoPPs = rankTblNoPPs.set_index('UID')
    rankTblNoPPs = rankTblNoPPs.reindex(index=rankTblWithPPs['UID'])
    rankTblNoPPs.reset_index(level=0, inplace=True)

    # add score and predicted class columns
    rankTblWithPPs.reset_index(inplace=True)
    rankTblWithPPs[['score No PPs', 'predicted class No PPs']] = rankTblNoPPs[['score', 'predicted class']]

    # drop UID column
    rankTblWithPPs.drop(columns=['index', 'UID'], inplace=True)
    rankTblWithPPs.to_csv(os.path.join(saveDir, 'ranking_{}set.csv'.format(dataset)), index=False)
    datasetsRankTbl[dataset] = rankTblWithPPs

#%% Plot scores

for dataset in datasets:

    f, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k:')
    # ax.plot(datasetsRankTbl[dataset]['score'], datasetsRankTbl[dataset]['score No PPs'])
    ax.scatter(datasetsRankTbl[dataset]['score'], datasetsRankTbl[dataset]['score No PPs'], c='r')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Model Trained with PPs')
    ax.set_ylabel('Model Trained without PPs')
    ax.set_xticks(np.linspace(0, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, 11, endpoint=True))
    ax.set_title('{} set non-KOIs\nTotal non-KOIs = {}\n'
                 'Classified as Positives (wo/ vs w/)= {} vs {}'.format(dataset, len(datasetsRankTbl[dataset]),
                                                                        datasetsRankTbl[dataset]['predicted class No PPs'].sum(),
                                                                        datasetsRankTbl[dataset]['predicted class'].sum()))
    ax.grid(True)
    # f.savefig(os.path.join(saveDir, 'scorespps_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    # f.savefig(os.path.join(saveDir, 'scoresconfirmedKOIs_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    # f.savefig(os.path.join(saveDir, 'scoresCFPs_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    # f.savefig(os.path.join(saveDir, 'scoresCFAs_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    # f.savefig(os.path.join(saveDir, 'scorescandidateKOIs_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    f.savefig(os.path.join(saveDir, 'scoresnonKOIs_ppmodelvsnonppmodel_{}set.svg'.format(dataset)))
    plt.close()

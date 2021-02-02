""" Script used to analyze the performance of models on different sets of dispositions. """

import pandas as pd
import os
from pathlib import Path

#%% Get rankings for each disposition and dataset

experimentRootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/')

experiment = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_testvalswitched'

# saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/misclassified_analysis/'
saveDir = experimentRootDir / experiment / 'misclassified_analysis'
saveDir.mkdir(exist_ok=True)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                     'rmcandandfpkois_norogues.csv')

dispToCheck = {
    # 'Possible Planet KOI': tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'POSSIBLE PLANET') &
    #                                   (tceTbl['koi_disposition'] != 'CONFIRMED')],
    'Confirmed KOI': tceTbl.loc[tceTbl['koi_disposition'] == 'CONFIRMED'],
    'Certified FP': tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'CERTIFIED FP') &
                               (tceTbl['koi_disposition'] != 'CONFIRMED')],
    'Certified FA': tceTbl.loc[(tceTbl['fpwg_disp_status'] == 'CERTIFIED FA') &
                               (tceTbl['koi_disposition'] != 'CONFIRMED')],
    'Candidate KOI': tceTbl.loc[tceTbl['koi_disposition'] == 'CANDIDATE'],
    'Non-KOI': tceTbl.loc[tceTbl['koi_disposition'].isna()],
}

datasets = ['train', 'test', 'val']

for disp, tbl in dispToCheck.items():

    print('Total number of {} in the dataset: {}'.format(disp, len(tbl)))

    tbl['UID'] = tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']),
                                                          axis=1)

    datasetsRankTbl = {dataset: None for dataset in datasets}
    for dataset in datasets:

        rankTbl = pd.read_csv(os.path.join(experimentRootDir, experiment,
                                           'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

        rankTbl['UID'] = rankTbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                               x['tce_plnt_num']),
                                                                      axis=1)

        # get only TCEs with given disposition in the ranking
        rankTbl = rankTbl[rankTbl['UID'].isin(tbl['UID'])]

        # drop UID column
        rankTbl.drop(columns=['UID'], inplace=True)
        rankTbl.to_csv(saveDir / 'ranking_{}_{}set.csv'.format(disp, dataset), index=False)

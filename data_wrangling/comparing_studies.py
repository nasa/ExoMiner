import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.metrics import AUC

#%% plot PR curves for different studies

studyMetrics = {'Astronet': np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_shallue_starshuffle/results_ensemble.npy', allow_pickle=True).item(),
                'Exonet': np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_fdl_starshuffle/results_ensemble.npy', allow_pickle=True).item(),
                'Planetfindernet': np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/bohb_keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost/results_ensemble.npy', allow_pickle=True).item()
                }

studyPRMetrics = {}
for study in studyMetrics:
    studyPRMetrics[study] = {'precision_thr': studyMetrics[study]['test_prec_thr'],
                             'recall_thr': studyMetrics[study]['test_rec_thr']
                             }

studyPRMetrics['Robovetter'] = {'precision_thr': np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/precision_thr.npy')[:-1],
                                'recall_thr': np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/recall_thr.npy')[:-1]
                                }

f, ax = plt.subplots()
for study in studyPRMetrics:
    ax.plot(studyPRMetrics[study]['recall_thr'], studyPRMetrics[study]['precision_thr'], label=study)
    ax.scatter(studyPRMetrics[study]['recall_thr'], studyPRMetrics[study]['precision_thr'])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Test set')
ax.legend()
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 21, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 21, endpoint=True))
# ax.set_xlim([0.5, 1])
# ax.set_ylim([0.5, 1])

#%% Compute AUC PR for KOIs only

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobobvetterKOIs.csv')

# remove non-KOIs
tceTbl = tceTbl.loc[~tceTbl['kepoi_name'].isna()]

rankingTbl = pd.read_csv(os.path.join('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_nongapped_glfluxbohb_norobovetterkois_starshuffle_glflux-lwks',
                                      'ensemble_ranked_predictions_testset'))

rankingScores, rankingLabels = [], []
for tce_i, tce in rankingTbl.iterrows():

    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(foundTce) == 1:
        rankingScores.append(tce['score'])
        rankingLabels.append(tce['label'])

num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds))

auc_pr = AUC(num_thresholds=num_thresholds,
                           summation_method='interpolation',
                           curve='PR',
                           name='auc_pr')

_ = auc_pr.update_state(rankingLabels, rankingScores)
print(auc_pr.result().numpy())

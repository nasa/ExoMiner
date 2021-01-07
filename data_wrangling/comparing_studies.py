""" Compare different models (ExoMiner, AstroNet, ExoNet, Robovetter, Armstrong et al. classifiers, ...) """

# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.metrics import AUC
from pathlib import Path

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
# ax.set_xlim([0.5, 1])
# ax.set_ylim([0.5, 1])
ax.set_xticks(np.linspace(0, 1, 21, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 21, endpoint=True))

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

#%%

comparing_studies_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/comparing_studies_12-23-2020')
comparing_studies_dir.mkdir(exist_ok=True)

# datasets tables
datasets = ['train', 'val', 'test']
datasets_tbl = None
for dataset in datasets:
    dataset_tbl = pd.read_csv(f'/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/'
                              f'split_6-1-2020/{dataset}set.csv')
    # remove Possible Planet KOIs
    dataset_tbl = dataset_tbl.loc[~((dataset_tbl['fpwg_disp_status'] == 'POSSIBLE PLANET') &
                                    (dataset_tbl['koi_disposition'] != 'CONFIRMED'))]
    dataset_tbl['dataset'] = dataset
    if datasets_tbl is None:
        datasets_tbl = dataset_tbl
    else:
        datasets_tbl = pd.concat([datasets_tbl, dataset_tbl], axis=0)

datasets_tbl['uid'] = datasets_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                 x['tce_plnt_num']),
                                                                        axis=1)
print(f'Dataset: {len(datasets_tbl)}\n{datasets_tbl["label"].value_counts()}')

# get scores for ExoMiner
exominer_tbl_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                        'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_'
                        'wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_'
                        'prad_per/')

exominer_tbl = None
for dataset in datasets:
    tbl_aux = pd.read_csv(exominer_tbl_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
    tbl_aux['dataset'] = dataset
    if exominer_tbl is None:
        exominer_tbl = tbl_aux
    else:
        exominer_tbl = pd.concat([exominer_tbl, tbl_aux], axis=0)

exominer_tbl.reset_index(drop=True, inplace=True)
exominer_tbl = exominer_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration',
                             'tce_time0bk', 'dataset', 'original_label', 'label', 'score', 'predicted class']]
exominer_tbl.rename({'score': 'exominer_score', 'predicted class': 'exominer_pred_class'}, inplace=True, axis=1)
exominer_tbl.to_csv(comparing_studies_dir / 'ranking_exominer.csv', index=False)
print(f'ExoMiner: {len(exominer_tbl)}\n{exominer_tbl["label"].value_counts()}')

# get scores for AstroNet
astronet_tbl_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                        'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronet_'
                        'secsymphase_nopps_ckoiper')

astronet_tbl = None
for dataset in datasets:
    tbl_aux = pd.read_csv(astronet_tbl_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
    tbl_aux['dataset'] = dataset
    if astronet_tbl is None:
        astronet_tbl = tbl_aux
    else:
        astronet_tbl = pd.concat([astronet_tbl, tbl_aux], axis=0)

astronet_tbl.reset_index(drop=True, inplace=True)
astronet_tbl = astronet_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration',
                             'tce_time0bk', 'dataset', 'original_label', 'label', 'score', 'predicted class']]
astronet_tbl.rename({'score': 'astronet_score', 'predicted class': 'astronet_pred_class'}, inplace=True, axis=1)
astronet_tbl.to_csv(comparing_studies_dir / 'ranking_astronet.csv', index=False)
print(f'AstroNet: {len(astronet_tbl)}\n{astronet_tbl["label"].value_counts()}')

# get scores for ExoNet
exonet_tbl_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                      'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonet_secsymphase_'
                      'nopps_ckoiper')

exonet_tbl = None
for dataset in datasets:
    tbl_aux = pd.read_csv(exonet_tbl_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
    tbl_aux['dataset'] = dataset
    if exonet_tbl is None:
        exonet_tbl = tbl_aux
    else:
        exonet_tbl = pd.concat([exonet_tbl, tbl_aux], axis=0)

exonet_tbl.reset_index(drop=True, inplace=True)
exonet_tbl = exonet_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration',
                             'tce_time0bk', 'dataset', 'original_label', 'label', 'score', 'predicted class']]
exonet_tbl.rename({'score': 'exonet_score', 'predicted class': 'exonet_pred_class'}, inplace=True, axis=1)
exonet_tbl.to_csv(comparing_studies_dir / 'ranking_exonet.csv', index=False)
print(f'ExoNet: {len(exonet_tbl)}\n{exonet_tbl["label"].value_counts()}')

# get scores for Robovetter
robovetter_tbl_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/without_PPs/'
                          '12-22-2020/')

robovetter_tbl = None
for dataset in datasets:
    tbl_aux = pd.read_csv(robovetter_tbl_dir / f'kplr_dr25_obs_robovetter_{dataset}.csv')
    tbl_aux['dataset'] = dataset
    if robovetter_tbl is None:
        robovetter_tbl = tbl_aux
    else:
        robovetter_tbl = pd.concat([robovetter_tbl, tbl_aux], axis=0)

robovetter_tbl['uid'] = robovetter_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                     x['tce_plnt_num']),
                                                                            axis=1)
robovetter_tbl = robovetter_tbl.loc[robovetter_tbl['uid'].isin(datasets_tbl['uid'])]
robovetter_tbl.reset_index(drop=True, inplace=True)
robovetter_tbl = robovetter_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration',
                             'tce_time0bk', 'dataset', 'label', 'score', 'Robovetter_disposition']]
robovetter_tbl.rename({'score': 'robovetter_score', 'Robovetter_disposition': 'robovetter_pred_class'}, inplace=True,
                      axis=1)
robovetter_tbl.to_csv(comparing_studies_dir / 'ranking_robovetter.csv', index=False)
print(f'Robovetter: {len(robovetter_tbl)}\n{robovetter_tbl["label"].value_counts()}')

# get scores for Armstrong et al GPC and RFC

armstrong_tbl_fp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                         'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_'
                         'wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_'
                         'prad_per/comparison_ranking_50exoplntpaper/ranking_comparison_with_paper_12-18-2020.csv')
armstrong_tbl = pd.read_csv(armstrong_tbl_fp)

armstrong_tbl['uid'] = armstrong_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                   x['tce_plnt_num']),
                                                                          axis=1)
armstrong_tbl = armstrong_tbl.loc[armstrong_tbl['uid'].isin(datasets_tbl['uid'])]

armstrong_tbl = armstrong_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration',
                               'tce_time0bk', 'dataset', 'original_label', 'label', 'PP_GPC', 'PP_RFC']]
armstrong_tbl.rename({'PP_GPC': 'GPC_score', 'PP_RFC': 'RFC_score'}, inplace=True, axis=1)
armstrong_tbl.to_csv(comparing_studies_dir / 'ranking_armstrong.csv', index=False)
print(f'Armstrong: {len(armstrong_tbl)}\n{armstrong_tbl["label"].value_counts()}')

#%% Check if vespa scores from Armstrong are the same as in the Q1-Q17 DR25 FPP catalog

fpp_koi_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                          'q1_q17_dr25_koifpp_2020.12.30_15.14.21.csv', header=39)

armstrong_koi_tbl = pd.read_csv('/data5/tess_project/Data/tables_armstrong_50exoplnt/TableA1.csv')

vespa_fpp_tbl = pd.DataFrame(columns=['kepoi_name', 'fpp_prob_archive', 'fpp_prob_armstrong'])

for koi_i, koi in fpp_koi_tbl.iterrows():

    koi_found = armstrong_koi_tbl.loc[armstrong_koi_tbl['kepoi_name'] == koi['kepoi_name']]

    assert len(koi_found) == 1

    vespa_fpp_tbl = pd.concat([vespa_fpp_tbl, pd.DataFrame(data={'kepoi_name': koi['kepoi_name'],
                                                                 'fpp_prob_archive': koi['fpp_prob'],
                                                                 'fpp_prob_armstrong': koi_found['fpp_prob']})])

print(f'Vespa scores are the same: {vespa_fpp_tbl["fpp_prob_archive"].equals(vespa_fpp_tbl["fpp_prob_armstrong"])}')

vespa_fpp_tbl['difference'] = \
    vespa_fpp_tbl[['fpp_prob_archive', 'fpp_prob_armstrong']].apply(lambda x: np.abs(x[0] - x[1]), axis=1)
print(f'Number of mismatches: {len(vespa_fpp_tbl.loc[vespa_fpp_tbl["difference"] > 1e-5])}')

vespa_fpp_tbl.to_csv('/data5/tess_project/Data/tables_armstrong_50exoplnt/comparing_vespa_scores.csv', index=False)

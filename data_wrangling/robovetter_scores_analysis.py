"""
Script used to analyse the Robovetter results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from tensorflow.keras.metrics import AUC, Precision, Recall

#%% Update Cumulative KOI list with the dispositions from the Certified False Positive list

cumKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                        'cumulative_2020.02.21_10.29.22.csv', header=90)
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

# initialize fpwg disposition column in the Cumulative KOI table
cumKoiTbl['fpwg_disp_status'] = ''

countUpdated = 0
for koi_i, koi in cfpTbl.iterrows():

    # validKoi = cumKoiTbl.loc[cumKoiTbl['kepoi_name'] == koi.kepoi_name]

    # assert len(validKoi) <= 1

    # if len(validKoi) == 1:
    cumKoiTbl.loc[cumKoiTbl['kepoi_name'] == koi.kepoi_name, 'fpwg_disp_status'] = koi['fpwg_disp_status']
    countUpdated += 1

print('Number of KOIs in the Cumulative KOI list which were matched to a KOI in the Certified False Positive list: '
      '{}'.format(countUpdated))

cumKoiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/cumulative_2020.02.21_10.29.22_'
                 'fpwgdisp.csv', index=False)

#%% Plot histograms of Robovetter score for each disposition

# filter out KOIs not associated with TCEs from Q1-Q17 DR25
cumKoiTbl = cumKoiTbl.loc[cumKoiTbl['koi_tce_delivname'] == 'q1_q17_dr25_tce']

cumKoiDisp = cumKoiTbl['koi_disposition'].unique()
cfpKoiDisp = cumKoiTbl['fpwg_disp_status'].unique()

bins = np.linspace(0, 1, num=11, endpoint=True)
fontSize = 15
# ymax = 5000
for disp in np.concatenate([cumKoiDisp, cfpKoiDisp]):
# for disp in ['CERTIFIED FP']:

    if disp in cumKoiDisp:
        scores = cumKoiTbl.loc[cumKoiTbl['koi_disposition'] == disp]['koi_score']
    else:
        # if disp == 'CERTIFIED FP':
        #     scores = cumKoiTbl.loc[cumKoiTbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA'])]['koi_score']
        # elif disp == 'CERTIFIED FA':
        #     continue
        # else:
        scores = cumKoiTbl.loc[cumKoiTbl['fpwg_disp_status'] == disp]['koi_score']

    f, ax = plt.subplots(figsize=(12, 8))
    ax.hist(scores, bins, edgecolor='k')
    ax.set_xlabel('Robovetter score', size=fontSize)
    ax.set_ylabel('Number of KOIs', size=fontSize)
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.tick_params(labelsize=15)
    # ax.set_ylim(top=ymax)
    # if disp == 'CERTIFIED FP':
    #     ax.set_title('{}'.format('CERTIFIED FP+FA'), size=fontSize)
    # else:
    ax.set_title('{}'.format(disp), size=fontSize)
    ax.grid(True)
    # aaa
    # if disp == 'CERTIFIED FP':
    #     f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_scores_analysis/'
    #               'hist_robovetterscore_{}.png'.format('CERTIFIED FP+FA'))
    # else:
    f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_scores_analysis/'
              'hist_robovetterscore_{}.png'.format(disp))
    plt.close()

#%% compute accuracy, precision, recall for Robovetter

mapDictDisp = {'Non-KOI': 0,
               'CANDIDATE': 1,
               'CONFIRMED': 1,
               'FALSE POSITIVE': 0}

mapDictLabel = {'PC': 1,
                'AFP': 0,
                'NTP': 0}


def disposition_to_label(disp, mapDict):

    return mapDict[disp]


tceTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                     'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/testset.csv')

tceTbl.loc[(tceTbl['koi_disposition'].isna()), ['koi_disposition']] = 'Non-KOI'

tceTbl['koi_disposition_bin'] = tceTbl['koi_disposition'].apply(disposition_to_label, args=(mapDictDisp,))
tceTbl['label_bin'] = tceTbl['label'].apply(disposition_to_label, args=(mapDictLabel,))

print(classification_report(tceTbl['label_bin'], tceTbl['koi_disposition_bin'], target_names=['non-PC', 'PC'],
                            digits=4))
clfReportDict = classification_report(tceTbl['label_bin'], tceTbl['koi_disposition_bin'], target_names=['non-PC', 'PC'],
                                      output_dict=True)

#%% compute AUC ROC, AUC PR for the Robovetter scores

columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']

robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ', skipinitialspace=False)

# print(robovetterTceTbl.head())

robovetterTceTbl['target_id'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0]))
robovetterTceTbl['tce_plnt_num'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1]))

tceTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                     'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/testset.csv')

for tce_i, tce in tceTbl.iterrows():

    foundTce = robovetterTceTbl.loc[(robovetterTceTbl['target_id'] == tce['target_id']) &
                                    (robovetterTceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    assert len(foundTce) > 0

    tceTbl.loc[tce_i, ['koi_score']] = foundTce.Robovetter_Score.values[0]

assert not tceTbl['koi_score'].isna().any()

tceTbl['label_bin'] = tceTbl['label'].apply(disposition_to_label, args=(mapDictLabel,))

num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds))

auc_pr = AUC(num_thresholds=num_thresholds,
                           summation_method='interpolation',
                           curve='PR',
                           name='auc_pr')
auc_roc = AUC(num_thresholds=num_thresholds,
                            summation_method='interpolation',
                            curve='ROC',
                            name='auc_roc')
precision_thr = Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
recall_thr = Recall(thresholds=threshold_range, top_k=None, name='rec_thr')

_ = auc_pr.update_state(tceTbl['label_bin'], tceTbl['koi_score'])
print(auc_pr.result().numpy())

_ = auc_roc.update_state(tceTbl['label_bin'], tceTbl['koi_score'])
print(auc_roc.result().numpy())

# save precision and recall values for each threshold
_ = precision_thr.update_state(tceTbl['label_bin'], tceTbl['koi_score'])
precision_thr_arr = precision_thr.result().numpy()
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/precision_thr.npy',
        precision_thr_arr)

_ = recall_thr.update_state(tceTbl['label_bin'], tceTbl['koi_score'])
recall_thr_arr = recall_thr.result().numpy()
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/recall_thr.npy',
        recall_thr_arr)

f, ax = plt.subplots(2, 1)
ax[0].plot(threshold_range, precision_thr_arr)
ax[0].scatter(threshold_range, precision_thr_arr, s=10)
ax[0].set_ylabel('Precision')
ax[1].plot(threshold_range, recall_thr_arr)
ax[1].scatter(threshold_range, recall_thr_arr, s=10)
ax[1].set_ylabel('Recall')
ax[1].set_xlabel('Threshold Range')

#%% Compute AUC PR for KOIs only

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobobvetterKOIs.csv')
tceTblTestSet = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/testset.csv')

# remove non-KOIs
tceTbl = tceTbl.loc[~tceTbl['kepoi_name'].isna()]

rankingScores, rankingLabels = [], []
for tce_i, tce in robovetterTceTbl.iterrows():

    # it is a KOI
    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]
    # it is in the test set
    foundTce2 = tceTblTestSet.loc[(tceTblTestSet['target_id'] == tce['target_id']) &
                                  (tceTblTestSet['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(foundTce) == 1 and len(foundTce2) == 1:
        rankingScores.append(tce['Robovetter_Score'])
        tceLabel = 1 if foundTce['label'].values[0] == 'PC' else 0
        rankingLabels.append(tceLabel)

num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds))

auc_pr = AUC(num_thresholds=num_thresholds,
                           summation_method='interpolation',
                           curve='PR',
                           name='auc_pr')

_ = auc_pr.update_state(rankingLabels, rankingScores)
print(auc_pr.result().numpy())

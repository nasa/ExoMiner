"""
Script used to analyse the Robovetter results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from tensorflow.keras.metrics import AUC, Precision, Recall, FalsePositives, BinaryAccuracy

#%% auxiliary functions


def disposition_to_label(disp, mapDict):

    return mapDict[disp]

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

#%% compute accuracy, precision, recall for Robovetter - THIS IS WRONG! METRICS ARE COMPUTED WITH RESPECT TO DIFFERENT
# LABELS THAN OUR DATASET
#
# mapDictDisp = {'Non-KOI': 0,
#                'CANDIDATE': 1,
#                'CONFIRMED': 1,
#                'FALSE POSITIVE': 0}
#
# mapDictLabel = {'PC': 1,
#                 'AFP': 0,
#                 'NTP': 0}
#
#
# def disposition_to_label(disp, mapDict):
#
#     return mapDict[disp]
#
#
# tceTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
#                      'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/testset.csv')
#
# tceTbl.loc[(tceTbl['koi_disposition'].isna()), ['koi_disposition']] = 'Non-KOI'
#
# tceTbl['koi_disposition_bin'] = tceTbl['koi_disposition'].apply(disposition_to_label, args=(mapDictDisp,))
# tceTbl['label_bin'] = tceTbl['label'].apply(disposition_to_label, args=(mapDictLabel,))
#
# print(classification_report(tceTbl['label_bin'], tceTbl['koi_disposition_bin'], target_names=['non-PC', 'PC'],
#                             digits=4))
# clfReportDict = classification_report(tceTbl['label_bin'], tceTbl['koi_disposition_bin'], target_names=['non-PC', 'PC'],
#                                       output_dict=True)

#%% compute AUC ROC, AUC PR for the Robovetter scores

columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']

robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/'
                               'robovetter_analysis/kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ', skipinitialspace=False)

# print(robovetterTceTbl.head())

robovetterTceTbl['target_id'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0]))
robovetterTceTbl['tce_plnt_num'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1]))

tceTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                     'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/valset.csv')

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

#%% Compute AUC ROC and AUC PR and plot PR curve and ROC

# load Q1-Q17 DR25 TCE table with the dispositions used in our experiments
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

# load and prepare Robovetter Q1-Q17 DR25 TCE (no rogue TCEs) table
# columns of interest from the Robovetter TCE table
columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']
robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                               'kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ',
                               skipinitialspace=False)
# print(robovetterTceTbl.head())
# create target_id and tce_plnt_num columns
robovetterTceTbl['target_id'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0]))
robovetterTceTbl['tce_plnt_num'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1]))

# load dataset TCE table
dataset = 'train'
datasetTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/'
                         '{}set.csv'.format(dataset))

# initialize label and score columns
datasetTbl['label'] = ''
datasetTbl['score'] = np.nan
labelMap = {'PC': 1, 'AFP': 0, 'NTP': 0}
# add labels and scores from the TCE table and the Robovetter TCE table, respectively
for tce_i, tce in datasetTbl.iterrows():

    foundTceRobovetter = robovetterTceTbl.loc[(robovetterTceTbl['target_id'] == tce['target_id']) &
                                              (robovetterTceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    assert len(foundTce) == 1 and len(foundTceRobovetter) == 1

    # add Robovetter score
    datasetTbl.loc[tce_i, ['score']] = foundTceRobovetter.Robovetter_Score.values[0]
    # add integer label
    datasetTbl.loc[tce_i, ['label']] = labelMap[foundTce.label.values[0]]

assert not datasetTbl['score'].isna().any()
assert (datasetTbl['label'] != '').all()

# add small score so that minimum is above 0 so the ROC goes to a FPR of 1
score_fuzzy_factor = 1e-6
datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x + score_fuzzy_factor if x < 1-score_fuzzy_factor else x)
# datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x - score_fuzzy_factor if x > 1-score_fuzzy_factor else x)

# define thresholds used to compute the metrics
num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

# compute AUC ROC, AUC PR
auc_pr = AUC(num_thresholds=num_thresholds,
                           summation_method='interpolation',
                           curve='PR',
                           name='auc_pr')
auc_roc = AUC(num_thresholds=num_thresholds,
                            summation_method='interpolation',
                            curve='ROC',
                            name='auc_roc')
_ = auc_pr.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
auc_pr = auc_pr.result().numpy()
_ = auc_roc.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
auc_roc = auc_roc.result().numpy()
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
        'aucroc_aucpr_{}.npy'.format(dataset), {'AUC ROC': auc_roc, 'AUC PR': auc_pr})

# compute precision and recall for the PR curve
precision_thr = Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
recall_thr = Recall(thresholds=threshold_range, top_k=None, name='rec_thr')
_ = precision_thr.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
precision_thr_arr = precision_thr.result().numpy()
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
        'precision_thr_{}.npy'.format(dataset), precision_thr_arr)
_ = recall_thr.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
recall_thr_arr = recall_thr.result().numpy()
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/recall_thr_{}.npy'.format(dataset),
        recall_thr_arr)

# compute TPR and FPR for the ROC
false_pos_thr = FalsePositives(thresholds=threshold_range, name='prec_thr')
_ = false_pos_thr.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
false_pos_thr_arr = false_pos_thr.result().numpy()
fpr_thr_arr = false_pos_thr_arr / len(datasetTbl.loc[datasetTbl['label'] == 0])
np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
        'fpr_thr_{}.npy'.format(dataset), fpr_thr_arr)

# plot PR curve
f, ax = plt.subplots()
ax.plot(recall_thr_arr, precision_thr_arr)
# ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.text(0.8, 0.1, 'AUC={:.3}'.format(auc_pr), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
          'precision-recall_curve_{}.svg'.format(dataset))

# plot PR curve
f, ax = plt.subplots()
ax.plot(recall_thr_arr, precision_thr_arr)
# ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 21))
ax.set_yticks(np.linspace(0, 1, 21))
ax.set_xlim([0.5, 1])
ax.set_ylim([0.5, 1])
ax.text(0.8, 0.6, 'AUC={:.3}'.format(auc_pr), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
          'precision-recall_curve_zoom_{}.svg'.format(dataset))

# plot ROC
f, ax = plt.subplots()
ax.plot(fpr_thr_arr, recall_thr_arr)
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.text(0.8, 0.1, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/roc_{}.svg'.format(dataset))

f, ax = plt.subplots()
ax.plot(fpr_thr_arr, recall_thr_arr)
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 21))
ax.set_yticks(np.linspace(0, 1, 21))
ax.set_xlim([0, 0.5])
ax.set_ylim([0.5, 1])
ax.text(0.3, 0.6, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/roc_zoom_{}.svg'.format(dataset))

#%% Compute precision at k for Robovetter

# load Q1-Q17 DR25 TCE table with the dispositions used in our experiments
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

# load and prepare Robovetter Q1-Q17 DR25 TCE (no rogue TCEs) table
# columns of interest from the Robovetter TCE table
columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']
robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                               'kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ',
                               skipinitialspace=False)
# print(robovetterTceTbl.head())
# create target_id and tce_plnt_num columns
robovetterTceTbl['target_id'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0]))
robovetterTceTbl['tce_plnt_num'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1]))

# load dataset TCE table
dataset = 'train'
datasetTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/'
                         '{}set.csv'.format(dataset))

datasetTbl['label'] = ''
datasetTbl['score'] = np.nan
datasetTbl['Robovetter_disposition'] = ''
labelMap = {'PC': 1, 'AFP': 0, 'NTP': 0}
labelMapRobovetter = {'PC': 1, 'FP': 0}
for tce_i, tce in datasetTbl.iterrows():

    foundTceRobovetter = robovetterTceTbl.loc[(robovetterTceTbl['target_id'] == tce['target_id']) &
                                              (robovetterTceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    assert len(foundTce) == 1 and len(foundTceRobovetter) == 1

    # add Robovetter score
    datasetTbl.loc[tce_i, ['score']] = foundTceRobovetter.Robovetter_Score.values[0]
    # add Robovetter disposition
    datasetTbl.loc[tce_i, ['Robovetter_disposition']] = labelMapRobovetter[foundTceRobovetter.Disposition.values[0]]

    # add integer label
    datasetTbl.loc[tce_i, ['label']] = labelMap[foundTce.label.values[0]]

assert not datasetTbl['score'].isna().any()
assert (datasetTbl['label'] != '').all()

score_fuzzy_factor = 1e-6
datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x + score_fuzzy_factor if x < 1-score_fuzzy_factor else x)
# datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x - score_fuzzy_factor if x > 1-score_fuzzy_factor else x)

# order by ascending score
datasetTblOrd = datasetTbl.sort_values('score', axis=0, ascending=True)

# compute precision at k
k_arr = {'train': [100, 1000, 2000], 'val': [50, 150, 250], 'test': [50, 150, 250]}
precision_at_k = {k: np.nan for k in k_arr[dataset]}
for k_i in range(len(k_arr[dataset])):
    if len(datasetTbl) < k_arr[dataset][k_i]:
        precision_at_k[k_arr[dataset][k_i]] = np.nan
    else:
        precision_at_k[k_arr[dataset][k_i]] = \
            np.sum(datasetTblOrd['label'][-k_arr[dataset][k_i]:]) / k_arr[dataset][k_i]

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
        'precision_at_k_{}.npy'.format(dataset), precision_at_k)

# compute precision at k curve
k_arr = np.linspace(25, 250, 10, endpoint=True, dtype='int')
precision_at_k = {k: np.nan for k in k_arr}
for k_i in range(len(k_arr)):
    if len(datasetTbl) < k_arr[k_i]:
        precision_at_k[k_arr[k_i]] = np.nan
    else:
        precision_at_k[k_arr[k_i]] = \
            np.sum(datasetTblOrd['label'][-k_arr[k_i]:]) / k_arr[k_i]

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
        'precision_at_k_curve_{}.npy'.format(dataset), precision_at_k)

f, ax = plt.subplots()
ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
ax.set_ylabel('Precision')
ax.set_xlabel('Top-K')
ax.grid(True)
# ax.set_xticks(np.linspace(k_arr[0], k_arr[-1], 11, endpoint=True))
ax.set_xticks(np.linspace(25, 250, 10, endpoint=True, dtype='int'))
# ax.set_yticks(np.linspace(0, 1, 21))
ax.set_xlim([k_arr[0], k_arr[-1]])
ax.set_ylim(top=1)
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
          'precision_at_k_{}.svg'.format(dataset))

#%%

#%% Compute AUC ROC and AUC PR and plot PR curve and ROC

# load Q1-Q17 DR25 TCE table with the dispositions used in our experiments
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

# load and prepare Robovetter Q1-Q17 DR25 TCE (no rogue TCEs) table
# columns of interest from the Robovetter TCE table
columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']
robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                               'kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ',
                               skipinitialspace=False)
# print(robovetterTceTbl.head())
# create target_id and tce_plnt_num columns
robovetterTceTbl['target_id'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0]))
robovetterTceTbl['tce_plnt_num'] = robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1]))

# load dataset TCE table
dataset = 'train'
datasetTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                         'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/'
                         '{}set.csv'.format(dataset))

# initialize label and score columns
datasetTbl['label'] = ''
datasetTbl['score'] = np.nan
labelMap = {'PC': 1, 'AFP': 0, 'NTP': 0}
datasetTbl['Robovetter_disposition'] = ''
labelMapRobovetter = {'PC': 1, 'FP': 0}
# add labels and scores from the TCE table and the Robovetter TCE table, respectively
for tce_i, tce in datasetTbl.iterrows():

    foundTceRobovetter = robovetterTceTbl.loc[(robovetterTceTbl['target_id'] == tce['target_id']) &
                                              (robovetterTceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    assert len(foundTce) == 1 and len(foundTceRobovetter) == 1

    # add Robovetter score
    datasetTbl.loc[tce_i, ['score']] = foundTceRobovetter.Robovetter_Score.values[0]
    # add Robovetter disposition
    datasetTbl.loc[tce_i, ['Robovetter_disposition']] = labelMapRobovetter[foundTceRobovetter.Disposition.values[0]]
    # add integer label
    datasetTbl.loc[tce_i, ['label']] = labelMap[foundTce.label.values[0]]

assert not datasetTbl['score'].isna().any()
assert (datasetTbl['label'] != '').all()

# add small score so that minimum is above 0 so the ROC goes to a FPR of 1
score_fuzzy_factor = 1e-6
datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x + score_fuzzy_factor if x < 1-score_fuzzy_factor else x)
# datasetTbl['score'] = datasetTbl['score'].apply(lambda x: x - score_fuzzy_factor if x > 1-score_fuzzy_factor else x)

# define thresholds used to compute the metrics
num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

# compute AUC ROC, AUC PR
auc_pr = AUC(num_thresholds=num_thresholds,
                           summation_method='interpolation',
                           curve='PR',
                           name='auc_pr')
auc_roc = AUC(num_thresholds=num_thresholds,
                            summation_method='interpolation',
                            curve='ROC',
                            name='auc_roc')

precision = Precision(name='precision')
recall = Recall(name='recall')

binary_accuracy = BinaryAccuracy(name='binary_accuracy')

_ = auc_pr.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
auc_pr = auc_pr.result().numpy()
_ = auc_roc.update_state(datasetTbl['label'].tolist(), datasetTbl['score'].tolist())
auc_roc = auc_roc.result().numpy()

_ = precision.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
precision = precision.result().numpy()
_ = recall.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
recall = recall.result().numpy()

_ = binary_accuracy.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
binary_accuracy = binary_accuracy.result().numpy()

robovetter_metrics = {'AUC PR': auc_pr, 'AUC ROC': auc_roc, 'Precision': precision, 'Recall': recall,
                      'Binary Accuracy': binary_accuracy}

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/metrics_{}set'.format(dataset),
        robovetter_metrics)

print(robovetter_metrics)

#%% Create Robovetter TCE table with our labels and also split it into training, validation and test tables

# load Q1-Q17 DR25 TCE table with the dispositions used in our experiments
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

# load and prepare Robovetter Q1-Q17 DR25 TCE (no rogue TCEs) table
# columns of interest from the Robovetter TCE table
columnNames = ['TCE', 'Robovetter_Score', 'Disposition', 'Not_Transit-Like_Flag', 'Stellar_Eclipse_Flag',
               'Centroid Offset_Flag', 'Ephemeris_Match_Flag', 'Minor_Descriptive_Flags']
robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                               'kplr_dr25_obs_robovetter_output.txt', skiprows=1, names=columnNames, sep=' ',
                               skipinitialspace=False)

# rename Disposition column to Robovetter_Disposition
robovetterTceTbl.rename(columns={'Disposition': 'Robovetter_Disposition'}, inplace=True)

# create target_id and tce_plnt_num columns
robovetterTceTbl.insert(0, 'target_id', robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[0])))
robovetterTceTbl.insert(1,'tce_plnt_num', robovetterTceTbl['TCE'].apply(lambda x: int(x.split('-')[1])))
robovetterTceTbl.drop(columns='TCE', inplace=True)

# initialize label and original_label columns
robovetterTceTbl['original_label'] = ''
robovetterTceTbl['label'] = -1
labelMap = {'PC': 1, 'AFP': 0, 'NTP': 0}
# add labels and scores from the TCE table and the Robovetter TCE table, respectively
for tce_i, tce in robovetterTceTbl.iterrows():

    print('Checking TCE {}/{}'.format(tce_i, len(robovetterTceTbl)))
    foundTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    assert len(foundTce) <= 1

    if len(foundTce) == 1:
        # # add integer label
        # tce['label'] = labelMap[foundTce.label.values[0]]
        # # add label
        # tce['original_label'] = foundTce.label.values[0]
        #
        # robovetterTceTbl = pd.concat([robovetterTceTbl, tce], axis=0)

        # add label
        robovetterTceTbl.loc[tce_i, ['original_label']] = foundTce.label.values[0]

        # add integer label
        robovetterTceTbl.loc[tce_i, ['label']] = labelMap[foundTce.label.values[0]]

# drop TCEs not part of our TCE table
robovetterTceTbl = robovetterTceTbl.loc[robovetterTceTbl['original_label'] != '']
assert (robovetterTceTbl['original_label'] != '').all()
assert (robovetterTceTbl['label'] != '').all()

robovetterTceTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                        'kplr_dr25_obs_robovetter_with_our_labels.csv', index=False)

# split Robovetter TCE table into training, validation and test tables according to the split in the experiments
datasets = ['train', 'val', 'test']
# robovetterTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
#                                'kplr_dr25_obs_robovetter_with_our_labels.csv')
for dataset in datasets:

    robovetterDatasetTbl = robovetterTceTbl.copy(deep=True)
    robovetterDatasetTbl['in'] = 0

    # load dataset TCE table
    datasetTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                             'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/'
                             'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/'
                             '{}set.csv'.format(dataset))

    # add labels and scores from the TCE table and the Robovetter TCE table, respectively
    for tce_i, tce in datasetTbl.iterrows():
        print('TCE {}/{}'.format(tce_i, len(datasetTbl)))
        foundTce = robovetterDatasetTbl.loc[
            (robovetterDatasetTbl['target_id'] == tce['target_id']) & (robovetterDatasetTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

        assert len(foundTce) <= 1

        # add TCE to Robovetter dataset table
        if len(foundTce) == 1:
            robovetterDatasetTbl['in'] = 1
        else:
            robovetterDatasetTbl['in'] = 0

    robovetterDatasetTbl = robovetterDatasetTbl.loc[robovetterDatasetTbl['in'] == 1]
    robovetterDatasetTbl.drop(columns='in', inplace=True)

    # order TCEs by descending score
    robovetterDatasetTbl.sort_values('Robovetter_Score', ascending=False, inplace=True)
    robovetterDatasetTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/'
                                'kplr_dr25_obs_robovetter_with_our_labels_{}.csv'.format(dataset), index=False)

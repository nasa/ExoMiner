""" Match TCEs in our model rankings to those dispositioned as discovered exoplanets [1] and those having high fpp
probability values. Also add these fpp probability values for all TCEs that it is available.

[1] Armstrong, David J., Jevgenij Gamper, and Theodoros Damoulas. "Exoplanet Validation with Machine Learning: 50 new
validated Kepler planets." Monthly Notices of the Royal Astronomical Society (2020).
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, Precision, Recall, FalsePositives, BinaryAccuracy

#%%

experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                      'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_'
                      'wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_'
                      'prad_per')

#%% Add KOI names to the ranking in order to match with their tables

ranking_tbl = pd.read_csv(experiment_dir / 'ensemble_ranked_predictions_predictset.csv')

# add kepoi names
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                      'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval.csv')

ranking_tbl['kepoi_name'] = ''
ranking_tbl['kepler_name'] = ''

for tce_i, tce in ranking_tbl.iterrows():
    ranking_tbl.loc[tce_i, ['kepoi_name', 'kepler_name']] = \
        tce_tbl.loc[(tce_tbl['target_id'] == tce['target_id']) &
                    (tce_tbl['tce_plnt_num'] == tce['tce_plnt_num']),
                    ['kepoi_name', 'kepler_name']].values[0]

# ranking_tbl.to_csv('/home/msaragoc/Downloads/a.csv', index=False)

#%% add column for 50 exoplanets found

exoplnt_tbl = pd.read_csv('/home/msaragoc/Downloads/tables_paper/Table7.csv')

ranking_tbl['kepler_name_new'] = ''
ranking_tbl['fpp_prob'] = np.nan
ranking_tbl['in_50_exoplnt'] = 'no'

for tce_i, tce in ranking_tbl.iterrows():
    koi_found = exoplnt_tbl.loc[exoplnt_tbl['kepoi_name'] == tce['kepoi_name'], ['kep_num', 'fpp_prob']]
    if len(koi_found) == 0:
        continue
    ranking_tbl.loc[tce_i, ['kepler_name_new', 'fpp_prob', 'in_50_exoplnt']] = np.concatenate((koi_found.values[0],
                                                                                               ['yes']))

# ranking_tbl.to_csv('/home/msaragoc/Downloads/b.csv', index=False)

#%% add column for KOIs that did not meet fpp threshold

exoplnt_tbl = pd.read_csv('/home/msaragoc/Downloads/tables_paper/Table8.csv')

ranking_tbl['in_high_fpp'] = 'no'

for tce_i, tce in ranking_tbl.iterrows():
    koi_found = exoplnt_tbl.loc[exoplnt_tbl['kepoi_name'] == tce['kepoi_name'], ['fpp_prob']]
    if len(koi_found) == 0:
        continue
    ranking_tbl.loc[tce_i, ['fpp_prob', 'in_high_fpp']] = np.concatenate((koi_found.values[0], ['yes']))

# ranking_tbl.to_csv('/home/msaragoc/Downloads/c.csv', index=False)

#%% add all possible fpp_prob values

exoplnt_tbl = pd.read_csv('/home/msaragoc/Downloads/tables_paper/TableA1.csv')

for tce_i, tce in ranking_tbl.iterrows():
    koi_found = exoplnt_tbl.loc[exoplnt_tbl['kepoi_name'] == tce['kepoi_name'], ['fpp_prob']]
    if len(koi_found) == 0:
        continue
    ranking_tbl.loc[tce_i, ['fpp_prob']] = koi_found.values[0]

ranking_tbl.to_csv(experiment_dir / 'comparison_ranking_50exoplntpaper'/ 'ranking_comparison_with_paper_12-15-2020.csv',
                   index=False)

#%%

experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                      'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_'
                      'wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_'
                      'prad_per/')

print('Joining training, validation, test and not-used TCE rankings...')
datasets = ['train', 'val', 'test']
ranking_tbls = [pd.read_csv(experiment_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
                for dataset in datasets]
# add column indicating from which set the TCEs came from
for tbl_i in range(len(ranking_tbls)):
    ranking_tbls[tbl_i]['dataset'] = datasets[tbl_i]

# join rankings
ranking_tbl = pd.concat(ranking_tbls, axis=0).reset_index(drop=True)

# merge with ranking with TCEs not used
ranking_tbl_notused = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                                  'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_'
                                  'secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_'
                                  'nopps_ckoiper_secparams_prad_per_predtcesnotused/'
                                  'ensemble_ranked_predictions_predictset.csv')
ranking_tbl_notused['dataset'] = 'not_used'
ranking_tbl = pd.concat([ranking_tbl, ranking_tbl_notused], axis=0).reset_index(drop=True)

# get only fields of interest
ranking_tbl = ranking_tbl[['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
                           'label', 'predicted class', 'score', 'dataset']]

# add kepoi names
print('Adding kepoi names...')
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                      'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                      'nomissingval.csv')
ranking_tbl['kepoi_name'] = ''
ranking_tbl['kepler_name'] = ''
for tce_i, tce in ranking_tbl.iterrows():
    koi_found = tce_tbl.loc[(tce_tbl['target_id'] == tce['target_id']) &
                            (tce_tbl['tce_plnt_num'] == tce['tce_plnt_num']), ['kepoi_name', 'kepler_name']]
    if len(koi_found) > 0:
        ranking_tbl.loc[tce_i, ['kepoi_name', 'kepler_name']] = koi_found.values[0]

# add data from Table4 (paper TCE table)
print('Adding data from paper TCE table...')
tce_tbl_paper = pd.read_csv('/data5/tess_project/Data/tables_50exoplnt_paper/Table4.csv')
cols_tbl = ['GPC_score', 'MLP_score', 'RFC_score', 'ET_score', 'PP_GPC', 'PP_RFC', 'PP_MLP', 'PP_ET',
            'planet', 'targetEB', 'targetHEB', 'targetHTP', 'backgroundBEB', 'backgroundBTP', 'secondsource',
            'nonastro', 'Binary', 'State', 'gaia', 'roboAO', 'MES', 'outlier_score_LOF', 'outlier_score_IF', 'class']
for col in cols_tbl:
    ranking_tbl[col] = np.nan
for tce_i, tce in ranking_tbl.iterrows():
    tce_found = tce_tbl_paper.loc[tce_tbl_paper['tce_id'] == f'{tce["target_id"]}_{tce["tce_plnt_num"]}',
                                  cols_tbl].values[0]
    if len(tce_found) > 0:
        ranking_tbl.loc[tce_i, cols_tbl] = tce_found

# add data from TableA1 (paper KOI table)
print('Adding data from paper KOI table...')
tce_tbl_paper = pd.read_csv('/data5/tess_project/Data/tables_50exoplnt_paper/TableA1.csv')
cols_tbl = ['PP_GP_final', 'PP_RFC_final', 'PP_MLP_final', 'PP_ET_final', 'fpp_prob', 'pp_host_rel_prob', 'posiscore',
            'outlier_score_LOF', 'outlier_score_IF']
cols_tbl_new = [col if col not in ['outlier_score_LOF', 'outlier_score_IF'] else f'{col}_koilist' for col in cols_tbl]
for col in cols_tbl_new:
    ranking_tbl[col] = np.nan
for tce_i, tce in ranking_tbl.iterrows():
    koi_found = tce_tbl_paper.loc[tce_tbl_paper['kepoi_name'] == tce['kepoi_name'], cols_tbl]
    if len(koi_found) > 0:
        ranking_tbl.loc[tce_i, cols_tbl_new] = koi_found.values[0]

# add data from Table7 (paper 50 exoplanets found)
print('Adding data from paper 50 exoplanets table...')
tce_tbl_paper = pd.read_csv('/data5/tess_project/Data/tables_50exoplnt_paper/Table7.csv')
ranking_tbl['kepler_name_new'] = ''
ranking_tbl['fpp_prob_recalc'] = np.nan
ranking_tbl['sublist'] = 'no'
for tce_i, tce in ranking_tbl.iterrows():
    koi_found = tce_tbl_paper.loc[tce_tbl_paper['kepoi_name'] == tce['kepoi_name'], ['kep_num', 'fpp_prob']]
    if len(koi_found) == 0:
        continue
    ranking_tbl.loc[tce_i, ['kepler_name_new', 'fpp_prob_recalc', 'sublist']] = np.concatenate((koi_found.values[0],
                                                                                                ['in_50_exoplnt']))
# add data from Table8 (paper TCEs high fpp prob value)
print('Adding data from paper TCEs with high fpp prob value table...')
tce_tbl_paper = pd.read_csv('/data5/tess_project/Data/tables_50exoplnt_paper/Table8.csv')
for tce_i, tce in ranking_tbl.iterrows():
    koi_found = tce_tbl_paper.loc[tce_tbl_paper['kepoi_name'] == tce['kepoi_name'], ['fpp_prob']]
    if len(koi_found) == 0:
        continue
    ranking_tbl.loc[tce_i, ['fpp_prob_recalc', 'sublist']] = np.concatenate((koi_found.values[0], ['in_high_fpp']))

ranking_tbl.to_csv(experiment_dir / 'comparison_ranking_50exoplntpaper'/ 'ranking_comparison_with_paper_12-18-2020.csv',
                   index=False)

#%% Calculate PR-Curve, Precision at k and Misclassified at k for GPC and RFC

ranking_tbl = pd.read_csv(experiment_dir / 'comparison_ranking_50exoplntpaper'/
                          'ranking_comparison_with_paper_12-18-2020.csv')

# define thresholds used to compute the metrics
num_thresholds = 1000
threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

# k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
k_arr = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # no PPs
k_curve_arr = {
    # 'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
    # 'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    'train': np.linspace(180, 1800, 100, endpoint=True, dtype='int'),  # no PPs
    'val': np.linspace(20, 220, 21, endpoint=True, dtype='int'),
    'test': np.linspace(10, 250, 25, endpoint=True, dtype='int'),
}
k_curve_arr_plot = {
    # 'train': np.linspace(200, 2000, 10, endpoint=True, dtype='int'),
    # 'val': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    'train': np.linspace(180, 1800, 10, endpoint=True, dtype='int'),  # no PPs
    'val': np.linspace(20, 220, 21, endpoint=True, dtype='int'),
    'test': np.linspace(10, 250, 25, endpoint=True, dtype='int')
}

clf = 'PP_RFC_final'

# load dataset TCE table
datasets = ['train', 'val', 'test']
for dataset in datasets:

    datasetTbl = ranking_tbl.loc[ranking_tbl['dataset'] == dataset]
    datasetTbl = ranking_tbl.loc[~ranking_tbl[clf].isna()]
    print(f'Number of TCEs in the {dataset} set = {len(datasetTbl)}')

    # compute metrics
    auc_pr = AUC(num_thresholds=num_thresholds,
                               summation_method='interpolation',
                               curve='PR',
                               name='auc_pr')
    auc_roc = AUC(num_thresholds=num_thresholds,
                                summation_method='interpolation',
                                curve='ROC',
                                name='auc_roc')

    # precision = Precision(name='precision')
    # recall = Recall(name='recall')

    # binary_accuracy = BinaryAccuracy(name='binary_accuracy')

    _ = auc_pr.update_state(datasetTbl['label'].tolist(), datasetTbl[clf].tolist())
    auc_pr = auc_pr.result().numpy()
    _ = auc_roc.update_state(datasetTbl['label'].tolist(), datasetTbl[clf].tolist())
    auc_roc = auc_roc.result().numpy()

    # _ = precision.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
    # precision = precision.result().numpy()
    # _ = recall.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
    # recall = recall.result().numpy()

    # _ = binary_accuracy.update_state(datasetTbl['label'].tolist(), datasetTbl['Robovetter_disposition'].tolist())
    # binary_accuracy = binary_accuracy.result().numpy()

    armstrong_metrics = {
        'AUC PR': auc_pr,
        'AUC ROC': auc_roc,
        # 'Precision': precision,
        # 'Recall': recall,
        # 'Binary Accuracy': binary_accuracy
    }
    print(armstrong_metrics)

    # compute precision and recall thresholds for the PR curve
    precision_thr = Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
    recall_thr = Recall(thresholds=threshold_range, top_k=None, name='rec_thr')
    _ = precision_thr.update_state(datasetTbl['label'].tolist(), datasetTbl[clf].tolist())
    precision_thr_arr = precision_thr.result().numpy()
    _ = recall_thr.update_state(datasetTbl['label'].tolist(), datasetTbl[clf].tolist())
    recall_thr_arr = recall_thr.result().numpy()

    armstrong_metrics.update({
        'precision_thr': precision_thr_arr,
        'recall_thr': recall_thr_arr
    })

    # compute FPR for the ROC
    false_pos_thr = FalsePositives(thresholds=threshold_range, name='prec_thr')
    _ = false_pos_thr.update_state(datasetTbl['label'].tolist(), datasetTbl[clf].tolist())
    false_pos_thr_arr = false_pos_thr.result().numpy()
    fpr_thr_arr = false_pos_thr_arr / len(datasetTbl.loc[datasetTbl['label'] == 0])

    armstrong_metrics.update({
        'fpr_thr': fpr_thr_arr
    })

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
    ax.text(0.8, 0.1, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper'/ f'precision-recall_curve_{clf}_{dataset}.svg')
    plt.close()

    # plot PR curve zoomed in
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
    ax.text(0.8, 0.6, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper'/ f'precision-recall_curve_zoom_{dataset}.svg')
    plt.close()

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
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper' / f'roc_{clf}_{dataset}.svg')
    plt.close()

    # plot ROC zoomed in
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
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper' / f'roc_zoom_{clf}_{dataset}.svg')
    plt.close()

    # order by ascending score
    datasetTblOrd = datasetTbl.sort_values(clf, axis=0, ascending=True)
    datasetTblOrd = datasetTblOrd.reset_index(drop=True)

    # compute precision at k
    precision_at_k = {k: np.nan for k in k_arr[dataset]}
    for k_i in range(len(k_arr[dataset])):
        if len(datasetTbl) < k_arr[dataset][k_i]:
            precision_at_k[k_arr[dataset][k_i]] = np.nan
        else:
            precision_at_k[k_arr[dataset][k_i]] = \
                np.sum(datasetTblOrd['label'][-k_arr[dataset][k_i]:]) / k_arr[dataset][k_i]
            print(k_arr[dataset][k_i], np.sum(datasetTblOrd['label'][-k_arr[dataset][k_i]:]),
                  precision_at_k[k_arr[dataset][k_i]])

    armstrong_metrics.update({'precision_at_k': precision_at_k})

    print(precision_at_k)

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr[dataset]}
    for k_i in range(len(k_curve_arr[dataset])):
        if len(datasetTbl) < k_curve_arr[dataset][k_i]:
            precision_at_k[k_curve_arr[dataset][k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[dataset][k_i]] = \
                np.sum(datasetTblOrd['label'][-k_curve_arr[dataset][k_i]:]) / k_curve_arr[dataset][k_i]

    armstrong_metrics.update({'precision_at_k_curve': precision_at_k})

    np.save(experiment_dir / 'comparison_ranking_50exoplntpaper' / f'metrics_{clf}_{dataset}set', armstrong_metrics)

    # plot precision at k curve
    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_yticks(np.linspace(0, 1, 11, endpoint=True))
    ax.set_xlim([k_curve_arr[dataset][0], k_curve_arr[dataset][-1]])
    ax.set_ylim([-0.01, 1.01])
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper'/ f'precision_at_k_{clf}_{dataset}.svg')
    plt.close()

    # plot misclassified examples at k curve
    kvalues = np.array(list(precision_at_k.keys()))
    precvalues = np.array(list(precision_at_k.values()))
    num_misclf_examples = kvalues - kvalues * precvalues
    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(kvalues, num_misclf_examples)
    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_xlim([k_curve_arr[dataset][0], k_curve_arr[dataset][-1]])
    ax.set_ylim(bottom=-0.01)
    f.savefig(experiment_dir / 'comparison_ranking_50exoplntpaper'/ f'misclassified_at_k_{clf}_{dataset}.svg')
    plt.close()

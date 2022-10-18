"""
Computing metrics for TESS CV experiments using different strategies for aggregating TOIs.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy  # , TopKCategoricalAccuracy
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import numpy as np

#%%

num_thresholds = 1000  # number of thresholds used to compute AUC
clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall
class_ids = [0, 1]

#%% Compute metrics when assigning aggregated scores for TCEs associated with the same TOI

top_k_vals = [50, 100, 250, 500, 750, 1000, 1250, 1400]
cats = {'T-CP': 1, 'T-KP': 1, 'T-FP': 0, 'T-EB': 0, 'T-FA': 0, 'T-NTP': 0}

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_smet_ourmatch.csv')

# experiment directory
exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/interns/charles_yates/results_10-4_10-10/cv_merged_fluxvar_noglobal_stellar_10-10-2022_1028')

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')

ranking_tbl = ranking_tbl.merge(tce_tbl[['uid', 'matched_toi_our']], on='uid', how='left', validate='one_to_one')

agg_cols = [
    'matched_toi_our',
    'tce_period',
    'tce_duration',
    'transit_depth',
    'score',
    'predicted class',
]
toi_ranking_tbl = ranking_tbl.loc[~ranking_tbl['matched_toi_our'].isna(), agg_cols].groupby('matched_toi_our').mean().reset_index().rename(columns={'matched_toi_our': 'uid'})
toi_ranking_tbl['matched_toi_our'] = toi_ranking_tbl['uid']
ranking_tbl_aggtois = ranking_tbl.copy()

for toi_i, toi in toi_ranking_tbl.iterrows():
    tcesintoi = ranking_tbl_aggtois['matched_toi_our'] == toi['uid']
    ranking_tbl_aggtois.loc[tcesintoi, 'score'] = toi['score']
    ranking_tbl_aggtois.loc[tcesintoi, 'predicted class'] = 0
    ranking_tbl_aggtois.loc[tcesintoi & (ranking_tbl_aggtois['score'] > clf_threshold), 'predicted class'] = 1

ranking_tbl_aggtois.to_csv(exp_dir / 'ensemble_ranked_predictions_allfolds_tces_score_aggtois.csv', index=False)

# compute metrics for each CV fold
metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
metrics_lst += [f'recall {cat}' for cat in cats]
metrics_lst += ['n_examples', 'n_pos_examples', 'n_neg_examples', 'frac_pos_examples']
metrics_lst += [f'n_{cat}' for cat in cats]

data_to_tbl = {col: [] for col in metrics_lst}

data_to_tbl['n_examples'].append(len(ranking_tbl_aggtois))

data_to_tbl['n_pos_examples'].append(
    (ranking_tbl_aggtois['original_label'].isin([k for k in cats if cats[k] == 1])).sum())
data_to_tbl['frac_pos_examples'].append(data_to_tbl['n_pos_examples'][-1] / data_to_tbl['n_examples'][-1])
data_to_tbl['n_neg_examples'].append(
    (ranking_tbl_aggtois['original_label'].isin([k for k in cats if cats[k] == 0])).sum())

for cat in cats:
    data_to_tbl[f'n_{cat}'].append((ranking_tbl_aggtois['original_label'] == cat).sum())

# compute metrics
auc_pr = AUC(num_thresholds=num_thresholds,
             summation_method='interpolation',
             curve='PR',
             name='auc_pr')
auc_roc = AUC(num_thresholds=num_thresholds,
              summation_method='interpolation',
              curve='ROC',
              name='auc_roc')

precision = Precision(name='precision', thresholds=clf_threshold)
recall = Recall(name='recall', thresholds=clf_threshold)

binary_accuracy = BinaryAccuracy(name='binary_accuracy', threshold=clf_threshold)

_ = auc_pr.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

_ = auc_roc.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

_ = precision.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['precision'].append(precision.result().numpy())
_ = recall.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['recall'].append(recall.result().numpy())

_ = binary_accuracy.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl_aggtois['label'],
                                                                ranking_tbl_aggtois['predicted class']))

data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl_aggtois['label'], ranking_tbl_aggtois['score']))

for cat, cat_lbl in cats.items():
    data_to_tbl[f'recall {cat}'].append(
        ((ranking_tbl_aggtois['original_label'] == cat) &
         (ranking_tbl_aggtois['predicted class'] == cat_lbl)).sum() /
        (ranking_tbl_aggtois['original_label'] == cat).sum())
for class_id in class_ids:
    data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl_aggtois['label'] == class_id) &
                                                       (ranking_tbl_aggtois['predicted class'] == class_id)).sum() +
                                                      ((ranking_tbl_aggtois['original_label'] != class_id) &
                                                       (ranking_tbl_aggtois['predicted class'] != class_id)).sum()) /
                                                     len(ranking_tbl_aggtois))

for k_val in top_k_vals:
    precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
    if data_to_tbl['n_examples'][-1] < k_val:
        data_to_tbl[f'precision at {k_val}'].append(np.nan)
    else:
        _ = precision_at_k.update_state(ranking_tbl_aggtois['label'].to_list(), ranking_tbl_aggtois['score'].to_list())
        data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())

metrics_df = pd.DataFrame(data_to_tbl)

metrics_df.to_csv(exp_dir / 'metrics_tces_score_aggtois.csv', index=False)

#%% Compute metrics when aggregating scores for TCEs associated with the same TOI and then only considering the TOI

top_k_vals = [50, 100, 250, 500]
cats = {'T-CP': 1, 'T-KP': 1, 'T-FP': 0, 'T-EB': 0, 'T-FA': 0, 'T-NTP': 0}

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_smet_ourmatch.csv')

# experiment directory
exp_dir = Path('/Users/msaragoc/Downloads/cv_merged_base_10-5-2022_1028/')

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')

ranking_tbl = ranking_tbl.merge(tce_tbl[['uid', 'matched_toi_our']], on='uid', how='left', validate='one_to_one')

non_toi_ranking_tbl = ranking_tbl.loc[ranking_tbl['matched_toi_our'].isna()]
agg_cols = [
    'matched_toi_our',
    'tce_period',
    'tce_duration',
    'transit_depth',
    'score',
    'predicted class',
]
toi_ranking_tbl = ranking_tbl.loc[~ranking_tbl['matched_toi_our'].isna(), agg_cols].groupby('matched_toi_our').mean().reset_index().rename(columns={'matched_toi_our': 'uid'})
toi_ranking_tbl['matched_toi_our'] = toi_ranking_tbl['uid']
toi_ranking_tbl['predicted class'] = 0
toi_ranking_tbl.loc[toi_ranking_tbl['score'] > clf_threshold, 'predicted class'] = 1

toi_ranking_tbl = toi_ranking_tbl.merge(ranking_tbl[['matched_toi_our', 'target_id', 'label', 'original_label']].drop_duplicates(subset='matched_toi_our'), on='matched_toi_our', how='left', validate='one_to_many')
ranking_tbl_aggtois = pd.concat([toi_ranking_tbl, non_toi_ranking_tbl[toi_ranking_tbl.columns]], axis=0)

ranking_tbl_aggtois.to_csv(exp_dir / 'ensemble_ranked_predictions_allfolds_aggtois.csv', index=False)

# compute metrics for each CV fold
metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
metrics_lst += [f'recall {cat}' for cat in cats]
metrics_lst += ['n_examples', 'n_pos_examples', 'n_neg_examples', 'frac_pos_examples']
metrics_lst += [f'n_{cat}' for cat in cats]

data_to_tbl = {col: [] for col in metrics_lst}

data_to_tbl['n_examples'].append(len(ranking_tbl_aggtois))

data_to_tbl['n_pos_examples'].append(
    (ranking_tbl_aggtois['original_label'].isin([k for k in cats if cats[k] == 1])).sum())
data_to_tbl['frac_pos_examples'].append(data_to_tbl['n_pos_examples'][-1] / data_to_tbl['n_examples'][-1])
data_to_tbl['n_neg_examples'].append(
    (ranking_tbl_aggtois['original_label'].isin([k for k in cats if cats[k] == 0])).sum())

for cat in cats:
    data_to_tbl[f'n_{cat}'].append((ranking_tbl_aggtois['original_label'] == cat).sum())

# compute metrics
auc_pr = AUC(num_thresholds=num_thresholds,
             summation_method='interpolation',
             curve='PR',
             name='auc_pr')
auc_roc = AUC(num_thresholds=num_thresholds,
              summation_method='interpolation',
              curve='ROC',
              name='auc_roc')

precision = Precision(name='precision', thresholds=clf_threshold)
recall = Recall(name='recall', thresholds=clf_threshold)

binary_accuracy = BinaryAccuracy(name='binary_accuracy', threshold=clf_threshold)

_ = auc_pr.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

_ = auc_roc.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

_ = precision.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['precision'].append(precision.result().numpy())
_ = recall.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['recall'].append(recall.result().numpy())

_ = binary_accuracy.update_state(ranking_tbl_aggtois['label'].tolist(), ranking_tbl_aggtois['score'].tolist())
data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl_aggtois['label'],
                                                                ranking_tbl_aggtois['predicted class']))

data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl_aggtois['label'], ranking_tbl_aggtois['score']))

for cat, cat_lbl in cats.items():
    data_to_tbl[f'recall {cat}'].append(
        ((ranking_tbl_aggtois['original_label'] == cat) &
         (ranking_tbl_aggtois['predicted class'] == cat_lbl)).sum() /
        (ranking_tbl_aggtois['original_label'] == cat).sum())
for class_id in class_ids:
    data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl_aggtois['label'] == class_id) &
                                                       (ranking_tbl_aggtois['predicted class'] == class_id)).sum() +
                                                      ((ranking_tbl_aggtois['original_label'] != class_id) &
                                                       (ranking_tbl_aggtois['predicted class'] != class_id)).sum()) /
                                                     len(ranking_tbl_aggtois))

for k_val in top_k_vals:
    precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
    if data_to_tbl['n_examples'][-1] < k_val:
        data_to_tbl[f'precision at {k_val}'].append(np.nan)
    else:
        _ = precision_at_k.update_state(ranking_tbl_aggtois['label'].to_list(), ranking_tbl_aggtois['score'].to_list())
        data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())

metrics_df = pd.DataFrame(data_to_tbl)

metrics_df.to_csv(exp_dir / 'metrics_aggtois.csv', index=False)

#%%

metrics_filenames = ['metrics_tces_score_aggtois.csv', 'metrics_aggtois.csv', 'metrics_tces_score.csv']
metrics_tbls_fps = [fp for fp in exp_dir.iterdir() if fp.name in metrics_filenames]

metrics_lst_concat = [m for m in metrics_lst if 'precision at' not in m]
metrics_tbls_concat = pd.concat([pd.read_csv(fp)[metrics_lst_concat] for fp in metrics_tbls_fps], axis=0)
metrics_tbls_concat.insert(0, 'metrics_filename', metrics_filenames, True)
metrics_tbls_concat.to_csv(exp_dir / 'metrics_different_methods.csv', index=False)

#%%

tbl = pd.read_csv('/Users/msaragoc/Downloads/compare_tcescore_vs_aggtoiscore.csv')

data_to_tbl = {'toi_id': [], 'delta_n_pos': [], 'label': []}
for toi in tbl['matched_toi_our'].unique():

    tcesintoi = tbl.loc[tbl['matched_toi_our'] == toi]

    if len(tcesintoi) == 0:
        continue

    # if tcesintoi['original_label'].values[0] != 'T-FA':
    #     continue

    data_to_tbl['toi_id'].append(toi)
    data_to_tbl['label'].append(tcesintoi['original_label'].values[0])
    # if len(tcesintoi) == 1:
    #     data_to_tbl['delta_n_pos'].append(0)
    #     continue

    n_clf_pos_toi = (tcesintoi['score'] > 0.5).sum()
    n_clf_pos_tce = (tcesintoi['tce_score'] > 0.5).sum()

    data_to_tbl['delta_n_pos'].append(n_clf_pos_toi - n_clf_pos_tce)

data_tbl = pd.DataFrame(data_to_tbl)
data_tbl.to_csv('/Users/msaragoc/Downloads/counts_pos_tcescore_vs_aggtoiscore.csv', index=False)

for cat in data_tbl['label'].unique():
    print(f'{cat}: {data_tbl.loc[data_tbl["label"] == cat, "delta_n_pos"].sum()}')

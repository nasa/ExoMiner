"""
Computing metrics for TESS CV experiments by using the best data for each TIC across all available sector runs.
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
top_k_vals = [50, 100, 250, 500, 750, 1000, 1250, 1400]
cats = {'T-CP': 1, 'T-KP': 1, 'T-FP': 0, 'T-EB': 0, 'T-FA': 0, 'T-NTP': 0}

#%% metrics using only the TCEs associated with the sector run for which the target is the earliest and has more sectors

exp_dir = Path('/Users/msaragoc/Downloads/cv_merged_base_10-5-2022_1028/')

# compute metrics for each CV fold
metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
metrics_lst += [f'recall {cat}' for cat in cats]
metrics_lst += ['n_examples', 'n_pos_examples', 'n_neg_examples', 'frac_pos_examples']
metrics_lst += [f'n_{cat}' for cat in cats]

# compute metrics for the whole dataset by combining the test set folds from all CV iterations
data_to_tbl = {col: [] for col in metrics_lst}

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')
ranking_tbl['sector_run'] = ranking_tbl['uid'].apply(lambda x: '-'.join(x.split('-')[2:]))
ranking_tbl['start_sector'] = ranking_tbl['sector_run'].apply(lambda x: int(x.split('-')[0][1:]))
ranking_tbl['end_sector'] = ranking_tbl['sector_run'].apply(lambda x: int(x.split('-')[-1][1:]) if len(x.split('-')) == 1 else int(x.split('-')[-1]))
ranking_tbl['n_sectors_in_run'] = ranking_tbl['end_sector'] - ranking_tbl['start_sector'] + 1

ranking_tbl_best_data = []
for target in ranking_tbl['target_id'].unique():
    target_sector_runs = ranking_tbl.loc[ranking_tbl['target_id'] == target].sort_values(['n_sectors_in_run', 'end_sector'], ascending=False)
    chosen_sector_run = target_sector_runs['sector_run'].values[0]  # chosen sector run with more sectors and more recent
    ranking_tbl_target = ranking_tbl.loc[(ranking_tbl['target_id'] == target) & (ranking_tbl['sector_run'] == chosen_sector_run)]
    ranking_tbl_best_data.append(ranking_tbl_target)

ranking_tbl_best_data = pd.concat(ranking_tbl_best_data, axis=0)
ranking_tbl_best_data.to_csv(exp_dir / 'ensemble_ranked_predictions_allfolds_bestdata_pertarget.csv', index=False)

data_to_tbl['n_examples'].append(len(ranking_tbl_best_data))

data_to_tbl['n_pos_examples'].append(
    (ranking_tbl_best_data['original_label'].isin([k for k in cats if cats[k] == 1])).sum())
data_to_tbl['frac_pos_examples'].append(data_to_tbl['n_pos_examples'][-1] / data_to_tbl['n_examples'][-1])
data_to_tbl['n_neg_examples'].append(
    (ranking_tbl_best_data['original_label'].isin([k for k in cats if cats[k] == 0])).sum())

for cat in cats:
    data_to_tbl[f'n_{cat}'].append((ranking_tbl_best_data['original_label'] == cat).sum())

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

_ = auc_pr.update_state(ranking_tbl_best_data['label'].tolist(), ranking_tbl_best_data['score'].tolist())
data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

_ = auc_roc.update_state(ranking_tbl_best_data['label'].tolist(), ranking_tbl_best_data['score'].tolist())
data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

_ = precision.update_state(ranking_tbl_best_data['label'].tolist(), ranking_tbl_best_data['score'].tolist())
data_to_tbl['precision'].append(precision.result().numpy())
_ = recall.update_state(ranking_tbl_best_data['label'].tolist(), ranking_tbl_best_data['score'].tolist())
data_to_tbl['recall'].append(recall.result().numpy())

_ = binary_accuracy.update_state(ranking_tbl_best_data['label'].tolist(), ranking_tbl_best_data['score'].tolist())
data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl_best_data['label'], ranking_tbl_best_data['predicted class']))

data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl_best_data['label'], ranking_tbl_best_data['score']))

for cat, cat_lbl in cats.items():
    data_to_tbl[f'recall {cat}'].append(
        ((ranking_tbl_best_data['original_label'] == cat) & (ranking_tbl_best_data['predicted class'] == cat_lbl)).sum() / (
                    ranking_tbl_best_data['original_label'] == cat).sum())
for class_id in class_ids:
    data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl_best_data['label'] == class_id) & (
            ranking_tbl_best_data['predicted class'] == class_id)).sum() +
                                                ((ranking_tbl_best_data['original_label'] != class_id) & (
                                                        ranking_tbl_best_data['predicted class'] != class_id)).sum()) \
                                               / len(ranking_tbl_best_data))

for k_val in top_k_vals:
    precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
    if data_to_tbl['n_examples'][-1] < k_val:
        data_to_tbl[f'precision at {k_val}'].append(np.nan)
    else:
        _ = precision_at_k.update_state(ranking_tbl_best_data['label'].to_list(), ranking_tbl_best_data['score'].to_list())
        data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())


metrics_df = pd.DataFrame(data_to_tbl)

metrics_df.to_csv(exp_dir / 'metrics_best_data_per_target.csv', index=False)

""" Computing metrics for TESS experiments by sector run. """

# 3rd party
import pandas as pd
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy  # , TopKCategoricalAccuracy
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import numpy as np

#%% Set experiment directory and auxiliary parameters

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/interns/charles_yates/cv_merged_fluxvar_2-4-2023_1-19-2023/')

num_thresholds = 1000  # number of thresholds used to compute AUC
clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall
# map categories to label ids
# cats = {'PC': 1, 'AFP': 0, 'NTP': 0}
# cats = {'PC': 1, 'AFP': 1, 'UNK': 0}
# cats = {'PC': 1, 'AFP': 1, 'NTP': 1, 'UNK': 0}
cats = {'T-CP': 1, 'T-KP': 1, 'T-FP': 0, 'T-EB': 0, 'T-FA': 0, 'T-NTP': 0}
class_ids = [0, 1]  # binary classification ids
# k values for precision-at-k
top_k_vals = [40, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]

#%% metrics per sector run

# compute metrics for each CV fold
metrics_lst = ['sector_run', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
metrics_lst += [f'recall {cat}' for cat in cats]
metrics_lst += ['n_examples', 'n_pos_examples', 'n_neg_examples', 'frac_pos_examples']
metrics_lst += [f'n_{cat}' for cat in cats]

# compute metrics for the whole dataset by combining the test set folds from all CV iterations
data_to_tbl = {col: [] for col in metrics_lst}

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')
ranking_tbl['sector_run'] = ranking_tbl['uid'].apply(lambda x: '-'.join(x.split('-')[2:]))

for sector_run in ranking_tbl['sector_run'].unique():

    ranking_tbl_sector = ranking_tbl.loc[ranking_tbl['sector_run'] == sector_run]

    data_to_tbl['n_examples'].append(len(ranking_tbl_sector))

    data_to_tbl['n_pos_examples'].append((ranking_tbl_sector['original_label'].isin([k for k in cats
                                                                                     if cats[k] == 1])).sum())
    data_to_tbl['frac_pos_examples'].append(data_to_tbl['n_pos_examples'][-1] / data_to_tbl['n_examples'][-1])
    data_to_tbl['n_neg_examples'].append((ranking_tbl_sector['original_label'].isin([k for k in cats
                                                                                     if cats[k] == 0])).sum())

    for cat in cats:
        data_to_tbl[f'n_{cat}'].append((ranking_tbl_sector['original_label'] == cat).sum())

    data_to_tbl['sector_run'].append(sector_run)

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

    _ = auc_pr.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
    data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

    _ = auc_roc.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
    data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

    _ = precision.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
    data_to_tbl['precision'].append(precision.result().numpy())
    _ = recall.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
    data_to_tbl['recall'].append(recall.result().numpy())

    _ = binary_accuracy.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
    data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

    data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl_sector['label'],
                                                                    ranking_tbl_sector['predicted class']))

    data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl_sector['label'],
                                                                ranking_tbl_sector['score']))

    for cat, cat_lbl in cats.items():
        data_to_tbl[f'recall {cat}'].append(
            ((ranking_tbl_sector['original_label'] == cat) &
             (ranking_tbl_sector['predicted class'] == cat_lbl)).sum() /
            (ranking_tbl_sector['original_label'] == cat).sum())
    for class_id in class_ids:
        data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl_sector['label'] == class_id) & (
                ranking_tbl_sector['predicted class'] == class_id)).sum() +
                                                    ((ranking_tbl_sector['original_label'] != class_id) & (
                                                            ranking_tbl_sector['predicted class'] != class_id)).sum()) \
                                                   / len(ranking_tbl_sector))

    for k_val in top_k_vals:
        precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
        if data_to_tbl['n_examples'][-1] < k_val:
            data_to_tbl[f'precision at {k_val}'].append(np.nan)
        else:
            _ = precision_at_k.update_state(ranking_tbl_sector['label'].to_list(),
                                            ranking_tbl_sector['score'].to_list())
            data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())


metrics_df = pd.DataFrame(data_to_tbl)

metrics_df.to_csv(exp_dir / 'metrics_per_sector_run.csv', index=False)

#%% Plot precision-recall curve

import matplotlib.pyplot as plt

ranking_tbl_sector = ranking_tbl.loc[ranking_tbl['sector_run'] == 'S33']
precision_thr = Precision(name='precision_thr', thresholds=list(np.linspace(0, 1, num_thresholds)))
recall_thr = Recall(name='recall_thr', thresholds=list(np.linspace(0, 1, num_thresholds)))
_ = precision_thr.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())
_ = recall_thr.update_state(ranking_tbl_sector['label'].tolist(), ranking_tbl_sector['score'].tolist())

prec, rec = precision_thr.result().numpy(), recall_thr.result().numpy()
idxs_valid = np.where(prec+rec != 0)
prec, rec = prec[idxs_valid], rec[idxs_valid]
f, ax = plt.subplots(figsize=(10, 8))
ax.plot(rec, prec)
ax.scatter(rec, prec, c='r', s=8)
ax.set_ylim([0, 1.01])
ax.set_xlim([0, 1.01])
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.grid(True)
ax.set_title(f'Sector 33 {len(ranking_tbl_sector)} TCEs\n{"".join([f" {el[0]}:{el[1]} |" for el in ranking_tbl_sector["original_label"].value_counts().items()])}')
f.savefig('/Users/msaragoc/Downloads/pr_curve_S33_kepler_tess.png')

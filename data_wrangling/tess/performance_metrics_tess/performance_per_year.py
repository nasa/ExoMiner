""" Computing metrics for TESS experiments by year (aka cycle). """

# 3rd party
import pandas as pd
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy  # , TopKCategoricalAccuracy
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import numpy as np

# %% Set experiment directory and auxiliary parameters

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
metrics_lst = ['year', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
metrics_lst += [f'recall {cat}' for cat in cats]
metrics_lst += ['n_examples', 'n_pos_examples', 'n_neg_examples', 'frac_pos_examples']
metrics_lst += [f'n_{cat}' for cat in cats]

# compute metrics for the whole dataset by combining the test set folds from all CV iterations
data_to_tbl = {col: [] for col in metrics_lst}

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')


def _sector_run_to_year(x):
    """ Map sector run in TCE unique id to TESS Year/Cycle.

    Args:
        x: str, TCE uid

    Returns: str, TESS Year/Cycle the TCE.

    """

    sector_run = '-'.join(x.split('-')[2:])[1:]

    if '-' in sector_run:
        sector_year = 'multi-sector'
    else:
        sector_run = int(sector_run)
        if sector_run in range(1, 14):
            sector_year = 'Y1'
        elif sector_run in range(14, 27):
            sector_year = 'Y2'
        elif sector_run in range(27, 40):
            sector_year = 'Y3'
        elif sector_run in range(40, 56):
            sector_year = 'Y4'
        elif sector_run in range(56, 70):
            sector_year = 'Y5'
        elif sector_run in range(70, 84):
            sector_year = 'Y6'

    return sector_year


ranking_tbl['sector_run'] = ranking_tbl['uid'].apply(lambda x: '-'.join(x.split('-')[2:]))
ranking_tbl['year'] = ranking_tbl['uid'].apply(_sector_run_to_year)

for year in ranking_tbl['year'].unique():

    ranking_tbl_year = ranking_tbl.loc[ranking_tbl['year'] == year]

    data_to_tbl['n_examples'].append(len(ranking_tbl_year))

    data_to_tbl['n_pos_examples'].append((ranking_tbl_year['original_label'].isin([k for k in cats
                                                                                   if cats[k] == 1])).sum())
    data_to_tbl['frac_pos_examples'].append(data_to_tbl['n_pos_examples'][-1] / data_to_tbl['n_examples'][-1])
    data_to_tbl['n_neg_examples'].append((ranking_tbl_year['original_label'].isin([k for k in cats
                                                                                   if cats[k] == 0])).sum())

    for cat in cats:
        data_to_tbl[f'n_{cat}'].append((ranking_tbl_year['original_label'] == cat).sum())

    data_to_tbl['year'].append(year)

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

    _ = auc_pr.update_state(ranking_tbl_year['label'].tolist(), ranking_tbl_year['score'].tolist())
    data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

    _ = auc_roc.update_state(ranking_tbl_year['label'].tolist(), ranking_tbl_year['score'].tolist())
    data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

    _ = precision.update_state(ranking_tbl_year['label'].tolist(), ranking_tbl_year['score'].tolist())
    data_to_tbl['precision'].append(precision.result().numpy())
    _ = recall.update_state(ranking_tbl_year['label'].tolist(), ranking_tbl_year['score'].tolist())
    data_to_tbl['recall'].append(recall.result().numpy())

    _ = binary_accuracy.update_state(ranking_tbl_year['label'].tolist(), ranking_tbl_year['score'].tolist())
    data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

    data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl_year['label'],
                                                                    ranking_tbl_year['predicted class']))

    data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl_year['label'],
                                                                ranking_tbl_year['score']))

    for cat, cat_lbl in cats.items():
        data_to_tbl[f'recall {cat}'].append(
            ((ranking_tbl_year['original_label'] == cat) & (ranking_tbl_year['predicted class'] == cat_lbl)).sum() / (
                        ranking_tbl_year['original_label'] == cat).sum())
    for class_id in class_ids:
        data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl_year['label'] == class_id) & (
                ranking_tbl_year['predicted class'] == class_id)).sum() +
                                                    ((ranking_tbl_year['original_label'] != class_id) & (
                                                            ranking_tbl_year['predicted class'] != class_id)).sum()) \
                                                   / len(ranking_tbl_year))

    for k_val in top_k_vals:
        precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
        if data_to_tbl['n_examples'][-1] < k_val:
            data_to_tbl[f'precision at {k_val}'].append(np.nan)
        else:
            _ = precision_at_k.update_state(ranking_tbl_year['label'].to_list(), ranking_tbl_year['score'].to_list())
            data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())


metrics_df = pd.DataFrame(data_to_tbl)

metrics_df.to_csv(exp_dir / 'metrics_per_year.csv', index=False)

""" Generate a confusion matrix based on a given set of dispositions for TESS data. """

# 3rd party
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, Precision, Recall

# %%

# filepath to ranking table produced by a model
exp_dir = Path(
    '/data5/tess_project/experiments/current_experiments/tess_experiments/results_ensemble/tess_s1-40-dv_g301-l31_5tr_spline_nongapped_configK_flux_per_10-5-2021')
ranking_tbl_fp = exp_dir / 'ensemble_ranked_predictions_predictset.csv'
ranking_tbl = pd.read_csv(ranking_tbl_fp)

# column used to get the dispositions
disp_col = 'original_label'

# how to deal with NaN or 'None' dispositions
ranking_tbl.loc[(ranking_tbl[disp_col].isna()) | (ranking_tbl[disp_col] == 'None'), disp_col] = 'NA'

ranking_tbl.loc[ranking_tbl[disp_col] == 'NA', 'label'] = 0


# %% Plot confusion matrix

def _map_tess_disp_to_label_id(label, label_map):
    """ Map TESS disposition to label id encoded in the label_map dictionary.

    :param label: ground truth label. Must be a key in label_map
    :param label_map: dict, maps from label to label id
    :return:
        int, ground truth label mapped to label id
    """
    return label_map[label]


# dictionary that maps ground truth label to label id (many to 3: PC (1), non-PC (0), missing label (-1))
label_map = {
    'PC': 1,
    'KP': 1,
    'FP': 0,
    'FA': 0,
    'EB': 0,
    'CP': 1,
    'APC': 0,
    'O': 0,
    'N/A': -1,  # class for missing disposition
}

# only when using TESS disposition as ground truth;
# map dispositions to label ids (1 for PC, 0 for non-PC, -1 for missing label)
# ranking_tbl['label'] = ranking_tbl['TESS Disposition'].apply(_map_tess_disp_to_label_id, axis=1, label_map=label_map)


def _map_pred_class_to_label(row, disp_col):
    """ Map predicted label id to label.

    :param row: Pandas time series, contains 'label' field with ground truth label id, 'predicted class' with predicted
    label id, and `disp_col` with ground truth label
    :param disp_col: str, name of field with ground truth label
    :return:
        int, predicted label id mapped to label
    """

    if row['label'] == row['predicted class']:
        return row[disp_col]
    else:
        return 'Misclf'

# map predicted label id to predicted label
ranking_tbl['predicted_label'] = \
    ranking_tbl[['label', disp_col, 'predicted class']].apply(_map_pred_class_to_label, axis=1, args=(disp_col,))

# compute confusion matrix for each disposition
cmat = confusion_matrix(ranking_tbl[disp_col].values, ranking_tbl['predicted_label'].values)

# get list of classes
disp_classes = sorted(np.append(ranking_tbl[disp_col].unique(), 'Misclf'))

# plot confusion matrix and save it
cmat_disp = ConfusionMatrixDisplay(cmat, display_labels=disp_classes)
cmat_fig = cmat_disp.plot()
cmat_fig.figure_.savefig(exp_dir / f'cmat_{disp_col}.png')

# %% Plot score distribution for different categories

bins = np.linspace(0, 1, 11, True)
norm = False
log_y = True

allowed_categories = ['PC', 'APC']  # ['KP', 'CP', 'FP', 'FA']
ranking_tbl_distr = ranking_tbl.loc[ranking_tbl['original_label'].isin(allowed_categories)]
categories = ranking_tbl_distr['original_label'].unique()
n_categories = len(categories)

hist, bin_edges = {}, {}
for class_label in categories:
    counts_cl = list(np.histogram(ranking_tbl_distr.loc[ranking_tbl_distr['original_label'] == class_label, 'score'],
                                  bins,
                                  density=False,
                                  range=(0, 1)))
    if norm:
        counts_cl[0] = counts_cl[0] / len(ranking_tbl_distr[class_label])
    hist[class_label] = counts_cl[0]
    bin_edges[class_label] = counts_cl[1]

bins_multicl = np.linspace(0, 1, n_categories * 10 + 1, True)
bin_width = bins_multicl[1] - bins_multicl[0]
bins_cl = {}
for i, class_label in enumerate(categories):
    bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                            for idx in range(i, len(bins_multicl) - 1, n_categories)]

f, ax = plt.subplots()
for class_label in categories:
    ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
if norm:
    ax.set_ylabel('Class fraction')
else:
    ax.set_ylabel('Number of examples')
ax.set_xlabel('Predicted output')
ax.set_xlim([0, 1])
if norm:
    ax.set_ylim([0, 1])
if log_y:
    ax.set_yscale('log')
ax.set_xticks(np.linspace(0, 1, 11, True))
ax.legend()
ax.set_title('Output distribution')
plt.savefig(exp_dir / f'hist_score_{"-".join(allowed_categories)}.png')
# plt.close()

# %% Compute performance metrics

allowed_categories = ['KP', 'CP', 'FP', 'FA']
ranking_tbl_metric = ranking_tbl.loc[ranking_tbl['original_label'].isin(allowed_categories)]

num_thresholds = 1000
thrs = [0.5]
threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

# compute metrics
performance_metrics = {
    'auc_pr': AUC(num_thresholds=num_thresholds,
                  summation_method='interpolation',
                  curve='PR',
                  name='auc_pr'),
    'auc_roc': AUC(num_thresholds=num_thresholds,
                   summation_method='interpolation',
                   curve='ROC',
                   name='auc_roc'),

    'precision': Precision(
        thresholds=thrs,
        name='precision'),

    'recall': Recall(
        thresholds=thrs,
        name='recall')
}

performance_metrics_res = {metric: np.nan for metric in performance_metrics}
for metric in performance_metrics:
    performance_metrics[metric].update_state(ranking_tbl_metric['label'].tolist(), ranking_tbl_metric['score'].tolist())
    performance_metrics_res[metric] = [performance_metrics[metric].result().numpy()]

metrics_tbl = pd.DataFrame(performance_metrics_res)
metrics_tbl.to_csv(exp_dir / 'metrics.csv', index=False)

# %% Join results from different experiments

exp_root_dir = Path('/data5/tess_project/experiments/current_experiments/tess_experiments/results_ensemble/')
exp_tbls = {exp_dir.name: pd.read_csv(exp_dir / 'metrics.csv')
            for exp_dir in exp_root_dir.iterdir() if 'tess_s1-40' in exp_dir.name}

for exp in exp_tbls.keys():
    exp_tbls[exp].insert(0, 'experiment', exp)

metrics_tbl = pd.concat(list(exp_tbls.values()))
metrics_tbl.to_csv(exp_root_dir / 'metrics_tess_experiments.csv', index=False)

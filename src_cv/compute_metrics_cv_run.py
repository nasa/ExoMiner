"""
Computing metrics for each CV fold and for the whole dataset (aggregates all separated test CV folds.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from src.compute_metrics_from_predictions_csv_file import compute_metrics_from_predictions


num_thresholds = 1000  # number of thresholds used to compute AUC
clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall
multiclass = False  # multiclass or bin class?
target_score = 'score_AFP'  # get auc_pr metrics for different class labels
class_name = 'label_id'
cat_name = 'label'

# mapping of category/disposition to label id
cats = {
    # data set (since each data set might contain different populations of examples)
    'train': {
        'PC': 1,
        'AFP': 0,
        'NTP': 0,
        # 'UNK': 0,
        # 'T-KP': 1,
        # 'T-CP': 1,
        # 'T-EB': 2,
        # 'T-FP': 2,
        # 'T-FA': 0,
        # 'T-NTP': 0,
    },
    'val': {
        'PC': 1,
        'AFP': 0,
        'NTP': 0,
        # 'UNK': 0,
        # 'T-KP': 1,
        # 'T-CP': 1,
        # 'T-EB': 2,
        # 'T-FP': 2,
        # 'T-FA': 0,
        # 'T-NTP': 0,
    },
    'test': {
        'PC': 1,
        'AFP': 0,
        'NTP': 0,
        # 'UNK': 0,
        # 'T-KP': 1,
        # 'T-CP': 1,
        # 'T-EB': 2,
        # 'T-FP': 2,
        # 'T-FA': 0,
        # 'T-NTP': 0,
    },
}
# cats = None
class_ids = [0, 1]  # should match unique label ids in 'cats'
top_k_vals = [50, 100, 200]
# top_k_vals = {
#     'train': [50, 100, 250, 500, 1000, 2000],  # , 2500]
#     'val': [25, 50, 100, 200],
#     'test': [25, 50, 100, 200],
#               }
datasets = [
    'train',
    'val',
    'test'
]

# compute metrics for the whole dataset by combining the test set folds from all CV iterations
# ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!
compute_metrics_all_dataset = False
# if True, computes mean and std metrics' values across CV iterations
compute_mean_std_metrics = False

# define list of metrics to be computed
metrics_lst = ['fold', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced_accuracy',
               'avg_precision']
metrics_lst += [f'precision_at_{k_val}' for k_val in top_k_vals]
metrics_lst += [f'recall_class_{class_id}' for class_id in class_ids]
metrics_lst += [f'n_{class_id}' for class_id in class_ids]

# cv experiment directories
cv_run_dirs = [
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_train_kepler_simulated_data_10-13-2023_1530'),
]
for cv_run_dir in cv_run_dirs:  # iterate through multiple CV runs

    print(f'Getting metrics for experiment {cv_run_dir}...')

    for dataset in datasets:

        # set metrics for data set in cv run
        metrics_lst_dataset = list(metrics_lst)
        if cats is not None:
            metrics_lst += [f'recall_{cat}' for cat in cats[dataset]]
            metrics_lst += [f'n_{cat}' for cat in cats[dataset]]

        print(f'Getting metrics for experiment {cv_run_dir} for data set {dataset}...')

        # get directories of cv iterations in cv run
        cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

        cv_iters_tbls = []
        for cv_iter_dir in cv_iters_dirs:  # iterate through each cv iteration

            # compute metrics
            metrics_df = compute_metrics_from_predictions(ranking_tbl, cats[dataset], num_thresholds, clf_threshold,
                                                          top_k_vals, class_name, cat_name)

        # mean and std across all CV folds
        if compute_mean_std_metrics:
            mean_df = metrics_df[[col for col in metrics_df.columns if col != 'fold']].mean(axis=0).to_frame().T
            mean_df['fold'] = 'mean'
            std_df = metrics_df[[col for col in metrics_df.columns if col != 'fold']].std(axis=0).to_frame().T
            std_df['fold'] = 'std'
            metrics_df = pd.concat([metrics_df, mean_df, std_df])

        if dataset == 'test' and compute_metrics_all_dataset:
            # compute metrics for the whole dataset by combining the test set folds from all CV iterations
            # ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!
            data_to_tbl = {col: [] for col in metrics_lst}

            ranking_tbl = pd.read_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv')

            # compute metrics
            metrics_df = compute_metrics_from_predictions(ranking_tbl, cats[dataset], num_thresholds, clf_threshold,
                                                          top_k_vals, class_name, cat_name)

        metrics_df.to_csv(cv_run_dir / f'metrics_{dataset}.csv', index=False)

"""
Computing metrics for each CV fold and for the whole dataset.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy  # , TopKCategoricalAccuracy
from sklearn.metrics import balanced_accuracy_score, average_precision_score

# %%

num_thresholds = 1000  # number of thresholds used to compute AUC
clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall

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
        # 'T-EB': 0,
        # 'T-FP': 0,
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
        # 'T-EB': 0,
        # 'T-FP': 0,
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
        # 'T-EB': 0,
        # 'T-FP': 0,
        # 'T-FA': 0,
        # 'T-NTP': 0,
    },
}
class_ids = [0, 1]
top_k_vals = [50, 100, 250, 500, 750, 1000, 1500, 2000]  # , 2500]
datasets = [
    'train',
    'val',
    'test'
]

# cv experiment directory
# cv_run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler-tess_weightedcats_tessonlytrainingset_1-25-2023_1318')
# cv_run_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments')
cv_run_dirs = [Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_frozenbeforefcconv_3-22-2023_1440')]  # [fp for fp in cv_run_root_dir.iterdir() if fp.is_dir()]
for cv_run_dir in cv_run_dirs:
    print(f'Getting metrics for experiment {cv_run_dir}...')
    for dataset in datasets:
        print(f'Getting metrics for experiment {cv_run_dir} for data set {dataset}...')
        # compute metrics for each CV fold
        metrics_lst = ['fold', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced accuracy', 'avg precision']
        metrics_lst += [f'precision at {k_val}' for k_val in top_k_vals]
        metrics_lst += [f'accuracy class {class_id}' for class_id in class_ids]
        metrics_lst += [f'recall {cat}' for cat in cats[dataset]]
        metrics_lst += [f'n_{cat}' for cat in cats[dataset]]
        data_to_tbl = {col: [] for col in metrics_lst}

        cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

        cv_iters_tbls = []
        for cv_iter_dir in cv_iters_dirs:

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

            ranking_tbl = pd.read_csv(cv_iter_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
            # # exclude Kepler examples from set
            # ranking_tbl = ranking_tbl.loc[~ranking_tbl['original_label'].isin(['PC', 'AFP', 'NTP'])]

            data_to_tbl['fold'].extend([cv_iter_dir.name.split('_')[-1]])

            _ = auc_pr.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

            _ = auc_roc.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

            _ = precision.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['precision'].append(precision.result().numpy())
            _ = recall.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['recall'].append(recall.result().numpy())

            _ = binary_accuracy.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

            data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl['label'],
                                                                            ranking_tbl['predicted class']))

            data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl['label'], ranking_tbl['score']))

            for cat, cat_lbl in cats[dataset].items():
                data_to_tbl[f'recall {cat}'].append(((ranking_tbl['original_label'] == cat) &
                                                     (ranking_tbl['predicted class'] == cat_lbl)).sum() /
                                                    (ranking_tbl['original_label'] == cat).sum())
                data_to_tbl[f'n_{cat}'].append((ranking_tbl['original_label'] == cat).sum())

            for class_id in class_ids:
                data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl['label'] == class_id) & (
                            ranking_tbl['predicted class'] == class_id)).sum() +
                                            ((ranking_tbl['original_label'] != class_id) & (
                                                        ranking_tbl['predicted class'] != class_id)).sum()) \
                                           / len(ranking_tbl))

            for k_val in top_k_vals:
                precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
                _ = precision_at_k.update_state(ranking_tbl['label'].to_list(), ranking_tbl['score'].to_list())
                data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())

        metrics_df = pd.DataFrame(data_to_tbl)

        # mean and std across all CV folds
        mean_df = metrics_df.mean(axis=0).to_frame().T
        mean_df['fold'] = 'mean'
        std_df = metrics_df.std(axis=0).to_frame().T
        std_df['fold'] = 'std'
        metrics_df = pd.concat([metrics_df, mean_df, std_df])

        if dataset == 'test':
            # compute metrics for the whole dataset by combining the test set folds from all CV iterations
            # ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!
            data_to_tbl = {col: [] for col in metrics_lst}

            ranking_tbl = pd.read_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv')

            data_to_tbl['fold'].extend(['all'])

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

            _ = auc_pr.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

            _ = auc_roc.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

            _ = precision.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['precision'].append(precision.result().numpy())
            _ = recall.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['recall'].append(recall.result().numpy())

            _ = binary_accuracy.update_state(ranking_tbl['label'].tolist(), ranking_tbl['score'].tolist())
            data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

            data_to_tbl['balanced accuracy'].append(balanced_accuracy_score(ranking_tbl['label'],
                                                                            ranking_tbl['predicted class']))

            data_to_tbl['avg precision'].append(average_precision_score(ranking_tbl['label'], ranking_tbl['score']))

            for cat, cat_lbl in cats[dataset].items():
                data_to_tbl[f'recall {cat}'].append(
                    ((ranking_tbl['original_label'] == cat) & (ranking_tbl['predicted class'] == cat_lbl)).sum() / (
                                ranking_tbl['original_label'] == cat).sum())
                data_to_tbl[f'n_{cat}'].append((ranking_tbl['original_label'] == cat).sum())

            for class_id in class_ids:
                data_to_tbl[f'accuracy class {class_id}'].append((((ranking_tbl['label'] == class_id) & (
                        ranking_tbl['predicted class'] == class_id)).sum() +
                                                            ((ranking_tbl['original_label'] != class_id) & (
                                                                    ranking_tbl['predicted class'] != class_id)).sum()) \
                                                           / len(ranking_tbl))

            for k_val in top_k_vals:
                precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
                _ = precision_at_k.update_state(ranking_tbl['label'].to_list(), ranking_tbl['score'].to_list())
                data_to_tbl[f'precision at {k_val}'].append(precision_at_k.result().numpy())

            metrics_df = pd.concat([metrics_df, pd.DataFrame(data_to_tbl)])

        metrics_df.to_csv(cv_run_dir / f'metrics_{dataset}.csv', index=False)

# #%% Per class accuracy and per category recall
#
# ranking_tbl = pd.read_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv')
#
# std_acc = accuracy_score(ranking_tbl['label'], ranking_tbl['predicted class'])
# bal_acc = balanced_accuracy_score(ranking_tbl['label'], ranking_tbl['predicted class'])
# # bin_acc = accuracy_score(ranking_tbl['label'], ranking_tbl['predicted class'], average='binary', pos_label=1)
# # avg_acc = accuracy_score(ranking_tbl['label'], ranking_tbl['predicted class'], average='macro')
#
# cats = {'PC': 1, 'AFP': 1, 'NTP': 1, 'UNK': 0}
# cat_recall = {cat: np.nan for cat in cats}
# class_ids = [0, 1]
# class_accuracy = {class_id: np.nan for class_id in class_ids}
# for cat, cat_lbl in cats.items():
#     cat_recall[cat] = ((ranking_tbl['original_label'] == cat) & (ranking_tbl['predicted class'] == cat_lbl)).sum() / \
#                       (ranking_tbl['original_label'] == cat).sum()
# for class_id in class_ids:
#     class_accuracy[class_id] = (((ranking_tbl['label'] == class_id) & (ranking_tbl['predicted class'] == class_id)).sum() +
#                          ((ranking_tbl['original_label'] != class_id) & (ranking_tbl['predicted class'] != class_id)).sum()) \
#                         / len(ranking_tbl)
#
# with open(cv_run_dir / 'additional_metrics.txt', 'w') as add_metrics_file:
#     add_metrics_file.write(f'Category to class id mapping: {cats}\n')
#     add_metrics_file.write(f'\nBalanced accuracy: {bal_acc:.4f}\n')
#     add_metrics_file.write(f'\nPer-category recall ----\n')
#     for cat, cat_rec in cat_recall.items():
#         add_metrics_file.write(f'Category {cat}: {cat_rec:.4}\n')
#
#     add_metrics_file.write(f'\nPer-class accuracy ----\n')
#     for class_id, class_acc in class_accuracy.items():
#         add_metrics_file.write(f'Class {class_id}: {class_acc:.4}\n')

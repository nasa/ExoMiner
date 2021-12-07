"""
Computing metrics for each CV fold and for the whole dataset.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy

# %%

# cv experiment directory
cv_run_dir = Path(
    '/data5/tess_project/experiments/current_experiments/cv_experiments/cv_keplerq1q17dr25_exominer_configk_addnewval_no_oddeven_features_12-3-2021')

num_thresholds = 1000  # number of thresholds used to compute AUC
clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall

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

# compute metrics for each CV fold
data_to_tbl = {col: [] for col in ['fold', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy']}

cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_testset.csv')
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

metrics_df = pd.DataFrame(data_to_tbl)

# mean and std across all CV folds
mean_df = metrics_df.mean(axis=0).to_frame().T
mean_df['fold'] = 'mean'
std_df = metrics_df.std(axis=0).to_frame().T
std_df['fold'] = 'std'
metrics_df = pd.concat([metrics_df, mean_df, std_df])

# compute metrics for the whole dataset by combining the test set folds from all CV iterations
data_to_tbl = {col: [] for col in ['fold', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy']}

ranking_tbl = pd.read_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv')

data_to_tbl['fold'].extend(['all'])

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

metrics_df = pd.concat([metrics_df, pd.DataFrame(data_to_tbl)])
metrics_df.to_csv(cv_run_dir / 'metrics.csv', index=False)

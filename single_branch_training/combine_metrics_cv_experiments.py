"""
Aggregating metrics for different CV experiments.
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd

#%%

# cv_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments')
cv_exp_dirs = [
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_ntl_4-13-2023_0951'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_centroid_4-12-2023_1648'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_sec_4-12-2023_1219'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_4-12-2023_0929'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/cv_kepler_single_branch_fpflags_combine_frozenbeforefcconv_4-14-2023_1427'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/cv_kepler_single_branch_fpflags_combine_frozenbranches_4-17-2023_0922'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/cv_kepler_single_branch_fpflags_combine_trainall_4-14-2023_1028')
               ]  # + [fp for fp in cv_root_dir.iterdir() if fp.is_dir()]
save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/')
datasets = [
    # 'train',
    # 'val',
    'test'
]
stats = [
    'all',
    'mean',
    'std',
    'se',
]
n_samples = 10  # number of CV folds

metrics = [
    'auc_pr',
    'auc_roc',
    'precision',
    'recall',
    'accuracy',
    'balanced_accuracy',
    'avg_precision',
    'precision_at_50',
    'precision_at_100',
    'precision_at_250',
    'precision_at_500',
    'precision_at_750',
    'precision_at_1000',
    'precision_at_1500',
    'precision_at_2000',
    # 'accuracy class 0',
    # 'accuracy class 1',
    'recall_class_0',
    'recall_class_1',
    # 'n_0',
    # 'n_1',
    'recall_PC',
    'recall_AFP',
    'recall_NTP',
    'n_PC',
    'n_AFP',
    'n_NTP',
]
for dataset in datasets:

    print(f'Iterating over data set {dataset}...')

    new_df = {'experiment': []}
    for metric in metrics:
        for stat in stats:
            new_df[f'{metric}_{stat}'] = []

    for exp_dir in cv_exp_dirs:
        print(f'Iterating over data set {dataset} for experiment {exp_dir.name}...')

        new_df['experiment'].append(exp_dir.name)

        metrics_tbl = pd.read_csv(exp_dir / f'metrics_{dataset}.csv')
        metrics_tbl.set_index('fold', inplace=True)
        metrics_tbl = metrics_tbl.loc[['mean', 'std', 'all']]

        for metric in metrics:
            for stat in stats:
                if stat == 'se':  # compute standard error of the mean from std estimate and number of CV folds
                    new_df[f'{metric}_{stat}'].append(metrics_tbl.loc['std', metric] / np.sqrt(n_samples))
                else:
                    new_df[f'{metric}_{stat}'].append(metrics_tbl.loc[stat, metric])

    new_df = pd.DataFrame(new_df)
    new_df.to_csv(save_dir / 'metrics_all_experiments_combine.csv', index=False)

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
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_ntl_4-5-2023_1042'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_centroid_4-4-2023_1027'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_centroid_allfeatureskepler_4-10-2023_0937'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_sec_4-4-2023_0049'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_sec_4-8-2023_1246'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_4-3-2023_1612'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_4-6-2023_1019'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_dvtceoedpbinstat_4-10-2023_1543'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_dvtceoedpbinstat_4-10-2023_2330')
               ]  # + [fp for fp in cv_root_dir.iterdir() if fp.is_dir()]
save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments')
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
    # 'precision at 50',
    # 'precision at 100',
    # 'precision at 250',
    # 'precision at 500',
    # 'precision at 750',
    # 'precision at 1000',
    # 'precision at 1500',
    # 'precision at 2000',
    # 'accuracy class 0',
    # 'accuracy class 1',
    # 'recall PC',
    # 'recall AFP',
    # 'recall NTP',
    # 'n_PC',
    # 'n_AFP',
    # 'n_NTP',
    'recall_class_0',
    'recall_class_1',
    'n_0',
    'n_1',
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
    new_df.to_csv(save_dir / 'metrics_all_experiments.csv', index=False)

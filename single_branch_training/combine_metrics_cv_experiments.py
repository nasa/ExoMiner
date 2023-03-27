"""
Aggregating metrics for different CV experiments.
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd

#%%

cv_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments')
cv_exp_dirs = [
                  Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_3-14-2023_2348'),
                  Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_full_exominer_3-9-2023_1147'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_3-21-2023_1255'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_trainalllayers_3-22-2023_0944'),
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_frozenbeforefcconv_3-22-2023_1440'),
               ] + \
                   [fp for fp in cv_root_dir.iterdir() if fp.is_dir()]
save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/')
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
    'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy',
    'balanced accuracy', 'avg precision', 'precision at 50',
    'precision at 100', 'precision at 250', 'precision at 500',
    'precision at 750', 'precision at 1000', 'precision at 1500',
    'precision at 2000', 'accuracy class 0', 'accuracy class 1',
    'recall PC', 'recall AFP', 'recall NTP', 'n_PC', 'n_AFP', 'n_NTP'
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

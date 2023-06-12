"""
Get statistics on number of training epochs over CV iterations for different CV experiments.
"""

# 3rd party
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.stats import mad_std

cv_experiments_fps = [
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_oddeven_4-3-2023_1612'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_sec_4-4-2023_0049'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_ntl_4-5-2023_1042'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_centroid_4-4-2023_1027'),
]

patience = 20

data_to_tbl = {
    'experiment': [],
    'mean': [],
    'std': [],
    'med': [],
    'mad_std': []
}

for cv_experiment_fp in cv_experiments_fps:

    # data_to_tbl['experiment'].append(cv_experiment_fp.name)
    # n_epochs_train_cv_iters = []
    for cv_iter_fp in [fp for fp in cv_experiment_fp.iterdir() if fp.name.startswith('cv_iter_') and fp.is_dir()]:
        data_to_tbl['experiment'].append(cv_iter_fp.name)
        n_epochs_train_cv_iters = []
        for model_fp in [fp for fp in (cv_iter_fp / 'models').iterdir() if fp.name.startswith('model') and fp.is_dir()]:
            res_eval_cv = np.load(model_fp / 'res_eval.npy', allow_pickle=True).item()
            n_epochs_train_cv_iters.append(len(res_eval_cv['loss']) - patience)

        data_to_tbl['mean'].append(np.mean(n_epochs_train_cv_iters))
        data_to_tbl['std'].append(np.std(n_epochs_train_cv_iters, ddof=1))
        data_to_tbl['med'].append(np.median(n_epochs_train_cv_iters))
        data_to_tbl['mad_std'].append(mad_std(n_epochs_train_cv_iters))

    # data_to_tbl['mean'].append(np.mean(n_epochs_train_cv_iters))
    # data_to_tbl['std'].append(np.std(n_epochs_train_cv_iters, ddof=1))
    # data_to_tbl['med'].append(np.median(n_epochs_train_cv_iters))
    # data_to_tbl['mad_std'].append(mad_std(n_epochs_train_cv_iters))

data_df = pd.DataFrame(data_to_tbl)
data_df.to_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/n_epochs_train_stats.csv', index=False)

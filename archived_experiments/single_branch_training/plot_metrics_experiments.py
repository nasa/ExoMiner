"""
Plots for analyzing results of experiments.
- Plot metric curve for each model and CV iteration for different experiments in individual plots.
- Plot metric curves for a model and CV iteration for multiple experiments in the same plot.
"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# local
from src.train.utils_train import plot_loss_metric

#%% Plot loss curves for all models in all CV iterations for a set of experiments

# cv_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments')
cv_exp_dirs = [
    Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_sec_4-8-2023_1246'),
               ]

# cv_exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments/cv_kepler_single_branch_local_centroid_3-11-2023_1026')

patience = 0
metric = 'auc_pr'

for cv_exp_dir in cv_exp_dirs:
    print(f'Iterating through experiment {cv_exp_dir}...')
    cv_iters_dirs = [fp for fp in cv_exp_dir.iterdir() if fp.is_dir()]

    save_dir = Path(cv_exp_dir / f'plot_loss_{metric}_{cv_exp_dir.name}')
    save_dir.mkdir(exist_ok=True)

    for cv_iter_dir in cv_iters_dirs:

        models_root_dir = cv_iter_dir / 'models'
        models_dirs = [fp for fp in models_root_dir.iterdir() if fp.is_dir()]
        for model_dir in models_dirs:
            res_eval = np.load(model_dir / 'res_eval.npy', allow_pickle=True).item()
            epochs = np.arange((len(res_eval['loss'])))
            plot_loss_metric(
                res_eval,
                epochs,
                -patience,
                save_dir / f'plot_loss_{metric}_{cv_exp_dir.name}_{cv_iter_dir.name}_{model_dir.name}.png',
                metric
            )

#%% Plot metric for a model in a CV iteration for different experiments

plot_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/plot_metric_curves_experiments')
plot_dir.mkdir(exist_ok=True)
metric_name = 'loss'
cv_iter = 7
model_id = 0
experiments = {
    'Combine Single Frozen Before FC in Branch 3-22-2023_1440': {'root_dir': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_frozenbeforefcconv_3-22-2023_1440'),
              'cv_iter': cv_iter,
              'model': 0,
                                                                 'line': 'r--',
              },
    'Full ExoMiner 3-9-2023_1147': {'root_dir': Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_full_exominer_3-9-2023_1147'),
              'cv_iter': cv_iter,
              'model': 0,
            'line': 'g-.',
              },
    'Combine Single Train All Layers 3-22-2023_0944': {'root_dir': Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_trainalllayers_3-22-2023_0944'),
              'cv_iter': cv_iter,
              'model': 0,
                'line': 'b',
              },
    'Combine Single Frozen Branches 3-21-2023_1255': {'root_dir': Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_combine_3-21-2023_1255'),
        'cv_iter': cv_iter,
        'model': 0,
'line': 'c.',
    },
#     'Local Flux 3-10-2023_1011': {'root_dir': Path(
#         '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments/cv_kepler_single_branch_local_flux_3-10-2023_1011'),
#         'cv_iter': cv_iter,
#         'model': 0,
# 'line': 'k-',
#     },
}

for cv_iter in range(0, 10):
    f, ax = plt.subplots(figsize=(12, 8))
    for experiment_name, experiment in experiments.items():
        res_eval = np.load(experiment['root_dir'] / f'cv_iter_{cv_iter}' / 'models' /
                           f'model{experiment["model"]}' / 'res_eval.npy',
                           allow_pickle=True).item()
        epochs = np.arange(len(res_eval[metric_name]))
        ax.plot(epochs, res_eval[metric_name], experiment['line'], label=experiment_name)
    ax.set_xlabel('Epoch Number')
    if 'val' not in metric_name:
        ax.set_ylabel(f'Train {metric_name}')
    else:
        ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid()
    ax.set_ylim(bottom=0, top=0.15)
    # ax.set_ylim(bottom=0.85, top=1)
    ax.set_xlim(left=0)
    ax.set_title(f'CV Iteration {cv_iter}\nModel {model_id}')
    f.tight_layout()
    f.savefig(plot_dir / f'plot_{metric_name}_cv_iter{cv_iter}_model{model_id}png')
    plt.close()

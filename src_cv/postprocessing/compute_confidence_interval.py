"""
Compute confidence intervals for performance metrics in a cross-validation experiment that has 'mean' and 'std'
statistics. Uses the t-distribution and standard error of them mean to compute the confidence intervals.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path


def compute_confidence_interval_for_cv_experiment(experiment_metrics, confidence_level, dataset, metric, n_statistics):
    """ Compute confidence interval for a given metric in a given dataset for a cross-validation experiment.

    Args:
        experiment_metrics: pandas DataFrame, experiment metrics table. Contains 'mean' and 'std' statistics for metric
            `metric`.
        confidence_level: float, confidence level for the confidence interval.
        dataset: str, dataset name: 'test', 'train', 'validation'.
        metric: str, metric name
        n_statistics: int, number of statistics (equal to number of CV iterations).

    Returns: confidence_interval, tuple of floats, (left_bound, right_bound) for the confidence interval.

    """

    # Filter the metrics for the specific dataset
    experiment_metrics_dataset = experiment_metrics.loc[experiment_metrics['dataset'] == dataset]

    # get mean and std statistics
    mean_stat = experiment_metrics_dataset.loc[experiment_metrics_dataset['statistic'] == 'mean', metric]
    std_stat = experiment_metrics_dataset.loc[experiment_metrics_dataset['statistic'] == 'std', metric]

    # compute standard error of the mean
    sem_stat = std_stat / np.sqrt(n_statistics)

    # Compute the confidence interval using t-distribution
    confidence_interval = stats.t.interval(confidence_level, n_statistics - 1, loc=mean_stat, scale=sem_stat)

    return confidence_interval[0][0], confidence_interval[1][0]


dataset = 'test'
n_statistics = 10
metric = 'auc_pr'
confidence_level = 0.95

metrics_tbl = pd.read_excel('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/metrics_per_set_ffivs2minvsffitl_4-7-2025_1440.xlsx')

experiments = metrics_tbl['experiment'].unique()

confidence_interval_dict = {field: [] for field in ['experiment_name', 'left_bound', 'right_bound']}
for experiment in experiments:

    experiment_metrics = metrics_tbl.loc[metrics_tbl['experiment'] == experiment]

    experiment_confidence_interval = compute_confidence_interval_for_cv_experiment(
        experiment_metrics,
        confidence_level,
        dataset,
        metric,
        n_statistics
    )

    confidence_interval_dict['experiment_name'].append(experiment)
    confidence_interval_dict['left_bound'].append(experiment_confidence_interval[0])
    confidence_interval_dict['right_bound'].append(experiment_confidence_interval[1])

confidence_interval_df = pd.DataFrame(confidence_interval_dict)
confidence_interval_df.to_csv(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/confidence_interval_{confidence_level}_{dataset}set_metric_{metric}_ffivs2minvsffitl_4-7-2025_1440.csv', index=False)

#%% Perform Wilcoxon Signed-Rank Test

metric = 'auc_pr'
dataset = 'test'
n_statistics = 10

exp1_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_nobatchnorm_4-4-2025_1545')
exp2_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_transferlearning_lr1e-7_fclayertrained_patience50ffi_nobatchnorm_4-7-2025_1737')

exp1_metric_vals, exp2_metric_vals = [], []
for stat_i in range(n_statistics):

    experiment1_res = np.load(exp1_dir / f'cv_iter_{stat_i}' / 'ensemble_model' / 'res_eval.npy', allow_pickle=True).item()
    experiment2_res = np.load(exp2_dir / f'cv_iter_{stat_i}' / 'ensemble_model' / 'res_eval.npy',
                              allow_pickle=True).item()

    experiment1_metric = experiment1_res[f'{dataset}_{metric}']
    experiment2_metric = experiment2_res[f'{dataset}_{metric}']

    exp1_metric_vals.append(experiment1_metric)
    exp2_metric_vals.append(experiment2_metric)

# Compute the Wilcoxon Signed-Rank Test
stat, p_value = wilcoxon(exp2_metric_vals, exp1_metric_vals)

print(f'Wilcoxon Signed-Rank Test statistic: {stat}')
print(f'p-value: {p_value}')

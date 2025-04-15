"""
Compute Cohen's effect size for performance metrics between two cross-validation experiments.
"""

# 3rd party
import numpy as np
import pandas as pd
from itertools import combinations


def compute_effect_size(experiment1, experiment2, metric, n_statistics):

    m1 = experiment1.loc[experiment1['statistic'] == 'mean', metric].values[0]
    m2 = experiment2.loc[experiment2['statistic'] == 'mean', metric].values[0]

    s1 = experiment1.loc[experiment1['statistic'] == 'std', metric].values[0]
    s2 = experiment2.loc[experiment2['statistic'] == 'std', metric].values[0]

    pooled_std  = np.sqrt(((n_statistics - 1) * s1 ** 2 + (n_statistics - 1) * s2 ** 2) / (2 * n_statistics - 2))

    mean_diff = m1 - m2

    effect_size = mean_diff / pooled_std

    return effect_size


dataset = 'test'
n_statistics = 10
metric = 'auc_pr'

metrics_tbl = pd.read_excel('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/metrics_per_set_ffivs2minvsffitl_4-7-2025_1440.xlsx')

experiments = metrics_tbl['experiment'].unique()
experiments_pair_combinations = list(combinations(experiments, 2))

effect_size_dict = {field: [] for field in ['experiment_1', 'experiment_2', 'cohen_effect_size']}
for experiment_pair in experiments_pair_combinations:

    if '2min' in experiment_pair:
        experiment_pair = sorted(experiment_pair)

    experiment1_metrics = metrics_tbl.loc[metrics_tbl['experiment'] == experiment_pair[0]]
    experiment2_metrics = metrics_tbl.loc[metrics_tbl['experiment'] == experiment_pair[1]]

    experiment1_metrics = experiment1_metrics.loc[experiment1_metrics['dataset'] == dataset]
    experiment2_metrics = experiment2_metrics.loc[experiment2_metrics['dataset'] == dataset]

    cohen_eff_size = compute_effect_size(
        experiment1_metrics,
        experiment2_metrics,
        metric,
        n_statistics
    )

    effect_size_dict['experiment_1'].append(experiment_pair[0])
    effect_size_dict['experiment_2'].append(experiment_pair[1])
    effect_size_dict['cohen_effect_size'].append(cohen_eff_size)

effect_size_df = pd.DataFrame(effect_size_dict)
effect_size_df.to_csv(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cohen_effect_size_{dataset}set_metric_{metric}_ffivs2minvsffitl_4-7-2025_1440.csv', index=False)

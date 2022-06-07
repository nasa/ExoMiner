# 3rd party
import pandas as pd
from pathlib import Path
# import itertools
# import numpy as np
from scipy.special._comb import _comb_int as comb
import multiprocessing
import numpy as np
from functools import reduce


def compute_mc(n_feats_in_run, n_features, score_config_run, score_config_run_minusfeat):
    """ Compute marginal contributions for a given feature and configuration of features.

    mc(feat, config_features) = n_features_in_config * Combinations(n_features_total, n_features_in_config) ^ (-1) * delta_score(feat, config_features)
    delta_score(feat, config_features) = score(config_features) - score(config_features_wo_feat)

    :param n_feats_in_run: int, number of features in configuration
    :param n_features: int, total number of  features
    :param score_config_run: float, score for model using all features in configuration
    :param score_config_run_minusfeat: float, score for model using  all features in configuration except feature for
    which marginal contribution is computed
    :return:
        mc: float, marginal contribution for feature feat and for a given configuration of features
    """

    mc = (n_feats_in_run * comb(n_features, n_feats_in_run)) ** (-1) * (score_config_run - score_config_run_minusfeat)

    return mc


def compute_shap_value(mc_feats):
    """ Compute SHAP value for a feature by summing the marginal contributions.

    :param mc_feats: list, marginal contributions for feature feat
    :return:
        float, SHAP value for the feature
    """

    return sum(mc_feats)


# def compute_for_example():
#  return


def compute_mc_for_feat_in_config_run(feat, config_run, feats_in_run, runs_tbl, examples_tbl):
    print(f'Computing MC for feature {feat} in run {config_run}')

    # get score of run that does not use the feature
    config_run_minusfeat = '-'.join([el for el in feats_in_run if el != feat])

    for example_i, example in runs_tbl.iterrows():  # iterate through examples

        print(f'Computing MC for feature {feat} in run {config_run} for example {example_i} '
              f'(out of {len(runs_tbl)})')

        # get score of run that includes the feature
        score_config_run = runs_tbl.xs(config_run, axis=0, level='run').loc[example_i[:2], 'score']

        if len(config_run_minusfeat) == 0:
            score_config_run_minusfeat = score_nofeats
        else:
            score_config_run_minusfeat = runs_tbl.xs(config_run_minusfeat, axis=0, level='run').loc[
                example_i[:2], 'score']

        # compute marginal contribution for adding feat to set of features in this run
        examples_tbl.loc[example_i[:2], feat] = \
            examples_tbl.loc[example_i[:2], feat] + \
            (n_feats_in_run * comb(n_features, n_feats_in_run)) ** (-1) * (
                    score_config_run - score_config_run_minusfeat)

    return examples_tbl


# %%

shap_root_dir = Path('/data5/tess_project/experiments/current_experiments/explainability/shap_1-11-2022/')

examples_tbls_dir = Path('/data5/tess_project/experiments/current_experiments/explainability/shap_1-11-2022/'
                         'results_ensemble/gf-lf-oe-lc-wks-s-dvtf/')
examples_cols = [
    'target_id',
    'tce_plnt_num',
    'label'
]
datasets = [
    'train',
    'val',
    'test',
    'predict'
]
examples_tbls = {dataset: pd.read_csv(examples_tbls_dir /
                                      f'ensemble_ranked_predictions_{dataset}set.csv')[examples_cols]
                 for dataset in datasets}

# score when no feature is used
score_nofeats = examples_tbls['train']['label'].mean()

features = [
    'gf',
    'lf',
    'oe',
    'lc',
    # 'gc-lc',
    'wks',
    's',
    'dvtf'
]

config_runs = []
with open(shap_root_dir / 'list_config_runs_only_lc.txt', 'r') as runs_file:
    for line in runs_file:
        config_runs.append(line.split('_')[2][:-1])

runs_tbl = []  # {dataset: [] for dataset  in datasets}
for dataset in datasets:
    for config_run in config_runs:
        runs_tbl.append(pd.read_csv(
            shap_root_dir / 'results_ensemble' / config_run / f'ensemble_ranked_predictions_{dataset}set.csv'))
        runs_tbl[-1]['dataset'] = dataset
        runs_tbl[-1]['run'] = config_run

runs_tbl = pd.concat(runs_tbl, axis=0)
runs_tbl = runs_tbl.set_index(['target_id', 'tce_plnt_num', 'run'])

# runs_tbl = pd.concat([runs_tbl,
#                       pd.DataFrame(data=np.zeros((len(runs_tbl), n_features)), columns=features)], axis=1)


# n_features = len(features)
examples_tbl = []  # {dataset: [] for dataset  in datasets}
for dataset in datasets:
    examples_tbl.append(pd.read_csv(
        shap_root_dir / 'results_ensemble' / examples_tbls_dir / f'ensemble_ranked_predictions_{dataset}set.csv'))
    examples_tbl[-1]['dataset'] = dataset

examples_tbl = pd.concat(examples_tbl, axis=0)
examples_tbl = examples_tbl.set_index(['target_id', 'tce_plnt_num'])

n_features = len(features)
for feat in features:
    examples_tbl[feat] = 0

# for dataset_name, dataset_tbl in examples_tbls.items():  # iterate through datasets
#     print(f'Iterating through dataset {dataset_name}')
#     for example_i, example in dataset_tbl.iterrows():  # iterate through examples
#         print(f'Iterating through example {example_i} in dataset {dataset_name}')
#         for feat in features:  # iterate through features
#             print(f'Iterating through feature {feat} for example {example_i} in dataset {dataset_name}')
#             mc_feat = []
#             config_runs_feat = [config_run for config_run in config_runs if feat in config_run]  # get runs that use the feature
#             for config_run in config_runs_feat:  # iterate through all runs that use the feature to compute MCs
#                 print(f'Computing MC for feature {feat} in run {config_run}, example {example_i} in dataset {dataset_name}')
#
#                 feats_in_run = config_run.split('-')  # set of features used in the run
#                 n_feats_in_run = len(feats_in_run)
#
#                 # get score of run that includes the feature
#                 # results_config_run = pd.read_csv(shap_root_dir / 'results_ensemble' / config_run / f'ensemble_ranked_predictions_{dataset_name}set.csv')
#                 # score_config_run = results_config_run.loc[example_i, 'score']
#                 score_config_run = runs_tbl[dataset].loc[(example['target_id'], example['tce_plnt_num'], config_run)]
#
#                 # get score of run that does not use the feature
#                 config_run_minusfeat = '-'.join([el for el in feats_in_run if el != feat])
#                 if len(config_run_minusfeat) == 0:
#                     score_config_run_minusfeat = score_nofeats
#                 else:
#                     # results_config_run_minusfeat = pd.read_csv(shap_root_dir / 'results_ensemble' / config_run_minusfeat / f'ensemble_ranked_predictions_{dataset_name}set.csv')
#                     # score_config_run_minusfeat = results_config_run_minusfeat.loc[example_i, 'score']
#                     score_config_run_minusfeat = runs_tbl[dataset].loc[(example['target_id'], example['tce_plnt_num'], config_run_minusfeat)]
#
#                 # compute marginal contribution for adding feat to set of features in this run
#                 mc_feat.append((n_feats_in_run * comb(n_features, n_feats_in_run)) ** (-1) *
#                                (score_config_run - score_config_run_minusfeat))
#
#             print(f'Computing SHAP value for feature {feat}, example {example_i} in dataset {dataset_name}')
#             # compute SHAP value of feat by summing its marginal contributions
#             dataset_tbl.loc[example_i, feat] = sum(mc_feat)

n_processes = 11
for config_run in config_runs:  # iterate through all runs that use the feature to compute MCs

    print(f'Computing MC for features in run {config_run}')

    feats_in_run = config_run.split('-')  # set of features used in the run
    n_feats_in_run = len(feats_in_run)

    feats_jobs = np.array_split(feats_in_run, n_processes)
    n_processes = len(feats_jobs)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(feat_job,) + (config_run, feats_in_run, runs_tbl, examples_tbl)
            for feat_job_i, feat_job in enumerate(feats_jobs)]
    async_results = [pool.apply_async(compute_mc_for_feat_in_config_run, job) for job in jobs]
    pool.close()

    # time.sleep(5)
    mc_tbls = [async_result.get()[features] for async_result in async_results]
    res = reduce(lambda df1, df2: df1.add(df2), mc_tbls)  # some MCs across feats runs

examples_tbl.to_csv(shap_root_dir / f'shap_values.csv')
# # save SHAP values results for each dataset table
# for dataset_name, dataset_tbl in examples_tbls.items():
#     dataset_tbl.to_csv(shap_root_dir / f'{dataset_name}set_shap_values.csv', index=False)

# %%

import time

t0 = time.time()
n = 10000

for i in range(n):
    b = pd.read_csv(
        shap_root_dir / 'results_ensemble' / config_run / f'ensemble_ranked_predictions_{dataset_name}set.csv')
t1 = time.time()
print(f'Time spent: ~{(t1 - t0) / n}')

# %%

# runs_tbl.loc[(slice(None), slice(None), 'gf')]
# runs_tbl.query("run == 'gf'")

a = runs_tbl.xs(config_run, axis=0, level='run').loc[example_i[:2], 'score']

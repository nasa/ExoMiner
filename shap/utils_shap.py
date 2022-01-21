"""
1) Prepare data to compute mc values.
2) Aggregate mc results into SHAP values
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
from functools import reduce

# %% Prepare data for runs to compute mc values

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

# # score when no feature is used
# score_nofeats = examples_tbls['train']['label'].mean()

features = [
    'gf',
    'lf',
    'oe',
    # 'lc',
    'gc-lc',
    'wks',
    's',
    'dvtf'
]

config_runs = []
config_runs_fn = 'list_config_runs'
with open(shap_root_dir / f'{config_runs_fn}.txt', 'r') as runs_file:
    for line in runs_file:
        config_runs.append(line.split('_')[2][:-1])
config_runs = [config_run for config_run in config_runs if 'gc-lc' not in config_run]
np.save(shap_root_dir / f'{config_runs_fn}_no_gc-lc.npy', config_runs)

runs_tbl = []
for dataset in datasets:
    for config_run in config_runs:
        runs_tbl.append(pd.read_csv(
            shap_root_dir / 'results_ensemble' / config_run / f'ensemble_ranked_predictions_{dataset}set.csv'))
        runs_tbl[-1]['dataset'] = dataset
        runs_tbl[-1]['run'] = config_run

runs_tbl = pd.concat(runs_tbl, axis=0)
runs_tbl = runs_tbl.set_index(['target_id', 'tce_plnt_num', 'run'])

runs_tbl.to_csv(shap_root_dir / 'runs_tbl.csv')

examples_tbl = []
for dataset in datasets:
    examples_tbl.append(pd.read_csv(
        shap_root_dir / 'results_ensemble' / examples_tbls_dir / f'ensemble_ranked_predictions_{dataset}set.csv'))
    examples_tbl[-1]['dataset'] = dataset

examples_tbl = pd.concat(examples_tbl, axis=0)
examples_tbl = examples_tbl.set_index(['target_id', 'tce_plnt_num'])

n_features = len(features)
for feat in features:
    examples_tbl[feat] = 0

examples_tbl.to_csv(shap_root_dir / 'examples_tbl.csv')

# %% Aggregate mcs into SHAP values

mc_dir = shap_root_dir / 'mc_results'

mc_tbls = [pd.read_csv(tbl_fp, index_col=['target_id', 'tce_plnt_num']) for tbl_fp in mc_dir.iterdir()]
shap_tbl = reduce(lambda df1, df2: df1.add(df2), mc_tbls)  # sum MCs across feature config runs

shap_tbl.to_csv(shap_root_dir / f'shap.csv')

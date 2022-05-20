"""
1) Prepare data to compute mc values. This involves creating examples and runs tables. The runs table contains the
scores for all examples across all runs needed for SHAP.
2) Aggregate marginal contributions from different runs into SHAP values for the different groups of features.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
from functools import reduce
import yaml

# %% Prepare data for runs to compute mc values

shap_root_dir = Path('/data5/tess_project/experiments/current_experiments/explainability/shap_1-11-2022/')

examples_tbls_dir = Path('/data5/tess_project/experiments/current_experiments/explainability/shap_1-11-2022/'
                         'results_ensemble/gf-lf-oe-lc-wks-s-dvtf/')

# get the configuration parameters
path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/shap/config_compute_mcs.yaml')

with(open(path_to_yaml, 'r')) as file:
    config = yaml.safe_load(file)

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

config_runs = []
config_runs_fn = 'list_config_runs_only_gc-lc'
with open(shap_root_dir / f'{config_runs_fn}.txt', 'r') as runs_file:
    for line in runs_file:
        config_runs.append(line.split('_')[2][:-1])
# config_runs = [config_run for config_run in config_runs if 'gc-lc' in config_run]
np.save(shap_root_dir / f'{config_runs_fn}.npy', config_runs)

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

n_features = len(config['features'])
for feat in config['features']:
    examples_tbl[feat] = 0

examples_tbl.to_csv(shap_root_dir / 'examples_tbl.csv')

# %% Aggregate mcs into SHAP values

shap_root_dir = Path('/data5/tess_project/experiments/current_experiments/explainability/shap_1-11-2022/')

# extract useful columns from examples table to add to shap table
examples_tbl = pd.read_csv(shap_root_dir / 'examples_tbl.csv')
examples_tbl_cols = ['target_id', 'tce_plnt_num', 'label', 'original_label', 'tce_duration', 'tce_period',
                     'tce_time0bk', 'transit_depth', 'score', 'predicted class', 'dataset']
examples_tbl = examples_tbl[examples_tbl_cols]

# load MC results
mc_dir = shap_root_dir / 'mc_results'
mc_tbls = [pd.read_csv(tbl_fp, index_col=['target_id', 'tce_plnt_num']) for tbl_fp in mc_dir.iterdir()
           if tbl_fp.suffix == '.csv']
shap_tbl = reduce(lambda df1, df2: df1.add(df2), mc_tbls)  # sum MCs across feature config runs
shap_tbl.reset_index(drop=False, inplace=True)

shap_tbl = examples_tbl.merge(shap_tbl, on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')

shap_tbl.to_csv(shap_root_dir / f'shap.csv', index=False)

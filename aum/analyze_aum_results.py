"""
Analyze AUM results:
- Aggregate results across runs.
"""

# 3rd party
from pathlib import Path
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.stats import mad_std

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-02-2022_1324')

runs_dir = experiment_dir / 'runs'
aum_tbls = {f'{run_dir.name}': pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv') for run_dir in
            sorted(runs_dir.iterdir())}
runs = list(aum_tbls.keys())

# unchanged columns in AUM tables
fixed_cols = [col for col in aum_tbls['run0'].columns if col not in ['margin', 'label', 'label_id']]
# get AUM values for each run
aum_vals_tbl = pd.concat(
    [aum_tbl[['margin']].rename(columns={'margin': f'margin_{run}'}) for run, aum_tbl in aum_tbls.items()], axis=1)
aum_allruns_tbl = pd.concat([
    aum_tbls['run0'][fixed_cols],
    aum_vals_tbl
], axis=1)

# set indicator for examples in the training set that changed their label in runs
trainset_tbls_dir = experiment_dir / 'trainset_runs'
trainset_tbls = {fp.stem: pd.read_csv(fp) for fp in trainset_tbls_dir.iterdir()}
aum_allruns_tbl['runs_changed'] = ''
for trainset_tbl_name, trainset_tbl in trainset_tbls.items():
    trainset_tbl.set_index(['target_id', 'tce_plnt_num'], inplace=True)
    for example_i, example in aum_allruns_tbl.iterrows():
        if aum_allruns_tbl.loc[example_i, 'dataset'] == 'train':
            if trainset_tbl.loc[(example['target_id'], example['tce_plnt_num']), 'label_changed']:
                aum_allruns_tbl.loc[example_i, 'runs_changed'] = \
                    aum_allruns_tbl.loc[example_i, 'runs_changed'] + f'{trainset_tbl_name.split("_")[-1]} '

# compute statistics  across runs; don't consider runs for which example was mislabeled
for stat in ['mean', 'std', 'median', 'mad_std']:
    aum_allruns_tbl[stat] = np.nan
for example_i, example in aum_allruns_tbl.iterrows():
    # if example['runs_changed'] != '':
    runs_unchanged = [f'margin_{run}' for run in runs if run not in example['runs_changed']]

    aum_allruns_tbl.loc[example_i, 'mean'] = np.mean(example[runs_unchanged])
    aum_allruns_tbl.loc[example_i, 'std'] = np.std(example[runs_unchanged], ddof=1)
    aum_allruns_tbl.loc[example_i, 'median'] = np.median(example[runs_unchanged])
    aum_allruns_tbl.loc[example_i, 'mad_std'] = mad_std(example[runs_unchanged])

# mean_aum_tbl = aum_vals_tbl.mean(axis=1).to_frame(name='aum_mean')
# std_aum_tbl = aum_vals_tbl.std(axis=1).to_frame(name='aum_std')
# median_aum_tbl = aum_vals_tbl.median(axis=1).to_frame(name='aum_median')
# mad_aum_tbl = aum_vals_tbl.mad(axis=1).to_frame(name='aum_mad')
#
# aum_allruns_tbl = pd.concat([
#     aum_tbls['run0'][fixed_cols],
#     mean_aum_tbl,
#     std_aum_tbl,
#     median_aum_tbl,
#     mad_aum_tbl,
#     aum_vals_tbl
# ], axis=1)

aum_allruns_tbl.to_csv(experiment_dir / 'aum.csv', index=False)

# %% Compute nth percentile

n_percentile = 99
noise_label = 'MISLABELED'

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-02-2022_1324')

runs_dir = experiment_dir / 'runs'
margin_thr = {}
for run_dir in sorted(runs_dir.iterdir()):
    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    margin_mislabeled = aum_tbl.loc[aum_tbl['label'] == noise_label, 'margin']

    margin_thr[run_dir.name] = np.percentile(margin_mislabeled, n_percentile)

runs = list(margin_thr.keys())
margin_thr_df = pd.Series(data=margin_thr, name='thr')
margin_thr_df['mean'] = np.mean(margin_thr_df[runs].values)
margin_thr_df['std'] = np.std(margin_thr_df[runs].values, ddof=1)
margin_thr_df['median'] = np.median(margin_thr_df[runs].values)
margin_thr_df['mad_std'] = mad_std(margin_thr_df[runs])

margin_thr_df.to_csv(experiment_dir / f'margin_thr_{n_percentile}_percentile.csv')

# %% Use margin threshold to determin which examples are mislabeled

margin_thr_df = pd.read_csv(experiment_dir / f'margin_thr_{n_percentile}_percentile.csv', squeeze=True, index_col=0)

aum_allruns_tbl = pd.read_csv(experiment_dir / 'aum.csv')

aum_allruns_tbl['mislabeled_by_aum'] = 'no'
aum_allruns_tbl.loc[aum_allruns_tbl['mean'] > margin_thr_df['mean'], 'mislabeled_by_aum'] = 'yes'

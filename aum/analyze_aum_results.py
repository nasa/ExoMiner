"""
Analyze AUM results:
- Aggregate results across runs.
- Compute threshold for each run and across-run stats.
- Compute AUM across runs and across-runs stats.
- Set mislabeled examples for each individual run and for aggregated results across runs.
- Check method's sensitivity to different sets of thresholded examples.
"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.stats import mad_std

# %% aggregate AUM across runs in an experiment

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-11-2022_1144')

runs_dir = experiment_dir / 'runs'
aum_tbls = {f'{run_dir.name}': pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv') for run_dir in
            sorted(runs_dir.iterdir()) if run_dir.is_dir()}
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

# experiment_dir = Path(
#     '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

runs_dir = experiment_dir / 'runs'
margin_thr = {}
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:
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

# %% Use margin threshold to determine which examples are mislabeled

# experiment_dir = Path(
#     '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

margin_thr_df = pd.read_csv(experiment_dir / f'margin_thr_{n_percentile}_percentile.csv', squeeze=True, index_col=0)

aum_allruns_tbl = pd.read_csv(experiment_dir / 'aum.csv')

aum_allruns_tbl['mislabeled_by_aum'] = 'no'
aum_allruns_tbl.loc[aum_allruns_tbl['mean'] < margin_thr_df['mean'], 'mislabeled_by_aum'] = 'yes'

aum_allruns_tbl.to_csv(experiment_dir / f'aum_mislabeled.csv', index=False)

# %% look at distribution of AUM

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-03-2022_1052/')

for run in range(9):

    # run_dir = experiment_dir / 'runs' / 'run0'
    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')

    # bins = np.linspace(-50, 50, 100)
    #
    # f, ax = plt.subplots()
    # ax.hist(aum_tbl.loc[aum_tbl['label'] != 'MISLABELED', 'margin'], bins, edgecolor='k', label='Other Samples')
    # ax.hist(aum_tbl.loc[aum_tbl['label'] == 'MISLABELED', 'margin'], bins, edgecolor='k', label='Thr. Samples')
    # ax.set_xlabel('AUM')
    # ax.set_ylabel('Counts')
    # ax.set_yscale('log')
    # ax.legend()
    # f.savefig(run_dir / 'hist_aum_thr_vs_normal_samples.png')
    # plt.close()
    #
    # labels = {
    #     'PC': {'color': 'b', 'zorder': 1, 'alpha': 1.0},
    #     'AFP': {'color': 'b', 'zorder': 2, 'alpha': 1.0},
    #     'MISLABELED': {'color': 'b', 'zorder': 3, 'alpha': 1.0},
    #     'NTP': {'color': 'b', 'zorder': 2, 'alpha': 1.0},
    #     'UNK': {'color': 'b', 'zorder': 2, 'alpha': 1.0},
    #
    # }
    # # f, ax = plt.subplots()
    # # for label in labels:  # aum_tbl['label'].unique():
    # #     ax.hist(aum_tbl.loc[aum_tbl['label'] != label, 'margin'], bins, edgecolor='k', label=f'{label}', **labels[label])
    # # ax.set_xlabel('AUM')
    # # ax.set_ylabel('Counts')
    # # ax.set_yscale('log')
    # # ax.legend()
    # # plt.show()
    # for label in labels:  # aum_tbl['label'].unique():
    #     f, ax = plt.subplots()
    #     ax.hist(aum_tbl.loc[aum_tbl['label'] != label, 'margin'], bins, edgecolor='k', label=f'{label}',
    #             **labels[label])
    #     ax.set_xlabel('AUM')
    #     ax.set_ylabel('Counts')
    #     ax.set_yscale('log')
    #     ax.legend()
    #     f.savefig(run_dir / f'hist_aum_{label}.png')
    # plt.close()

    # %% look at margin change over epochs

    # experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-03-2022_1052')

    # run_dir = experiment_dir / 'runs' / 'run0'

    margins_dir = run_dir / 'models' / 'model1' / 'margins'

    cols_tbl = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label', 'margin']
    drop_dupl_cols = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label']

    margins_tbl = [pd.read_csv(tbl, usecols=cols_tbl) if tbl_i == 0 else pd.read_csv(tbl, usecols=['margin'])
                   for tbl_i, tbl in enumerate(margins_dir.iterdir())]
    margins_tbl = pd.concat(margins_tbl, axis=1)
    margins_tbl.to_csv(run_dir / 'margins_tbl.csv', index=False)
    # margins_tbl.drop_duplicates(drop_dupl_cols, inplace=True)

    margins_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

    n_epochs = 50

    f, ax = plt.subplots()
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(10982872, 3), 'margin'], label='PC')
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(7983756, 1), 'margin'], label='AFP')
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(6141300, 1), 'margin'], label='NTP')
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(10460984, 1), 'margin'], label='MISLABELED')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Margin')
    ax.legend()
    f.savefig(run_dir / 'margin_over_epochs_examples_all.png')
    plt.close()

    labels = {
        'PC': {'color': 'b', 'zorder': 1, 'alpha': 1.0},
        'AFP': {'color': 'g', 'zorder': 2, 'alpha': 1.0},
        'MISLABELED': {'color': 'r', 'zorder': 3, 'alpha': 1.0},
        'NTP': {'color': 'c', 'zorder': 2, 'alpha': 1.0},
        'UNK': {'color': 'm', 'zorder': 2, 'alpha': 1.0},

    }
    mean_margin = {label: np.mean(margins_tbl.loc[margins_tbl['label'] == label, 'margin']).values for label in labels}
    std_margin = {label: np.std(margins_tbl.loc[margins_tbl['label'] == label, 'margin'], ddof=1).values for label in
                  labels}

    f, ax = plt.subplots()
    for label in labels:
        ax.plot(mean_margin[label], label=label, color=labels[label]['color'])
        # ax.plot(mean_margin[label] + std_margin[label], linestyle='--', color=labels[label]['color'])
        # ax.plot(mean_margin[label] - std_margin[label], linestyle='--', color=labels[label]['color'])
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Margin')
    ax.legend()
    f.savefig(run_dir / 'margin_over_epochs_all.png')
    plt.close()

    f, ax = plt.subplots()
    for label in ['PC', 'AFP', 'MISLABELED']:
        ax.plot(mean_margin[label], label=label, color=labels[label]['color'])
        ax.plot(mean_margin[label] + std_margin[label], linestyle='--', color=labels[label]['color'])
        ax.plot(mean_margin[label] - std_margin[label], linestyle='--', color=labels[label]['color'])
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Margin')
    ax.legend()
    f.savefig(run_dir / 'margin_over_epochs_pc-afp-mislabeled_var.png')
    plt.close()

# %% determine mislabeled examples per run to check method's sensitivity to different sets of thresholded samples

n_percentile = 99
noise_label = 'MISLABELED'

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-11-2022_1144')

runs_dir = experiment_dir / 'runs'

aum_tbls = []
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:
    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    margin_mislabeled = aum_tbl.loc[aum_tbl['label'] == noise_label, 'margin']

    # computer threshold using thresholded examples
    margin_thr = np.percentile(margin_mislabeled, n_percentile)
    aum_tbl[f'margin_thr_{n_percentile}'] = margin_thr

    aum_tbl['mislabeled_by_aum'] = 'no'
    # set examples with AUM lower than threshold to mislabeled
    aum_tbl.loc[aum_allruns_tbl['mean'] < margin_thr, 'mislabeled_by_aum'] = 'yes'

    aum_tbl.to_csv(run_dir / f'aum_mislabeled.csv', index=False)

    aum_tbls.append(aum_tbl)

# count number of times each example was determined as mislabeled across runs
aum_cnts = aum_tbls[0][['target_id', 'tce_plnt_num', 'original_label', 'dataset', 'shard_name']]
aum_cnts['counts_mislabeled'] = 0
for aum_tbl in aum_tbls:
    aum_cnts.loc[aum_tbl['mislabeled_by_aum'] == 'yes', 'counts_mislabeled'] += 1
aum_cnts.to_csv(experiment_dir / 'aum_mislabeled_cnts.csv', index=False)

# plot histogram of counts per label
bins = np.linspace(0, 10, 11, endpoint=True, dtype='int')
for label in ['PC', 'AFP', 'NTP', 'UNK']:
    f, ax = plt.subplots()
    ax.hist(aum_cnts.loc[aum_cnts['original_label'] == label, 'counts_mislabeled'], bins, edgecolor='k')
    ax.set_ylabel(f'Counts {label}')
    ax.set_xlabel('Number of runs example is determined mislabeled')
    ax.set_yscale('log')
    ax.set_xticks(bins + 0.5)
    ax.set_xticklabels(bins)
    ax.set_xlim([bins[0], bins[-1]])
    ax.grid(axis='y')
    f.savefig(experiment_dir / f'hist_mislabeled_cnts_{label}.png')

# plot histogram of counts per dataset
for dataset in ['train', 'val', 'test', 'predict']:
    f, ax = plt.subplots()
    ax.hist(aum_cnts.loc[aum_cnts['dataset'] == dataset, 'counts_mislabeled'], bins, edgecolor='k')
    ax.set_ylabel(f'Counts {dataset}')
    ax.set_xlabel('Number of runs example is determined mislabeled')
    ax.set_yscale('log')
    ax.set_xticks(bins + 0.5)
    ax.set_xticklabels(bins)
    ax.set_xlim([bins[0], bins[-1]])
    ax.grid(axis='y')
    f.savefig(experiment_dir / f'hist_mislabeled_cnts_{dataset}.png')

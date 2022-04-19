"""
Analyze AUM results:
- Aggregate results across runs.
- Compute threshold for each run and across-run stats.
- Compute AUM and relative stats across runs.
- Set mislabeled examples for each individual run and for aggregated results across runs.
- Check method's sensitivity to different sets of thresholded examples using ranking similarity,
counting number of runs in which examples are mislabeled or in top-k.
- Compute recall for retrieving injected label noise examples.
"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.stats import mad_std
from itertools import combinations
import yaml
import rbo
from scipy.stats import kendalltau, weightedtau
import multiprocessing

# local
from src.utils_visualization import plot_class_distribution
from aum.create_mislabeled_dataset import create_tfrec_dataset_tbl

# %% aggregate AUM across runs in an experiment

inj_thr = False  # set to True if label noise was injected into the experiment and used to threshold examples
add_inj_info = False
# unchanged columns in AUM tables
fixed_cols = ['target_id', 'tce_plnt_num', 'label_id', 'original_label', 'shard_name', 'dataset']
epoch_chosen = 499  # epoch chosen to get AUM values

# experiment directory
experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')

# get AUM tables for each run
runs_dir = experiment_dir / 'runs'
aum_tbls = {f'{run_dir.name}':
                pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')[fixed_cols + [f'epoch_{epoch_chosen}']]
            for run_dir in sorted(runs_dir.iterdir()) if run_dir.is_dir()}
runs = list(aum_tbls.keys())

# concatenate AUM values for each run
aum_vals_tbl = pd.concat(
    [aum_tbl[[f'epoch_{epoch_chosen}']].rename(columns={f'epoch_{epoch_chosen}': f'{run}'})
     for run, aum_tbl in aum_tbls.items()], axis=1)
aum_allruns_tbl = pd.concat([aum_tbls['run0'][fixed_cols], aum_vals_tbl], axis=1)

# set indicator for examples in the training set that changed their label in runs
trainset_tbls_dir = experiment_dir / 'trainset_runs'
trainset_tbls = {fp.stem: pd.read_csv(fp) for fp in trainset_tbls_dir.iterdir()}
aum_allruns_tbl['runs_changed'] = ''
for trainset_tbl_name, trainset_tbl in trainset_tbls.items():
    trainset_tbl.set_index(['target_id', 'tce_plnt_num'], inplace=True)
    if inj_thr:
        trainset_tbl.loc[trainset_tbl['label_changed_to_other_class'].isna()] = False
    for example_i, example in aum_allruns_tbl.iterrows():
        if aum_allruns_tbl.loc[example_i, 'dataset'] == 'train':

            if inj_thr:
                col_name = 'label_changed_to_other_class'
            else:
                col_name = 'label_changed_to_mislabeled'

            if trainset_tbl.loc[(example['target_id'], example['tce_plnt_num']), col_name]:
                aum_allruns_tbl.loc[example_i, 'runs_changed'] = \
                    aum_allruns_tbl.loc[example_i, 'runs_changed'] + f'{trainset_tbl_name.split("_")[-1]} '

# compute statistics  across runs; don't consider runs for which example was mislabeled
for stat in ['mean', 'std', 'median', 'mad_std']:
    aum_allruns_tbl[stat] = np.nan
for example_i, example in aum_allruns_tbl.iterrows():
    # if example['runs_changed'] != '':
    runs_unchanged = [f'{run}' for run in runs if run not in example['runs_changed']]

    aum_allruns_tbl.loc[example_i, 'mean'] = np.mean(example[runs_unchanged])
    aum_allruns_tbl.loc[example_i, 'std'] = np.std(example[runs_unchanged], ddof=1)
    aum_allruns_tbl.loc[example_i, 'median'] = np.median(example[runs_unchanged])
    aum_allruns_tbl.loc[example_i, 'mad_std'] = mad_std(example[runs_unchanged])

# add column related to injected stochastic noise to the AFP and PC populations in the different datasets
cols = ['target_id', 'tce_plnt_num', 'label_changed_to_other_class']
aum_allruns_tbl['label_changed_to_other_class'] = False
if add_inj_info:
    # trainset_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-01-2022_1433/train_set_labels_switched.csv')[cols]
    valtestset_tbl = pd.read_csv(experiment_dir / 'val_test_sets_labels_switched.csv')[cols]
    dataset_tbl = valtestset_tbl  # pd.concat([trainset_tbl, valtestset_tbl], axis=0)
    aum_allruns_tbl = aum_allruns_tbl.merge(dataset_tbl,
                                            on=['target_id', 'tce_plnt_num'],
                                            how='left',
                                            validate='one_to_one')

aum_allruns_tbl.to_csv(experiment_dir / f'aum_allruns_epoch{epoch_chosen}.csv', index=False)

# %% Compute nth percentile for each run and relative stats

n_percentile = 99
noise_label = 'MISLABELED'
inj_thr = False  # injected noise examples in the training set used to set AUM threshold for mislabeling detection

# experiment_dir = Path(
#     '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

# iterate over each run, get thresholded examples used to compute the n-th percentile for AUM
runs_dir = experiment_dir / 'runs'
aum_thr = {}
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:
    print(f'Computing {n_percentile}-percentile margin threshold for run {run_dir}')

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')

    if not inj_thr:
        aum_thr_examples = aum_tbl.loc[aum_tbl['label'] == noise_label, f'epoch_{epoch_chosen}']
    else:
        trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                                   f'trainset_{run_dir.name}.csv')[['target_id',
                                                                    'tce_plnt_num',
                                                                    'label_changed_to_other_class']]
        aum_tbl = aum_tbl.merge(trainset_tbl,
                                on=['target_id', 'tce_plnt_num'],
                                how='left',
                                validate='one_to_one')
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
        aum_thr_examples = aum_tbl.loc[(aum_tbl['label_changed_to_other_class']) &
                                       (aum_tbl['dataset'] == 'train'), f'epoch_{epoch_chosen}']

    if len(aum_thr_examples) == 0:
        aum_thr[run_dir.name] = np.nan
    else:
        aum_thr[run_dir.name] = np.percentile(aum_thr_examples, n_percentile)

runs = list(aum_thr.keys())
aum_thr_df = pd.Series(data=aum_thr, name='thr')
aum_thr_df['mean'] = np.mean(aum_thr_df[runs].values)
aum_thr_df['std'] = np.std(aum_thr_df[runs].values, ddof=1)
aum_thr_df['median'] = np.median(aum_thr_df[runs].values)
aum_thr_df['mad_std'] = mad_std(aum_thr_df[runs])

aum_thr_df.to_csv(experiment_dir / f'aum_thr_{n_percentile}_percentile.csv')

# %% Use AUM threshold to determine which examples are mislabeled across runs

# experiment_dir = Path(
#     '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

aum_thr_df = pd.read_csv(experiment_dir / f'aum_thr_{n_percentile}_percentile.csv', squeeze=True, index_col=0)

aum_allruns_tbl = pd.read_csv(experiment_dir / f'aum_allruns_epoch{epoch_chosen}.csv')

aum_allruns_tbl['mislabeled_by_aum'] = 'no'
aum_allruns_tbl.loc[aum_allruns_tbl['mean'] < aum_thr_df['mean'], 'mislabeled_by_aum'] = 'yes'

aum_allruns_tbl.to_csv(experiment_dir / f'aum_allruns_epoch{epoch_chosen}_mislabeled.csv', index=False)

# %% plot distribution of AUM at a given epoch and values over epochs

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')
runs_dir = experiment_dir / 'runs'
n_runs = len([fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()])
n_epochs = 499
inj_thr = False
labels = {
    'PC': {'color': 'b', 'zorder': 1, 'alpha': 1.0},
    'AFP': {'color': 'r', 'zorder': 2, 'alpha': 1.0},
    'MISLABELED': {'color': 'g', 'zorder': 3, 'alpha': 1.0},
    'NTP': {'color': 'k', 'zorder': 2, 'alpha': 1.0},
    'UNK': {'color': 'm', 'zorder': 2, 'alpha': 1.0},
    'INJ': {'color': 'y', 'zorder': 4, 'alpha': 1.0},

}
bins = {
    'PC': np.linspace(-5, 5, 100),
    'AFP': np.linspace(-5, 5, 100),
    'NTP': np.linspace(-50, 50, 100),
    'UNK': np.linspace(-25, 25, 100),
    'MISLABELED': np.linspace(-5, 5, 100),
    'INJ': np.linspace(-5, 5, 100),
}
noise_label = 'MISLABELED'

cols_tbl = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label', 'margin']
drop_dupl_cols = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label']

for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl['label_changed_to_other_class'] = False

    if inj_thr:
        # adding column for examples injected with label noise
        trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                                   f'trainset_run{run}.csv')[['target_id',
                                                              'tce_plnt_num',
                                                              'label_changed_to_other_class']]
        valtestset_tbl = pd.read_csv(experiment_dir /
                                     'val_test_sets_labels_switched.csv')[['target_id',
                                                                           'tce_plnt_num',
                                                                           'label_changed_to_other_class']]
        dataset_tbl = pd.concat([trainset_tbl, valtestset_tbl], axis=0)
        aum_tbl = aum_tbl.merge(dataset_tbl,
                                on=['target_id', 'tce_plnt_num'],
                                how='left',
                                validate='one_to_one')
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'], 'label'] = 'INJ'  # noise_label
    else:
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'], 'label'] = noise_label

    # plot histogram of AUM for thresholded and normal examples
    f, ax = plt.subplots()
    ax.hist(aum_tbl.loc[aum_tbl['label'] != 'MISLABELED', f'epoch_{epoch_chosen}'],
            np.linspace(-50, 50, 100),
            edgecolor='k', label='Other Samples')
    if inj_thr:
        ax.hist(aum_tbl.loc[(aum_tbl['label_changed_to_other_class']), f'epoch_{epoch_chosen}'],
                np.linspace(-50, 50, 100),
                edgecolor='k', label='Thr. Samples')
    else:
        ax.hist(aum_tbl.loc[aum_tbl['label'] == 'MISLABELED', f'epoch_{epoch_chosen}'],
                np.linspace(-50, 50, 100),
                edgecolor='k',
                label='Thr. Samples')
    ax.set_xlabel('AUM')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    f.savefig(run_dir / 'hist_aum_thr_vs_normal_samples.png')
    plt.close()

    # plot histogram of AUM for PC, AFP and threshold examples
    f, ax = plt.subplots()
    ax.hist(aum_tbl.loc[aum_tbl['label'].isin(['PC', 'AFP']), f'epoch_{epoch_chosen}'],
            np.linspace(-5, 5, 100),
            edgecolor='k', label='Other Samples PC/AFP', zorder=1, alpha=1)
    if inj_thr:
        ax.hist(aum_tbl.loc[(aum_tbl['label_changed_to_other_class']), f'epoch_{epoch_chosen}'],
                np.linspace(-5, 5, 100),
                edgecolor='k', label='Thr. Samples', zorder=2, alpha=0.5)
    else:
        ax.hist(aum_tbl.loc[aum_tbl['label'] == 'MISLABELED', f'epoch_{epoch_chosen}'],
                np.linspace(-5, 5, 100),
                edgecolor='k', label='Thr. Samples', zorder=2, alpha=0.5)
    ax.set_xlabel('AUM')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    f.savefig(run_dir / 'hist_aum_thr_vs_normal_samples_pc-afp.png')
    plt.close()

    # plot histogram of AUM for each category
    for label in labels:  # aum_tbl['label'].unique():
        f, ax = plt.subplots()
        ax.hist(aum_tbl.loc[aum_tbl['label'] == label, f'epoch_{epoch_chosen}'],
                bins[label],
                edgecolor='k', label=f'{label}', **labels[label])
        ax.set_xlabel('AUM')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.legend()
        f.savefig(run_dir / f'hist_aum_{label}.png')
        plt.close()

    # look at margin change over epochs
    margins_dir = run_dir / 'models' / 'model1' / 'margins'
    margins_tbl = pd.read_csv(margins_dir / 'margins_allepochs.csv')
    margins_tbl['label_changed_to_other_class'] = False

    if inj_thr:
        # add column indicator for injected noise to the margins table
        margins_tbl = margins_tbl.merge(dataset_tbl,
                                        on=['target_id', 'tce_plnt_num'],
                                        how='left',
                                        validate='one_to_one')
        margins_tbl.loc[margins_tbl['label_changed_to_other_class'], 'label'] = 'INJ'  # noise_label

    margins_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

    # # plot margin for a few examples
    # f, ax = plt.subplots()
    # ax.plot(np.arange(n_epochs), margins_tbl.loc[(10982872, 3), 'margin'], label='PC')
    # ax.plot(np.arange(n_epochs), margins_tbl.loc[(7983756, 1), 'margin'], label='AFP')
    # ax.plot(np.arange(n_epochs), margins_tbl.loc[(6141300, 1), 'margin'], label='NTP')
    # # ax.plot(np.arange(n_epochs), margins_tbl.loc[(10460984, 1), 'margin'], label='MISLABELED')
    # ax.set_xlabel('Epoch Number')
    # ax.set_ylabel('Margin')
    # ax.legend()
    # f.savefig(run_dir / 'margin_over_epochs_examples_all.png')
    # plt.close()

    mean_margin = {label: np.mean(margins_tbl.loc[margins_tbl['label'] == label, f'epoch_{epoch_chosen}'])
                   for label in labels}
    std_margin = {label: np.std(margins_tbl.loc[margins_tbl['label'] == label, f'epoch_{epoch_chosen}'], ddof=1)
                  for label in labels}

    # plot avg margin for all examples in all categories
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

    # plot avg margin for PC, AFP, mislabeled examples
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

    # plot avg margin for PC, AFP, mislabeled and injected label noise examples
    f, ax = plt.subplots()
    for label in ['PC', 'AFP', 'MISLABELED', 'INJ']:
        ax.plot(mean_margin[label], label=label, color=labels[label]['color'])
        ax.plot(mean_margin[label] + std_margin[label], linestyle='--', color=labels[label]['color'])
        ax.plot(mean_margin[label] - std_margin[label], linestyle='--', color=labels[label]['color'])
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Margin')
    ax.legend()
    f.savefig(run_dir / 'margin_over_epochs_pc-afp-mislabeled-inj_var.png')
    plt.close()

# %% determine mislabeled examples per run to check method's sensitivity to different sets of thresholded samples

n_percentile = 99  # AUM nth-percentile threshold for mislabeling detection
noise_label = 'MISLABELED'
inj_thr = False  # injected noise examples in the training set used to set AUM threshold for mislabeling detection
labels = ['PC', 'AFP', 'NTP', 'UNK']
experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')

runs_dir = experiment_dir / 'runs'

aum_tbls = []
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:

    print(f'Getting ranking order for run {run_dir.name}...')

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl['label_changed_to_other_class'] = False

    if inj_thr:
        # adding column for examples injected with label noise
        trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                                   f'trainset_{run_dir.name}.csv')[['target_id',
                                                                    'tce_plnt_num',
                                                                    'label_changed_to_other_class']]
        valtestset_tbl = pd.read_csv(experiment_dir /
                                     'val_test_sets_labels_switched.csv')[['target_id',
                                                                           'tce_plnt_num',
                                                                           'label_changed_to_other_class']]
        dataset_tbl = pd.concat([trainset_tbl, valtestset_tbl], axis=0)
        aum_tbl = aum_tbl.merge(dataset_tbl,
                                on=['target_id', 'tce_plnt_num'],
                                how='left',
                                validate='one_to_one')
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
        aum_tbl.loc[aum_tbl['label_changed_to_other_class'], 'label'] = noise_label

    if not inj_thr:
        # get indexes for thresholded examples in the training set
        idxs_mislabeled = (aum_tbl['label'] == noise_label) & (aum_tbl['dataset'] == 'train')
    else:
        # get indexes for noise injected examples in the training set
        idxs_mislabeled = (aum_tbl['label_changed_to_other_class']) & (aum_tbl['dataset'] == 'train')
    # get AUM values for those examples
    aum_thr_examples = aum_tbl.loc[idxs_mislabeled, f'epoch_{epoch_chosen}']

    # compute threshold using AUM values for those thresholded/noise injected examples
    if len(aum_thr_examples) == 0:
        aum_thr = np.nan
    else:
        aum_thr = np.percentile(aum_thr_examples, n_percentile)
    aum_tbl[f'aum_thr_{n_percentile}'] = aum_thr

    # set examples with AUM lower than threshold to mislabeled that are not included in the subset used to compute the
    # threshold
    aum_tbl['mislabeled_by_aum'] = 'no'
    aum_tbl.loc[(aum_tbl[f'epoch_{epoch_chosen}'] < aum_thr) & ~idxs_mislabeled, 'mislabeled_by_aum'] = 'yes'

    aum_tbl.to_csv(run_dir / f'aum_epoch{epoch_chosen}_mislabeled.csv', index=False)

    aum_tbls.append(aum_tbl)

# count number of times each example was determined as mislabeled across runs
aum_cnts = aum_tbls[0][['target_id', 'tce_plnt_num', 'original_label', 'dataset', 'shard_name']]
aum_cnts['counts_mislabeled'] = 0
for aum_tbl in aum_tbls:
    aum_cnts.loc[aum_tbl['mislabeled_by_aum'] == 'yes', 'counts_mislabeled'] += 1
aum_cnts.to_csv(experiment_dir / 'aum_mislabeled_cnts.csv', index=False)

print(aum_cnts['counts_mislabeled'].value_counts().sort_index())
print(aum_cnts.loc[aum_cnts['counts_mislabeled'] > 0, 'counts_mislabeled'].describe())

# plot histogram of counts per label
bins = np.linspace(0, 10, 11, endpoint=True, dtype='int')
for label in labels:
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
    plt.close()

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
    plt.close()

# %% plot histogram of injected noise examples vs regular examples vs mislabeled examples

bins = np.linspace(-5, 5, 100)

# # plot histogram of injected noise examples vs regular examples across all runs
# for dataset in ['train', 'val', 'test']:
#     aum_dataset = aum_allruns_tbl.loc[
#         (aum_allruns_tbl['dataset'] == dataset) & (aum_allruns_tbl['original_label'].isin(['PC', 'AFP']))]
#
#     f, ax = plt.subplots()
#     ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'], 'mean'], bins, edgecolor='k',
#             label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
#     ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'] == False, 'mean'], bins, edgecolor='k',
#             label='Regular examples', zorder=1)
#     ax.set_xlabel('AUM')
#     ax.set_ylabel('Counts')
#     ax.set_yscale('log')
#     ax.legend()
#     f.savefig(experiment_dir / f'hist_aum_inj_vs_normal_samples_{dataset}.png')
#     plt.close()

# plot histogram of injected noise examples vs regular examples vs mislabeled examples for each run
n_runs = 9
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl['label_changed_to_other_class'] = False

    # adding column for examples injected with label noise
    trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                               f'trainset_run{run}.csv')[['target_id',
                                                          'tce_plnt_num',
                                                          'label_changed_to_other_class']]
    valtestset_tbl = pd.read_csv(experiment_dir / 'val_test_sets_labels_switched.csv')[['target_id',
                                                                                        'tce_plnt_num',
                                                                                        'label_changed_to_other_class']]
    dataset_tbl = pd.concat([trainset_tbl, valtestset_tbl], axis=0)
    aum_tbl = aum_tbl.merge(dataset_tbl,
                            on=['target_id', 'tce_plnt_num'],
                            how='left',
                            validate='one_to_one')
    aum_tbl.loc[aum_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False

    for dataset in ['train', 'val', 'test']:
        aum_dataset = aum_tbl.loc[(aum_tbl['dataset'] == dataset) & (aum_tbl['original_label'].isin(['PC', 'AFP']))]

        f, ax = plt.subplots(figsize=(16, 10))
        ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'], f'epoch_{epoch_chosen}'], bins,
                edgecolor='k',
                label='Stochastic injected label noise examples', zorder=3, alpha=0.5)
        ax.hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] != 'MISLABELED'), f'epoch_{epoch_chosen}'], bins, edgecolor='k',
                label='Regular examples',
                zorder=1)
        ax.hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] == 'MISLABELED'), f'epoch_{epoch_chosen}'], bins, edgecolor='k',
                label='Thr. examples',
                zorder=2, alpha=0.3)
        ax.set_xlabel('AUM')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.legend()
        f.tight_layout()
        f.savefig(run_dir / f'hist_aum_inj_vs_normal_vs_thr_samples_{dataset}.png')
        plt.close()

        # breakdown for PC and AFP categories
        f, ax = plt.subplots(2, 1, figsize=(10, 14))
        ax[0].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['original_label'] == 'PC'), f'epoch_{epoch_chosen}'], bins, edgecolor='k',
                   label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
        ax[0].hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] != 'MISLABELED') & (aum_dataset['original_label'] == 'PC'),
                                   f'epoch_{epoch_chosen}'], bins,
                   edgecolor='k', label='Regular examples', zorder=1)
        ax[0].hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] == 'MISLABELED') & (aum_dataset['original_label'] == 'PC'),
                                   f'epoch_{epoch_chosen}'], bins,
                   edgecolor='k', label='Thr. examples', zorder=3, alpha=0.5)
        ax[1].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['original_label'] == 'AFP'), f'epoch_{epoch_chosen}'], bins, edgecolor='k',
                   label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
        ax[1].hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] != 'MISLABELED') & (aum_dataset['original_label'] == 'AFP'),
                                   f'epoch_{epoch_chosen}'], bins,
                   edgecolor='k', label='Regular examples', zorder=1)
        ax[1].hist(aum_dataset.loc[(~aum_dataset['label_changed_to_other_class']) & (
                aum_dataset['label'] == 'MISLABELED') & (aum_dataset['original_label'] == 'AFP'),
                                   f'epoch_{epoch_chosen}'], bins,
                   edgecolor='k', label='Thr. examples', zorder=3, alpha=0.5)
        ax[1].set_xlabel('AUM')
        ax[0].set_ylabel('Counts')
        ax[1].set_ylabel('Counts')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[0].set_title('PCs')
        ax[1].set_title('AFPs')
        ax[1].set_yscale('log')
        ax[1].legend()
        f.savefig(run_dir / f'hist_aum_inj_vs_normal_vs_thr_samples_{dataset}_pc_afp.png')
        plt.close()

# %% plot score distribution for different categories and injected label noise examples and add column to rankings for each run

datasets = ['train', 'val', 'test']
n_runs = 9
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    for dataset in datasets:

        ranking_tbl = pd.read_csv(run_dir / 'models' / 'model1' / f'ranking_{dataset}.csv')

        if dataset == 'train':
            # adding column for examples injected with label noise
            injected_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                                       f'trainset_run{run}.csv')[['target_id',
                                                                  'tce_plnt_num',
                                                                  'label_changed_to_other_class']]
        else:
            injected_tbl = pd.read_csv(experiment_dir /
                                       'val_test_sets_labels_switched.csv')[['target_id',
                                                                             'tce_plnt_num',
                                                                             'label_changed_to_other_class']]

        injected_tbl.loc[injected_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
        ranking_tbl = ranking_tbl.merge(injected_tbl,
                                        on=['target_id', 'tce_plnt_num'],
                                        how='left',
                                        validate='one_to_one')

        ranking_tbl.to_csv(run_dir / 'models' / 'model1' / f'ranking_{dataset}_inj.csv', index=False)

        ranking_tbl.loc[(ranking_tbl['label_changed_to_other_class']) & (ranking_tbl['original_label'] == 'AFP'),
                        'original_label'] = 'PC_to_AFP'
        ranking_tbl.loc[(ranking_tbl['label_changed_to_other_class']) & (ranking_tbl['original_label'] == 'PC'),
                        'original_label'] = 'AFP_to_PC'
        output_cl = {label: None for label in ranking_tbl['original_label'].unique()}
        for label in output_cl:
            if label == 'PC_to_AFP':
                output_cl[label] = ranking_tbl.loc[ranking_tbl['original_label'] == 'PC', 'score_PC'].values
            elif label == 'AFP_to_PC':
                output_cl[label] = ranking_tbl.loc[ranking_tbl['original_label'] == 'AFP', 'score_AFP'].values
            else:
                output_cl[label] = ranking_tbl.loc[ranking_tbl['original_label'] == label, f'score_{label}'].values

        plot_class_distribution(output_cl,
                                run_dir /
                                f'class_predoutput_distribution_{dataset}_inj.svg')

# %% compute performance on injected labeling noise

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-10-2022_0950')
n_runs = 9
n_epochs = 50
inj_thr = True
noise_label = 'MISLABELED'

perf_metrics_runs = []
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl['label_changed_to_other_class'] = False

    # adding column for examples injected with label noise
    trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                               f'trainset_run{run}.csv')[['target_id',
                                                          'tce_plnt_num',
                                                          'label_changed_to_other_class']]
    valtestset_tbl = pd.read_csv(experiment_dir / 'val_test_sets_labels_switched.csv')[['target_id',
                                                                                        'tce_plnt_num',
                                                                                        'label_changed_to_other_class']]
    dataset_tbl = pd.concat([trainset_tbl, valtestset_tbl], axis=0)
    dataset_tbl.loc[dataset_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
    aum_tbl = aum_tbl.merge(dataset_tbl,
                            on=['target_id', 'tce_plnt_num'],
                            how='left',
                            validate='one_to_one')
    aum_tbl.loc[aum_tbl['label_changed_to_other_class'], 'label'] = 'INJ'  # noise_label

    # get aum threshold
    if not inj_thr:
        # using thresholded examples
        aum_thr_examples = aum_tbl.loc[(aum_tbl['label'] == noise_label) & (aum_tbl['dataset'] == 'train'),
                                       f'epoch_{epoch_chosen}']
    else:
        # using injected label noise examples
        aum_thr_examples = aum_tbl.loc[(aum_tbl['label_changed_to_other_class']) &
                                       (aum_tbl['dataset'] == 'train'), f'epoch_{epoch_chosen}']

    # computer threshold as nth-percentile
    aum_thr = np.percentile(aum_thr_examples, n_percentile)

    aum_tbl['mislabeled_by_aum'] = 'no'
    # set examples with AUM lower than threshold to mislabeled
    aum_tbl.loc[aum_tbl[f'epoch_{epoch_chosen}'] < aum_thr, 'mislabeled_by_aum'] = 'yes'

    # compute precision and recall for detecting injected label noise
    perf_metrics = {}
    # train set
    aum_tbl_train = aum_tbl.loc[aum_tbl['dataset'] == 'train']
    aum_tbl_train = aum_tbl_train.loc[aum_tbl_train['label'] != noise_label]  # don't count thresholded examples
    perf_metrics['train_recall'] = ((aum_tbl_train['mislabeled_by_aum'] == 'yes') & (
                aum_tbl_train['label'] == 'INJ')).sum() / (aum_tbl_train['label'] == 'INJ').sum()
    perf_metrics['train_precision'] = ((aum_tbl_train['mislabeled_by_aum'] == 'yes') & (
                aum_tbl_train['label'] == 'INJ')).sum() / (aum_tbl_train['mislabeled_by_aum'] == 'yes').sum()
    # val and test sets
    aum_tbl_train = aum_tbl.loc[aum_tbl['dataset'].isin(['val', 'test'])]
    perf_metrics['test_recall'] = ((aum_tbl_train['mislabeled_by_aum'] == 'yes') & (
                aum_tbl_train['label'] == 'INJ')).sum() / (aum_tbl_train['label'] == 'INJ').sum()
    perf_metrics['test_precision'] = ((aum_tbl_train['mislabeled_by_aum'] == 'yes') & (
                aum_tbl_train['label'] == 'INJ')).sum() / (aum_tbl_train['mislabeled_by_aum'] == 'yes').sum()

    perf_metrics_s = pd.Series(perf_metrics)
    perf_metrics_runs.append(perf_metrics_s)
    perf_metrics_s.to_csv(run_dir / 'perf_metrics_inj.csv')

perf_metrics_runs = pd.concat(perf_metrics_runs, axis=1)

std_perf_metrics = perf_metrics_runs.std(axis=1)
perf_metrics_runs['mean'] = perf_metrics_runs.mean(axis=1)
perf_metrics_runs['std'] = perf_metrics_runs.std(axis=1)

perf_metrics_runs.to_csv(experiment_dir / 'perf_metrics_inj.csv')


# %% Compare rankings across runs to study model's sensitivity to choice of thresholded examples


def overlap(r1, r2, d_i):
    """ Compute overlap between rankings `r1` and `r2` at depth `d`, i.e., the number of common elements that show up in
    both rankings up to depth `d`.

    :param r1: NumPy array, ranking 1
    :param r2: NumPy array, ranking 2
    :param d_i: int, depth
    :return:
        float, overlap
    """
    return len(np.intersect1d(r1[:d_i], r2[:d_i])) / d_i


# def rbo(r1, r2, d, p):
#
#     return (1 - p) * np.sum([p ** (d_i - 1) * overlap(r1, r2, d_i) for d_i in range(1, d + 1)])


def avg_overlap(r1, r2, d):
    """ Compute average overlap between rankings `r1` and `r2` at a cut-off depth `d`.

    :param r1: NumPy array, ranking 1
    :param r2: NumPy array, ranking 2
    :param d: int, cut-off depth
    :return:
        float, average overlap
    """

    return 1 / d * np.sum([overlap(r1, r2, d_i) for d_i in range(1, d + 1)])


experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-17-2022_1532')

n_runs = 10
noise_label = 'MISLABELED'
datasets = ['train', 'val_test']
depth = 30
rbo_p = 0.8
kendall_tau_params = {'rank': False, 'weigher': None, }

for dataset in datasets:  # iterate through the datasets

    print(f'Iterating over dataset {dataset}...')

    aum_rankings = {f'run_{run}': None for run in range(n_runs)}
    ranking_combs = list(combinations(list(aum_rankings.keys()), 2))
    avg_op = np.nan * np.ones((n_runs, n_runs))  # {ranking_comb: np.nan for ranking_comb in ranking_combs}
    rbo_metric = np.nan * np.ones((n_runs, n_runs))  # {ranking_comb: np.nan for ranking_comb in ranking_combs}
    kendall_tau_mat = np.nan * np.ones((n_runs, n_runs))  # {ranking_comb: np.nan for ranking_comb in ranking_combs}

    with(open(experiment_dir / 'dataset_params.yaml', 'r')) as file:
        dataset_params = yaml.safe_load(file)
    dataset_tbl = pd.read_csv(Path(dataset_params['src_tfrec_dir']) / 'tfrec_tbl.csv')
    dataset_tbl['thresholded_examples'] = False
    dataset_tbl['id'] = dataset_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                  x['tce_plnt_num']),
                                                                         axis=1)

    # get AUM rankings for each run
    for run in range(n_runs):
        print(f'[{dataset}] Iterating over run {run}...')
        aum_rankings[f'run_{run}'] = \
            pd.read_csv(experiment_dir /
                        'runs' /
                        f'run{run}' /
                        'aum_mislabeled.csv')[['target_id',
                                               'tce_plnt_num',
                                               'label',
                                               'dataset',
                                               f'epoch_{epoch_chosen}']].sort_values(by='margin',
                                                                                     axis=0,
                                                                                     ascending=True)
        # select dataset
        if dataset == 'train':
            aum_rankings[f'run_{run}'] = \
                aum_rankings[f'run_{run}'].loc[aum_rankings[f'run_{run}']['dataset'] == dataset]
        else:
            aum_rankings[f'run_{run}'] = \
                aum_rankings[f'run_{run}'].loc[aum_rankings[f'run_{run}']['dataset'].isin(['val', 'test'])]

        # exclude examples that are not PC nor AFP
        aum_rankings[f'run_{run}'] = \
            aum_rankings[f'run_{run}'].loc[aum_rankings[f'run_{run}']['label'].isin(['PC', 'AFP', noise_label])]
        # # remove thresholded examples
        # aum_rankings[f'run_{run}'] = aum_rankings[f'run_{run}'].loc[aum_rankings[f'run_{run}']['label'] != noise_label]

        # set examples' ID
        aum_rankings[f'run_{run}']['id'] = \
            aum_rankings[f'run_{run}'][['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                     x['tce_plnt_num']),
                                                                            axis=1)

        # add indicator for thresholded examples across all runs to dataset table
        aum_ranking_mis = aum_rankings[f'run_{run}'].loc[aum_rankings[f'run_{run}']['label'] == noise_label]
        dataset_tbl.loc[dataset_tbl['id'].isin(aum_ranking_mis['id']), 'thresholded_examples'] = True

    # exclude from computation examples that were used for thresholding in at least one run
    dataset_tbl_mis = dataset_tbl.loc[dataset_tbl['thresholded_examples']]
    for run in range(n_runs):
        aum_rankings[f'run_{run}'] = \
            aum_rankings[f'run_{run}'].loc[~aum_rankings[f'run_{run}']['id'].isin(dataset_tbl_mis['id'])]

    # compute similarity score between each ranking pair
    print('Computing ranking similarities...')
    for ranking_1, ranking_2 in ranking_combs:
        # ranking_sims[(ranking_1, ranking_2)] = avg_overlap(aum_rankings[ranking_1]['id'].values,
        #                                                    aum_rankings[ranking_2]['id'].values,
        #                                                    depth)
        run_i_1, run_i_2 = int(ranking_1.split('_')[1]), int(ranking_2.split('_')[1])
        avg_op[run_i_1, run_i_2] = avg_overlap(aum_rankings[ranking_1]['id'].values,
                                               aum_rankings[ranking_2]['id'].values,
                                               depth)
        avg_op[run_i_2, run_i_1] = avg_op[run_i_1, run_i_2]
        rbo_metric[run_i_1, run_i_2] = rbo.RankingSimilarity(aum_rankings[ranking_1]['id'].values,
                                                             aum_rankings[ranking_2]['id'].values).rbo(k=depth,
                                                                                                       p=rbo_p)
        rbo_metric[run_i_2, run_i_1] = rbo_metric[run_i_1, run_i_2]

        ranking1_to_int = aum_rankings[ranking_1][['id']].copy(deep=True)
        ranking1_to_int['rank'] = np.arange(len(ranking1_to_int))
        ranking2_to_int = aum_rankings[ranking_2][['id']].copy(deep=True)
        ranking2_to_int = ranking2_to_int.merge(ranking1_to_int, on=['id'], how='left', validate='one_to_one')
        kendall_tau_mat[run_i_1, run_i_2], _ = weightedtau(ranking1_to_int['rank'].values[:depth],
                                                           ranking2_to_int['rank'].values[:depth],
                                                           **kendall_tau_params)
        kendall_tau_mat[run_i_2, run_i_1] = kendall_tau_mat[run_i_1, run_i_2]

    avg_op_df = pd.DataFrame(avg_op, columns=list(aum_rankings.keys()), index=list(aum_rankings.keys()))
    avg_op_df_mean = avg_op_df.mean(axis=0, skipna=True)
    avg_op_df_std = avg_op_df.std(axis=0, ddof=1, skipna=True)
    avg_op_df['mean'] = avg_op_df_mean
    avg_op_df['std'] = avg_op_df_std
    # print(f'{dataset}: \n{avg_op_df}')
    avg_op_df.to_csv(experiment_dir / f'average_overlap_depth{depth}_{dataset}.csv')

    rbo_df = pd.DataFrame(rbo_metric, columns=list(aum_rankings.keys()), index=list(aum_rankings.keys()))
    rbo_df_mean = rbo_df.mean(axis=0, skipna=True)
    rbo_df_std = rbo_df.std(axis=0, ddof=1, skipna=True)
    rbo_df['mean'] = rbo_df_mean
    rbo_df['std'] = rbo_df_std
    # print(f'{dataset}: \n{rbo_df}')
    rbo_df.to_csv(experiment_dir / f'rbo_p{rbo_p}_depth{depth}_{dataset}.csv')

    kendalltau_df = pd.DataFrame(kendall_tau_mat, columns=list(aum_rankings.keys()), index=list(aum_rankings.keys()))
    kendalltau_df_mean = kendalltau_df.mean(axis=0, skipna=True)
    kendalltau_df_std = kendalltau_df.std(axis=0, ddof=1, skipna=True)
    kendalltau_df['mean'] = kendalltau_df_mean
    kendalltau_df['std'] = kendalltau_df_std
    # print(f'{dataset}: \n{kendalltau_df}')
    kendalltau_df.to_csv(experiment_dir / f'weighted_kendalltau_depth{depth}_{dataset}.csv')

# %% plot logits, margins and AUMs over epochs for some examples

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-14-2022_1155')
runs_dir = experiment_dir / 'runs'
n_runs = len([fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()])
n_epochs = 500
inj_thr = False
noise_label = 'MISLABELED'
tfrec_tbl_features = {
    'target_id': {'dtype': 'int64'},
    'tce_plnt_num': {'dtype': 'int64'},
    'label': {'dtype': 'str'},
    # 'original_label': {'dtype': 'str'},
    # 'label_id': {'dtype': 'int64'}
}
datasets = ['train', 'predict']
cols_tbl = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label', 'margin']
epoch_chosen = 499
# for run in range(n_runs):
#
#     run_dir = experiment_dir / 'runs' / f'run{run}'
#
#     examples_dir = run_dir / 'examples'
#     examples_dir.mkdir(exist_ok=True)
#
#     model_dir = run_dir / 'models' / 'model1'
#
#     aum_tbl = pd.read_csv(model_dir / 'aum.csv')
#     aum_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
#
#     # create TFRecord dataset table
#     tfrec_dir = experiment_dir / 'tfrecords' / f'run{run}'
#     tfrec_fps = [fp for fp in tfrec_dir.iterdir() if 'shard' in fp.name]
#
#     tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=tfrec_tbl_features)
#
#     ranking_tbls = []
#     for dataset in datasets:
#         ranking_tbls.append(pd.read_csv(model_dir / f'ranking_{dataset}.csv'))
#         ranking_tbls[-1]['dataset'] = dataset
#     ranking_tbl = pd.concat(ranking_tbls, axis=0)
#     ranking_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
#     ranking_tbl.to_csv(model_dir / 'ranking_alldatasets.csv')
#
#     logits_tbls = {logit_i: pd.concat([pd.read_csv(model_dir / f'logits_epoch-{epoch_i}.csv',
#                                                    usecols=[logit_i],
#                                                    names=[f'epoch_{epoch_i}'],
#                                                    header=0)
#                                        for epoch_i in range(n_epochs)], axis=1) for logit_i in [0, 1]
#                    }
#
#     logits_tbls = {logit_i: tfrec_tbl.merge(logit_tbl, how='left',
#                                             left_index=True,
#                                             right_index=True,
#                                             validate='one_to_one')
#                    for logit_i, logit_tbl in logits_tbls.items()}
#     for logit_i, logits_tbl in logits_tbls.items():
#         logits_tbl.to_csv(model_dir / f'logit{logit_i}_allepochs.csv', index=False)
#
#     # get margins over epochs for all examples
#     margins_dir = model_dir / 'margins'
#     margins_tbl = []
#     for epoch_i in range(n_epochs):
#         margins_tbl.append(pd.read_csv(margins_dir / f'margins_epoch-{epoch_i}.csv',
#                                        usecols=cols_tbl,
#                                        )
#                            if epoch_i == 0 else pd.read_csv(margins_dir / f'margins_epoch-{epoch_i}.csv',
#                                                             usecols=['margin'])
#                            )
#         margins_tbl[-1] = margins_tbl[-1].rename(columns={'margin': f'epoch_{epoch_i}'})
#     margins_tbl = pd.concat(margins_tbl, axis=1)
#     margins_tbl.to_csv(margins_dir / 'margins_allepochs.csv', index=False)

# aum_tbl = pd.read_csv(experiment_dir / 'runs/run0/models/model1/aum.csv')
# aum_tbl = aum_tbl.loc[aum_tbl['dataset'] == 'train']
# aum_tbl.sort_values(by='margin', ascending=True, inplace=True)
# n_examples_per_cat = 20
# examples_chosen = pd.concat([aum_tbl.loc[aum_tbl['label'] == 'AFP'][:n_examples_per_cat],
#                              aum_tbl.loc[aum_tbl['label'] == 'PC'][:n_examples_per_cat]])
# examples = {tuple(example): {'label': f'top-{n_examples_per_cat} AUM for its category'}
#             for example in examples_chosen[['target_id', 'tce_plnt_num']].to_numpy()}
# aum_ranks_top_cnts = pd.read_csv(
#     '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-24-2022_1044/aum_ranks_allruns_cnts_top_30.csv')
# aum_ranks_top_cnts = aum_ranks_top_cnts.sort_values(by='top_30_cnts', ascending=False)
# aum_ranks_top_cnts = aum_ranks_top_cnts.loc[aum_ranks_top_cnts['top_30_cnts'] >= 6]
# examples_chosen = aum_ranks_top_cnts.copy(deep=True)
# examples = {tuple(example): {'label': f'top-30 cnts >= 6 AUM for its category'}
#             for example in examples_chosen[['target_id', 'tce_plnt_num']].to_numpy()}
# examples = {(6699368, 1): {'label': 'low AUM, but high score'}}
# examples = {
#     # (11030475, 1): {'label': 'mislabeled PC lowest AUM'},  # mislabeled PC lowest AUM
#     # (10227863, 1): {'label': 'mislabeled AFP low AUM'},  # mislabeled AFP low AUM
#     # (10904857, 1): {'label': 'PC with low AUM'},  # PC with low AUM
#     # (4769931, 1): {'label': 'AFP with low AUM'},  # AFP with low AUM
#     # (9777087, 1): {'label': 'AFP highest AUM'},  # AFP highest AUM
#     # (7907423, 2): {'label': 'PC high AUM'},  # PC high AUM
#     (6058896, 1): {'label': 'AFP highest AUM'},
#     (5868793, 1): {'label': 'PC highest AUM'},
#     (7661065, 1): {'label': 'AFP lowest AUM'},
#     (7532973, 1): {'label': 'PC lowest AUM'},
#
# }
# demoted confirmed planets
examples = {
    # (7532973, 1): {'label': 'Kepler-854 b'},
    # (11517719, 1): {'label': 'Kepler-840  b'},
    # (6061119, 1): {'label': 'Kepler-699 b'},
    # (5780460, 1): {'label': 'Kepler-747 b'}
    # easy examples with high AUM
    (6058896, 1): {'label': 'KIC 6058896.1 (AFP)'},
    (5868793, 1): {'label': 'KIC 5868793.1 (PC)'}
}

# plot margins, AUMs for each example for all runs
examples_dir = experiment_dir / 'examples_easy'
examples_dir.mkdir(exist_ok=True)
# aum_tbl_allruns = pd.read_csv(experiment_dir / f'aum_allruns_epoch{epoch_chosen}.csv')
# aum_tbl_allruns.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
nprocesses = 12


def plot_logits_margins_aum_example_all_runs(example, n_runs, n_epochs, experiment_dir, examples_dir):
    """ Plot logits, margins and AUM values over epochs for all runs for each example.

    :param example: tuple (kic, tce_plnt_num), TCE ID
    :param n_runs: int, number of runs
    :param n_epochs: int, number of training epochs
    :param experiment_dir: Path, experiment directory
    :param examples_dir: Path, directory to save results
    :return:
    """

    plt.ioff()
    print(f'Plotting for example {example[0]}.{example[1]}...')

    aum_tbl_allruns = pd.read_csv(experiment_dir / f'aum_allruns_epoch{n_epochs - 1}.csv')
    aum_tbl_allruns.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

    score_og_label = []

    f, ax = plt.subplots(3, 1, figsize=(8, 8))

    for run in range(n_runs):

        print(f'[{example[0]}.{example[1]}] Run {run}')

        run_dir = experiment_dir / 'runs' / f'run{run}'
        model_dir = run_dir / 'models' / 'model1'

        # get AUM for last training epoch
        aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
        aum_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        ranking_tbl = pd.read_csv(model_dir / 'ranking_alldatasets.csv')
        ranking_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        # grab model score for this example
        score_og_label.append(ranking_tbl.loc[example, f'score_{aum_tbl.loc[example, "original_label"]}'])

        # get logits over epochs tables
        logits_tbls = {logit_i: pd.read_csv(model_dir / f'logit{logit_i}_allepochs.csv') for logit_i in [0, 1]}
        for logit_i in logits_tbls:
            logits_tbls[logit_i] = logits_tbls[logit_i].set_index(keys=['target_id', 'tce_plnt_num'])

        # get margins over epochs table
        margins_dir = model_dir / 'margins'
        margins_tbl = pd.read_csv(margins_dir / 'margins_allepochs.csv')
        margins_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        # plot logits
        for logit_i in logits_tbls:
            ax[0].plot(np.arange(n_epochs),
                       logits_tbls[logit_i].loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]],
                       label=f'Logit {logit_i}' if run == 0 else None,
                       color='orange' if logit_i == 1 else 'b')
        # plot margins
        ax[1].plot(np.arange(n_epochs),
                   margins_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]])
        # plot AUM values
        ax[2].plot(np.arange(n_epochs),
                   np.cumsum(margins_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].values) /
                   np.arange(1, n_epochs + 1))
    ax[2].set_xlabel('Epoch Number')
    ax[0].set_ylabel('Logit Value')
    ax[1].set_ylabel('Margin')
    ax[2].set_ylabel('AUM')
    ax[0].set_title(f'Example {example} {aum_tbl_allruns.loc[example, "original_label"]}\n'
                    f'{aum_tbl_allruns.loc[example, "dataset"]} dataset | '
                    f'Score {np.mean(score_og_label):.4f}+-{np.std(score_og_label, ddof=1):.4f} | '
                    # f'Predicted class {ranking_tbl.loc[example, "predicted class"]}\n'
                    f'{examples[example]["label"]}\n '
                    f'AUM: {aum_tbl_allruns.loc[example, "mean"]:.4f}+-{aum_tbl_allruns.loc[example, "std"]:.4f}')
    ax[0].legend()
    f.savefig(examples_dir /
              f'{example[0]}.{example[1]}-{aum_tbl_allruns.loc[example, "original_label"]}_logits_margin.png')
    # plt.close()


# for example in examples:

pool = multiprocessing.Pool(processes=nprocesses)
jobs = [(example, n_runs, n_epochs, experiment_dir, examples_dir)
        for example in examples]
async_results = [pool.apply_async(plot_logits_margins_aum_example_all_runs, job) for job in jobs]
pool.close()
for async_result in async_results:
    async_result.get()

# %% Count number of times each example is in top-K AUM ranking across runs to study method's sensitivity.

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')
epoch_chosen = 499
runs_dir = experiment_dir / 'runs'
n_runs = len([fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()])
aum_rank = None
cols_of_interest = [
    'target_id',
    'tce_plnt_num',
    'original_label',
    'dataset'
]
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl = aum_tbl.loc[aum_tbl['dataset'] == 'train']
    # aum_tbl = aum_tbl.sort_values(by='margin', ascending=True).reset_index(drop=True)
    aum_tbl = aum_tbl.sort_values(by=f'epoch_{epoch_chosen}', ascending=True).reset_index(drop=True)
    aum_tbl[f'rank_{run_dir.name}'] = aum_tbl.index

    if aum_rank is None:
        aum_rank = aum_tbl[cols_of_interest + [f'rank_{run_dir.name}']].copy(deep=True)
    else:
        aum_rank = aum_rank.merge(aum_tbl[['target_id', 'tce_plnt_num', f'rank_{run_dir.name}']],
                                  on=['target_id', 'tce_plnt_num'],
                                  validate='one_to_one')

aum_rank.to_csv(experiment_dir / f'aum_ranks_allruns_epoch{epoch_chosen}.csv', index=False)


# %%


def count_on_top_k(example, top_k):
    """ Count number of times example is in top-k across runs.

    :param example: pandas Series, rankings for example across runs
    :param top_k: int, top-k
    :return:
        int, number of times examples is in top-k across runs.
    """

    return (example < top_k).sum()


top_k = 30

aum_rank = pd.read_csv(experiment_dir / f'aum_ranks_allruns_epoch{epoch_chosen}.csv')
aum_rank[f'top_{top_k}_cnts'] = aum_rank[[f'rank_run{run}' for run in range(n_runs)]].apply(count_on_top_k,
                                                                                            args=(top_k,), axis=1)
aum_rank.to_csv(experiment_dir / f'aum_ranks_allruns_epoch{epoch_chosen}_cnts_top_{top_k}.csv', index=False)

bins = np.arange(0, n_runs + 2)
f, ax = plt.subplots()
ax.hist(aum_rank[f'top_{top_k}_cnts'], bins, edgecolor='k')
ax.set_ylabel(f'Counts')
ax.set_xlabel(f'Number of runs example is in top-{top_k}')
ax.set_yscale('log')
ax.set_xticks(bins + 0.5)
ax.set_xticklabels(bins)
ax.set_xlim([bins[0], bins[-1]])
ax.grid(axis='y')
f.savefig(experiment_dir / f'hist_top_{top_k}.png')
plt.close()

# %% Study changes in margin as proxy to training dynamics reaching an equilibrium

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')

n_epochs = 500
n_runs = 10
n_examples = 34030
fixed_cols = ['target_id', 'tce_plnt_num', 'label', 'label_id', 'original_label', 'shard_name', 'dataset']

margins_arr = np.nan * np.ones((n_runs, n_examples, n_epochs), dtype='float')
margins_tbl = None
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    margin_tbl_run = pd.read_csv(run_dir / 'models' / 'model1' / 'margins' / 'margins_allepochs.csv')
    if margins_tbl is None:
        margins_tbl = margin_tbl_run[fixed_cols]
    margins_arr[run, :, :] = margin_tbl_run[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()

# aggregate margins from all runs
margins_agg = np.median(margins_arr, axis=0)
margins_agg_tbl = pd.concat([
    margins_tbl,
    pd.DataFrame(data=margins_agg, columns=[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)])
],
    axis=1, ignore_index=False)
margins_agg_tbl.to_csv(experiment_dir / f'margins_agg_allruns.csv', index=False)

# %%

margins_agg_tbl = pd.read_csv(experiment_dir / f'margins_agg_allruns.csv')

window = 10
for example_i, example in margins_agg_tbl.iterrows():
    margins_agg_tbl.loc[example_i, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]] = \
        np.convolve(margins_agg_tbl.loc[example_i, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]],
                    np.ones(window) / window, mode='same')
margins_agg_tbl.to_csv(experiment_dir / f'margins_agg_allruns_avgsmooth_win{window}.csv', index=False)
margins_agg = margins_agg_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()

margins_agg_diff_rel = np.diff(margins_agg, axis=1, prepend=np.nan) / margins_agg
margins_agg_diff_rel_tbl = pd.concat([
    margins_agg_tbl[fixed_cols],
    pd.DataFrame(data=margins_agg_diff_rel, columns=[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)])
],
    axis=1, ignore_index=False)
# margins_agg_diff_rel_tbl.to_csv(experiment_dir / f'margins_agg_diff_rel_allruns.csv', index=False)
margins_agg_diff_rel_tbl.to_csv(experiment_dir / f'margins_agg_avgsmooth_win{window}_diff_rel_allruns.csv', index=False)

margins_agg_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
margins_agg_diff_rel_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
# %%


example = (7767559, 1)
window = 20
f, ax = plt.subplots()
# ax.plot(np.arange(n_epochs), margins_agg_diff_rel_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]])
ax.plot(np.arange(n_epochs)[window - 1:],
        np.convolve(margins_agg_diff_rel_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]],
                    np.ones(window) / window, mode='valid'))
# ax.plot(np.arange(n_epochs), margins_agg_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]])
# ax.plot(np.arange(n_epochs)[window-1:], np.convolve(margins_agg_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]], np.ones(window) / window, mode='valid'))
ax.set_title(f'KIC {example[0]}.{example[1]}')

# %%

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')

margins_agg_diff_rel_tbl = pd.read_csv(experiment_dir / f'margins_agg_diff_rel_allruns.csv').set_index(
    keys=['target_id', 'tce_plnt_num'])
margins_agg_tbl = pd.read_csv(experiment_dir / f'margins_agg_allruns.csv').set_index(keys=['target_id', 'tce_plnt_num'])

n_epochs = 500
margin_metric = margins_agg_diff_rel_tbl.loc[
    margins_agg_diff_rel_tbl['dataset'] == 'train', [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()
n_percentile = 50
margins_agg_diff_rel_nper = np.percentile(margin_metric, n_percentile, axis=0)  # unsmoothed
window = 10
# moving average smoothed
margins_agg_diff_rel_nper = np.percentile(
    [np.convolve(vals_example, np.ones(window) / window, mode='valid') for vals_example in margin_metric], n_percentile,
    axis=0)
f, ax = plt.subplots()
ax.plot(np.arange(n_epochs)[window - 1:], margins_agg_diff_rel_nper)
ax.set_title(f'{n_percentile}th-percentile')
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Margin delta')

examples = {
    (7532973, 1): {'kepler_name': 'Kepler-854 b'},
    (11517719, 1): {'kepler_name': 'Kepler-840  b'},
    (6061119, 1): {'kepler_name': 'Kepler-699 b'},
    (5780460, 1): {'kepler_name': 'Kepler-747 b'}
}
f, ax = plt.subplots()
for n_percentile in []:  # [99, 90, 50]:
    ax.plot(np.arange(n_epochs)[window - 1:],
            np.percentile([np.convolve(vals_example, np.ones(window) / window, mode='valid')
                           for vals_example in margin_metric], n_percentile,
                          axis=0),
            label=f'{n_percentile}th-percentile')
for example in examples:
    # ax.plot(np.arange(n_epochs), margins_agg_diff_rel_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]], label=f'{examples[example]["kepler_name"]}')
    ax.plot(np.arange(n_epochs), margins_agg_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]],
            label=f'{examples[example]["kepler_name"]}')
ax.set_title(f'nth-percentile\nAveraging Window = {window}')
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Relative Margin 1st Order Difference')
ax.legend()
ax.set_xlim([window, n_epochs - 1])
ax.grid(axis='y')
f.savefig(experiment_dir / f'margin_diff_rel_npercentiles_avgsmooth_{window}.png')
# ax.set_yscale('log')

# %% Study changes in AUM as proxy to training dynamics reaching an equilibrium


experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_04-13-2022_1155')

n_epochs = 500
n_runs = 10
n_examples = 34030
fixed_cols = ['target_id', 'tce_plnt_num', 'label', 'label_id', 'original_label', 'shard_name', 'dataset']

# %%

# get AUM values over epochs across all runs
aum_arr = np.nan * np.ones((n_runs, n_examples, n_epochs), dtype='float')
aum_tbl = None
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl_run = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    if aum_tbl is None:
        aum_tbl = aum_tbl_run[fixed_cols]
    aum_arr[run, :, :] = aum_tbl_run[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()

# aggregate AUMs from all runs
aum_agg = np.median(aum_arr, axis=0)  # using median
aum_agg_tbl = pd.concat([
    aum_tbl,
    pd.DataFrame(data=aum_agg, columns=[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)])
],
    axis=1, ignore_index=False)
aum_agg_tbl.to_csv(experiment_dir / f'aum_agg_allruns.csv', index=False)
aum_agg_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

# compute relative 1st order difference
aum_agg_diff_rel = np.diff(aum_agg, axis=1, prepend=np.nan) / aum_agg
aum_agg_diff_rel_tbl = pd.concat([
    aum_agg_tbl[fixed_cols],
    pd.DataFrame(data=aum_agg_diff_rel, columns=[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)])
],
    axis=1, ignore_index=False)
aum_agg_diff_rel_tbl.to_csv(experiment_dir / f'aum_agg_diff_rel_allruns.csv', index=False)

# %%

aum_agg_diff_rel_tbl = pd.read_csv(experiment_dir / f'aum_agg_diff_rel_allruns.csv')
aum_agg_diff_rel_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

aum_metric = aum_agg_diff_rel_tbl.loc[
    aum_agg_diff_rel_tbl['dataset'] == 'train', [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()
# # compute percentiles
# aum_metric = aum_agg_diff_rel_tbl.loc[
#     aum_agg_diff_rel_tbl['dataset'] == 'train', [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].to_numpy()
# n_percentile = 50
# margins_agg_diff_rel_nper = np.percentile(margin_metric, n_percentile, axis=0)

# %%
examples = {
    # demoted confirmed planets
    # (7532973, 1): {'label': 'Kepler-854 b'},
    # (11517719, 1): {'label': 'Kepler-840  b'},
    # (6061119, 1): {'label': 'Kepler-699 b'},
    # (5780460, 1): {'label': 'Kepler-747 b'},
    # difficult cases
    (7661065, 1): {'label': 'KIC 7661065.1 (AFP)'},
    (6696462, 1): {'label': 'KIC 6696462.1 (AFP)'},
    (6061119, 1): {'label': 'KIC 6061119.1 Kepler-699 b'},
    (12004971, 1): {'label': 'KIC 12004971.1 (AFP)'},
    (10904857, 1): {'label': 'KIC 10904857.1 (Kepler-488 b)'},
    (7532973, 1): {'label': 'KIC 7532973.1 Kepler-854 b'},
    (7767559, 1): {'label': 'KIC 7767559.1 (AFP)'},

    # easy examples with high AUM
    # (6058896, 1): {'label': 'KIC 6058896.1 (AFP)'},
    # (5868793, 1): {'label': 'KIC 5868793.1 (PC)'}
}

n_percentiles = [
    99,
    90,
    50,
    1
]
# plot percentiles
f, ax = plt.subplots()
for n_percentile in n_percentiles:
    ax.plot(np.arange(n_epochs),
            np.percentile(np.abs(aum_metric), n_percentile, axis=0),
            label=f'{n_percentile}th-percentile',
            linestyle='dashed')
for example in examples:
    ax.plot(np.arange(n_epochs),
            np.abs(aum_agg_diff_rel_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]]),
            label=f'{examples[example]["label"]}')
#     ax.plot(np.arange(n_epochs), margins_agg_tbl.loc[example, [f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]], label=f'{examples[example]["kepler_name"]}')
ax.set_title(f'nth-percentile')
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Relative AUM 1st Order Difference')
ax.legend()
ax.set_xlim([0, n_epochs - 1])
ax.grid(axis='y')
ax.set_ylim([0, 0.1])
aa
f.savefig(experiment_dir / f'aum_diff_rel_npercentiles.png')

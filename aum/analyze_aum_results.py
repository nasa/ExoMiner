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

# local
from src.utils_visualization import plot_class_distribution

# %% aggregate AUM across runs in an experiment

experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-03-2022_1215')

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
    trainset_tbl.loc[trainset_tbl['label_changed_to_other_class'].isna()] = False
    for example_i, example in aum_allruns_tbl.iterrows():
        if aum_allruns_tbl.loc[example_i, 'dataset'] == 'train':
            # if trainset_tbl.loc[(example['target_id'], example['tce_plnt_num']), 'label_changed_to_mislabeled']:
            if trainset_tbl.loc[(example['target_id'], example['tce_plnt_num']), 'label_changed_to_other_class']:
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

# add column related to injected stochastic noise to the AFP and PC populations in the different datasets
cols = ['target_id', 'tce_plnt_num', 'label_changed_to_other_class']
# trainset_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-01-2022_1433/train_set_labels_switched.csv')[cols]
valtestset_tbl = pd.read_csv(experiment_dir / 'val_test_sets_labels_switched.csv')[cols]
dataset_tbl = valtestset_tbl  # pd.concat([trainset_tbl, valtestset_tbl], axis=0)
aum_allruns_tbl = aum_allruns_tbl.merge(dataset_tbl,
                                        on=['target_id', 'tce_plnt_num'],
                                        how='left',
                                        validate='one_to_one')

aum_allruns_tbl.to_csv(experiment_dir / 'aum.csv', index=False)

# %% Compute nth percentile

n_percentile = 99
noise_label = 'MISLABELED'

# experiment_dir = Path(
#     '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

runs_dir = experiment_dir / 'runs'
margin_thr = {}
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:
    print(f'Computing {n_percentile}-percentile margin threshold for run {run_dir}')

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')

    # margin_mislabeled = aum_tbl.loc[aum_tbl['label'] == noise_label, 'margin']

    trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                               f'trainset_{run_dir.name}.csv')[['target_id',
                                                                'tce_plnt_num',
                                                                'label_changed_to_other_class']]
    aum_tbl = aum_tbl.merge(trainset_tbl,
                            on=['target_id', 'tce_plnt_num'],
                            how='left',
                            validate='one_to_one')
    aum_tbl.loc[aum_tbl['label_changed_to_other_class'].isna(), 'label_changed_to_other_class'] = False
    margin_mislabeled = aum_tbl.loc[
        (aum_tbl['label_changed_to_other_class'] == True) & (aum_tbl['dataset'] == 'train'), 'margin']

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
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-03-2022_1215')
n_runs = 9
n_epochs = 50
labels = {
    'PC': {'color': 'b', 'zorder': 1, 'alpha': 1.0},
    'AFP': {'color': 'r', 'zorder': 2, 'alpha': 1.0},
    'MISLABELED': {'color': 'g', 'zorder': 3, 'alpha': 1.0},
    'NTP': {'color': 'k', 'zorder': 2, 'alpha': 1.0},
    'UNK': {'color': 'm', 'zorder': 2, 'alpha': 1.0},
}
bins = {
    'PC': np.linspace(-5, 5, 100),
    'AFP': np.linspace(-5, 5, 100),
    'NTP': np.linspace(-50, 50, 100),
    'UNK': np.linspace(-25, 25, 100),
    'MISLABELED': np.linspace(-5, 5, 100)
}
noise_label = 'MISLABELED'

for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
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
    aum_tbl.loc[aum_tbl['label_changed_to_other_class'] == True, 'label'] = noise_label

    f, ax = plt.subplots()
    ax.hist(aum_tbl.loc[aum_tbl['label'] != 'MISLABELED', 'margin'], np.linspace(-50, 50, 100), edgecolor='k',
            label='Other Samples')
    ax.hist(aum_tbl.loc[aum_tbl['label'] == 'MISLABELED', 'margin'], np.linspace(-50, 50, 100), edgecolor='k',
            label='Thr. Samples')
    ax.set_xlabel('AUM')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    f.savefig(run_dir / 'hist_aum_thr_vs_normal_samples.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(aum_tbl.loc[aum_tbl['label'].isin(['PC', 'AFP']), 'margin'], np.linspace(-5, 5, 100),
            edgecolor='k', label='Other Samples PC/AFP', zorder=1, alpha=1)
    ax.hist(aum_tbl.loc[aum_tbl['label'] == 'MISLABELED', 'margin'], np.linspace(-5, 5, 100),
            edgecolor='k', label='Thr. Samples', zorder=2, alpha=0.5)
    ax.set_xlabel('AUM')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    f.savefig(run_dir / 'hist_aum_thr_vs_normal_samples_pc-afp.png')
    plt.close()

    # f, ax = plt.subplots()
    # for label in labels:  # aum_tbl['label'].unique():
    #     ax.hist(aum_tbl.loc[aum_tbl['label'] != label, 'margin'], bins, edgecolor='k', label=f'{label}', **labels[label])
    # ax.set_xlabel('AUM')
    # ax.set_ylabel('Counts')
    # ax.set_yscale('log')
    # ax.legend()
    # plt.show()
    for label in labels:  # aum_tbl['label'].unique():
        f, ax = plt.subplots()
        ax.hist(aum_tbl.loc[aum_tbl['label'] == label, 'margin'], bins[label], edgecolor='k', label=f'{label}',
                **labels[label])
        ax.set_xlabel('AUM')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.legend()
        f.savefig(run_dir / f'hist_aum_{label}.png')
        plt.close()

    # look at margin change over epochs
    margins_dir = run_dir / 'models' / 'model1' / 'margins'

    cols_tbl = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label', 'margin']
    drop_dupl_cols = ['target_id', 'tce_plnt_num', 'label', 'shard_name', 'example_i', 'original_label']

    margins_tbl = [pd.read_csv(tbl, usecols=cols_tbl) if tbl_i == 0 else pd.read_csv(tbl, usecols=['margin'])
                   for tbl_i, tbl in enumerate(margins_dir.iterdir())]
    margins_tbl = pd.concat(margins_tbl, axis=1)
    # add column indicator for injected noise to the margins table
    margins_tbl = margins_tbl.merge(dataset_tbl,
                                    on=['target_id', 'tce_plnt_num'],
                                    how='left',
                                    validate='one_to_one')
    margins_tbl.loc[margins_tbl['label_changed_to_other_class'] == True, 'label'] = noise_label
    margins_tbl.to_csv(run_dir / 'margins_tbl.csv', index=False)
    # margins_tbl.drop_duplicates(drop_dupl_cols, inplace=True)

    margins_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

    f, ax = plt.subplots()
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(10982872, 3), 'margin'], label='PC')
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(7983756, 1), 'margin'], label='AFP')
    ax.plot(np.arange(n_epochs), margins_tbl.loc[(6141300, 1), 'margin'], label='NTP')
    # ax.plot(np.arange(n_epochs), margins_tbl.loc[(10460984, 1), 'margin'], label='MISLABELED')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Margin')
    ax.legend()
    f.savefig(run_dir / 'margin_over_epochs_examples_all.png')
    plt.close()

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
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-03-2022_1215')

runs_dir = experiment_dir / 'runs'

aum_tbls = []
for run_dir in [fp for fp in sorted(runs_dir.iterdir()) if fp.is_dir()]:
    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')

    # adding column for examples injected with label noise
    trainset_tbl = pd.read_csv(experiment_dir / 'trainset_runs' /
                               f'trainset_{run_dir.name}.csv')[['target_id',
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
    aum_tbl.loc[aum_tbl['label_changed_to_other_class'] == True, 'label'] = noise_label

    # margin_mislabeled = aum_tbl.loc[(aum_tbl['label'] == noise_label) & (aum_tbl['dataset'] == 'train'), 'margin']
    margin_mislabeled = aum_tbl.loc[
        (aum_tbl['label_changed_to_other_class'] == True) & (aum_tbl['dataset'] == 'train'), 'margin']

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

# plot histogram of injected noise examples vs regular examples across all runs
for dataset in ['train', 'val', 'test']:
    aum_dataset = aum_allruns_tbl.loc[
        (aum_allruns_tbl['dataset'] == dataset) & (aum_allruns_tbl['original_label'].isin(['PC', 'AFP']))]
    f, ax = plt.subplots()
    ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'], 'mean'], bins, edgecolor='k',
            label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
    ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'] == False, 'mean'], bins, edgecolor='k',
            label='Regular examples', zorder=1)
    ax.set_xlabel('AUM')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    f.savefig(experiment_dir / f'hist_aum_inj_vs_normal_samples_{dataset}.png')
    plt.close()

# plot histogram of injected noise examples vs regular examples vs mislabeled examples for each run
n_runs = 9
for run in range(n_runs):

    run_dir = experiment_dir / 'runs' / f'run{run}'

    aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
    aum_tbl = aum_tbl.merge(aum_allruns_tbl[['target_id', 'tce_plnt_num', 'label_changed_to_other_class']],
                            on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')

    for dataset in ['train', 'val', 'test']:
        aum_dataset = aum_tbl.loc[(aum_tbl['dataset'] == dataset) & (aum_tbl['original_label'].isin(['PC', 'AFP']))]

        f, ax = plt.subplots(figsize=(16, 10))
        ax.hist(aum_dataset.loc[aum_dataset['label_changed_to_other_class'], 'margin'], bins, edgecolor='k',
                label='Stochastic injected label noise examples', zorder=3, alpha=0.5)
        ax.hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] != 'MISLABELED'), 'margin'], bins, edgecolor='k', label='Regular examples',
                zorder=1)
        ax.hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] == 'MISLABELED'), 'margin'], bins, edgecolor='k', label='Thr. examples',
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
                    aum_dataset['original_label'] == 'PC'), 'margin'], bins, edgecolor='k',
                   label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
        ax[0].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] != 'MISLABELED') & (aum_dataset['original_label'] == 'PC'), 'margin'], bins,
                   edgecolor='k', label='Regular examples', zorder=1)
        ax[0].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] == 'MISLABELED') & (aum_dataset['original_label'] == 'PC'), 'margin'], bins,
                   edgecolor='k', label='Thr. examples', zorder=3, alpha=0.5)
        ax[1].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class']) & (
                    aum_dataset['original_label'] == 'AFP'), 'margin'], bins, edgecolor='k',
                   label='Stochastic injected label noise examples', zorder=2, alpha=0.5)
        ax[1].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] != 'MISLABELED') & (aum_dataset['original_label'] == 'AFP'), 'margin'], bins,
                   edgecolor='k', label='Regular examples', zorder=1)
        ax[1].hist(aum_dataset.loc[(aum_dataset['label_changed_to_other_class'] == False) & (
                    aum_dataset['label'] == 'MISLABELED') & (aum_dataset['original_label'] == 'AFP'), 'margin'], bins,
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
            injected_tbl = pd.read_csv(experiment_dir / 'val_test_sets_labels_switched.csv')[['target_id',
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

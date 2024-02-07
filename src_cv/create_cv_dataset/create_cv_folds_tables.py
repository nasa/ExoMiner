""" Prepare dataset TCE tables into folds for cross-validation. """

# 3rd party
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# %% set up CV experiment variables

data_dir = Path(f'/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_updated_labels_{datetime.now().strftime("%m-%d-%Y_%H%M")}')

rnd_seed = 24
n_folds_eval = 10  # which is also the number of shards
n_folds_predict = 10
tce_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels.csv')
dataset_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_s1-s67_all_12-6-2023_1039_adddiffimg_perimgnormdiffimg_updatedlabels_2-2-2024/shards_tbl.csv')
# unlabeled cats TCEs become part of the predict set; not evaluation
unlabeled_cats = [
    # # Kepler
    # 'UNK',
    # TESS
    'UNK',
    # 'PC',
    # 'APC',
]

#%% prepare the shards for CV by splitting the TCE table into shards (n folds)

data_dir.mkdir(exist_ok=True)

shard_tbls_dir = data_dir / 'shard_tables' / 'eval'
shard_tbls_dir.mkdir(exist_ok=True, parents=True)

# set up logger
logger = logging.getLogger(name='prepare_cv_tables_run')
logger_handler = logging.FileHandler(filename=data_dir / f'prepare_cv_tables_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

logger.info(f'Setting random seed to {rnd_seed}')
rng = np.random.default_rng(rnd_seed)

logger.info(f'Number of folds used for CV: {n_folds_eval}')

# n_models = 10
# logger.info(f'Number of models in the ensemble: {n_models}')

# load the TCE table
logger.info(f'Reading TCE table {tce_tbl_fp}')
tce_tbl = pd.read_csv(tce_tbl_fp)

# load table with TCEs used in the dataset
dataset_tbl = pd.read_csv(dataset_tbl_fp)

# dataset_tbls_dir = Path(
#     '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_12-03-2021_1106')
# dataset_tbl = pd.concat([pd.read_csv(set_tbl) for set_tbl in dataset_tbls_dir.iterdir()
#                          if set_tbl.name.endswith('.csv') and not set_tbl.name.startswith('predict')])
# logger.info(f'Using dataset tables from {str(dataset_tbls_dir)} to filter examples used.')
logger.info(f'Using data set table: {dataset_tbl_fp}')

# remove TCEs not used in the dataset
logger.info(f'Removing examples not in the data set... (total number of examples in dataset before removing '
            f'examples: {len(dataset_tbl)})')

tces_found_in_src_dataset = tce_tbl['uid'].isin(dataset_tbl['uid'])
logger.info(f'TCEs found in source data set that are in the TCE table: {tces_found_in_src_dataset.sum()} (out of {len(tce_tbl)})')
tce_tbl_dataset = tce_tbl.loc[tces_found_in_src_dataset]
# removing unlabeled examples or examples not used for evaluation
used_tces = ~tce_tbl_dataset['label'].isin(unlabeled_cats)
working_tce_tbl = tce_tbl_dataset.loc[used_tces]

logger.info(f'Removing unlabeled examples or examples not used for evaluation: {used_tces.sum()} examples left.')

logger.info(f'Total number of examples in dataset after removing examples: {len(working_tce_tbl)}')

# shuffle per target stars
logger.info('Shuffling TCE table per target stars...')
target_star_grps = [df for _, df in working_tce_tbl.groupby('target_id')]
logger.info(f'Number of target stars: {len(target_star_grps)}')
rng.shuffle(target_star_grps)
working_tce_tbl = pd.concat(target_star_grps).reset_index(drop=True)
# tce_tbl = pd.concat(rng.permutation(tce_tbl.groupby('target_id')))

# # shuffle per TCE
# logger.info('Shuffling TCE table per TCE...')
# tce_tbl = tce_tbl.sample(frac=1, random_state=rnd_seed).reset_index(drop=True)

working_tce_tbl.to_csv(data_dir / f'{tce_tbl_fp.stem}_shuffled.csv', index=False)

# # define test set for paper dataset as one of the folds
# test_set_tbl = pd.read_csv(dataset_tbls_dir / 'testset.csv')
# test_set_tbl['tceid'] = test_set_tbl[['target_id', 'tce_plnt_num']].apply(
#     lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
# fold_tce_tbl = tce_tbl.loc[tce_tbl['tceid'].isin(test_set_tbl['tceid'])]
# fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold9.csv', index=False)
# n_folds -= 1
# tce_tbl = tce_tbl.loc[~tce_tbl['tceid'].isin(test_set_tbl['tceid'])]

# split TCE table into n fold tables (all TCEs from the same target star should be in the same table)
logger.info(f'Split TCE table into {n_folds_eval} fold TCE tables...')

# split at the target star level
logger.info('Splitting at target star level...')
target_star_grps = [df for _, df in working_tce_tbl.groupby('target_id')]
target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds_eval, axis=0)
for fold_i, target_stars_split in enumerate(target_stars_splits):
    fold_tce_tbl = pd.concat(target_star_grps[target_stars_split[0]:target_stars_split[-1] + 1])
    # shuffle TCEs in each fold
    fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng.integers(10),
                                       axis=0).reset_index(drop=True)
    fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

# # split at the TCE level
# logger.info('Splitting at TCE level...')
# tce_splits = np.array_split(range(len(tce_tbl)), n_folds, axis=0)
# for fold_i, tce_split in enumerate(tce_splits):
#     fold_tce_tbl = tce_tbl[tce_split[0]:tce_split[-1] + 1]
#     # shuffle TCEs in each fold
#     fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng.integers(10),
#                                        axis=0).reset_index(drop=True)
#     fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

# tces not in the eval set
logger.info('Splitting at TCE level...')
shard_tbls_dir = data_dir / 'shard_tables' / 'predict'
shard_tbls_dir.mkdir(exist_ok=True, parents=True)
unused_tces = ~tce_tbl_dataset['uid'].isin(working_tce_tbl['uid'])
logger.info(f'Number of TCEs in the predict set: {unused_tces.sum()}')
tce_tbl_noteval = tce_tbl_dataset.loc[unused_tces]
tce_tbl_noteval.to_csv(data_dir / f'examples_noteval.csv', index=False)
tce_splits = np.array_split(range(len(tce_tbl_noteval)), n_folds_predict, axis=0)
for fold_i, tce_split in enumerate(tce_splits):
    fold_tce_tbl = tce_tbl_noteval[tce_split[0]:tce_split[-1] + 1]
    fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

#%% Check distribution of examples per fold

shard_tbls_dir = data_dir / 'shard_tables' / 'eval'

for tbl_fp in shard_tbls_dir.iterdir():

    tbl = pd.read_csv(tbl_fp)
    print(f'Fold {tbl_fp}')
    print(f'Disposition distribution:\n{tbl["label"].value_counts()}')

    cnt_tces_target = \
        tbl['target_id'].value_counts().to_frame(name='num_tces_target').reset_index().rename(
            columns={'index': 'target_id'})

    print(f'Number of TCEs per target:\n{cnt_tces_target["num_tces_target"].value_counts()}')
    print(f'{"#" * 100}')

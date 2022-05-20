""" Prepare dataset TCE tables into folds for cross-validation. """

# 3rd party
import logging
# import multiprocessing
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# # local
# from src_cv.utils_cv import create_shard_fold

# %% set up CV experiment variables

experiment = f'cv_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
data_dir = Path(f'/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/{experiment}')
data_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='prepare_cv_tables_run')
logger_handler = logging.FileHandler(filename=data_dir / f'prepare_cv_tables_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

rnd_seed = 24
logger.info(f'Setting random seed to {rnd_seed}')
rng = np.random.default_rng(rnd_seed)

n_folds = 10  # which is also the number of shards
logger.info(f'Number of folds used for CV: {n_folds}')
# n_models = 10
# logger.info(f'Number of models in the ensemble: {n_models}')

#%% prepare the shards for CV by splitting the TCE table into shards (n folds)

shard_tbls_dir = data_dir / 'shard_tables'
shard_tbls_dir.mkdir(exist_ok=True)

# load the TCE table
tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
logger.info(f'Reading TCE table {tce_tbl_fp}')
tce_tbl = pd.read_csv(tce_tbl_fp)

# load table with TCEs used in the dataset
dataset_tbls_dir = Path(
    '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_12-03-2021_1106')
dataset_tbl = pd.concat([pd.read_csv(set_tbl) for set_tbl in dataset_tbls_dir.iterdir()
                         if set_tbl.name.endswith('.csv') and not set_tbl.name.startswith('predict')])
logger.info(f'Using dataset tables from {str(dataset_tbls_dir)} to filter examples used.')

# remove TCEs not used in the dataset
logger.info(
    f'Removing examples not used... (Total number of examples in dataset before removing examples: {len(dataset_tbl)})')
dataset_tbl['tceid'] = dataset_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
tce_tbl['tceid'] = tce_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
tce_tbl = tce_tbl.loc[tce_tbl['tceid'].isin(dataset_tbl['tceid'])]
logger.info(f'(Total number of examples in dataset after removing examples: {len(dataset_tbl)})')

# shuffle per target stars
logger.info('Shuffling TCE table per target stars...')
target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
logger.info(f'Number of target stars: {len(target_star_grps)}')
rng.shuffle(target_star_grps)
tce_tbl = pd.concat(target_star_grps).reset_index(drop=True)
# tce_tbl = pd.concat(rng.permutation(tce_tbl.groupby('target_id')))

# # shuffle per TCE
# logger.info('Shuffling TCE table per TCE...')
# tce_tbl = tce_tbl.sample(frac=1, random_state=rnd_seed).reset_index(drop=True)

tce_tbl.to_csv(data_dir / f'{tce_tbl_fp.stem}_shuffled.csv', index=False)

# # define test set for paper dataset as one of the folds
# test_set_tbl = pd.read_csv(dataset_tbls_dir / 'testset.csv')
# test_set_tbl['tceid'] = test_set_tbl[['target_id', 'tce_plnt_num']].apply(
#     lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
# fold_tce_tbl = tce_tbl.loc[tce_tbl['tceid'].isin(test_set_tbl['tceid'])]
# fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold9.csv', index=False)
# n_folds -= 1
# tce_tbl = tce_tbl.loc[~tce_tbl['tceid'].isin(test_set_tbl['tceid'])]

# split TCE table into n fold tables (all TCEs from the same target star should be in the same table)
logger.info(f'Split TCE table into {n_folds} fold TCE tables...')

# split at the target star level
logger.info('Splitting at target star level...')
target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds, axis=0)
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

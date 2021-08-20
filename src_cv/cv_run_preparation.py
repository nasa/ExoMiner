""" Prepare dataset into folds for cross-validation. """

import logging
import multiprocessing
from datetime import datetime
# 3rd party
from pathlib import Path

import numpy as np
import pandas as pd

# local
from src_cv.utils_cv import create_shard_fold

# %% set up CV experiment variables

experiment = f'cv_{datetime.now().strftime("%m-%d-%Y_%H-%M")}'
data_dir = Path(f'/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/{experiment}')
data_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='train-eval_run')
logger_handler = logging.FileHandler(filename=data_dir / f'cv_run.log', mode='w')
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
tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                  'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                  'symsecphase_confirmedkoiperiod_sec.csv')
logger.info(f'Reading TCE table {tce_tbl_fp}')
tce_tbl = pd.read_csv(tce_tbl_fp)

# load table with TCEs used in the dataset
dataset_tbls_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020')
dataset_tbl = pd.concat([pd.read_csv(set_tbl) for set_tbl in dataset_tbls_dir.iterdir()
                         if set_tbl.name.endswith('.csv')])

# remove TCEs not used in the dataset
dataset_tbl['tceid'] = dataset_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
tce_tbl['tceid'] = tce_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
tce_tbl = tce_tbl.loc[tce_tbl['tceid'].isin(dataset_tbl['tceid'])]

# # shuffle per target stars
# logger.info('Shuffling TCE table per target stars...')
# target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
# rng.shuffle(target_star_grps)
# tce_tbl = pd.concat(target_star_grps).reset_index(drop=True)
# # tce_tbl = pd.concat(rng.permutation(tce_tbl.groupby('target_id')))

# shuffle per TCE
logger.info('Shuffling TCE table per TCE...')
tce_tbl = tce_tbl.sample(frac=1, random_state=rnd_seed).reset_index(drop=True)

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

# # split at the target star level
# target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
# target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds, axis=0)
# for fold_i, target_stars_split in enumerate(target_stars_splits):
#     fold_tce_tbl = pd.concat(target_star_grps[target_stars_split[0]:target_stars_split[-1] + 1])
#     # shuffle TCEs in each fold
#     fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng.integers(10),
#                                        axis=0).reset_index(drop=True)
#     fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

# split at the TCE level
tce_splits = np.array_split(range(len(tce_tbl)), n_folds, axis=0)
for fold_i, tce_split in enumerate(tce_splits):
    fold_tce_tbl = tce_tbl[tce_split[0]:tce_split[-1] + 1]
    # shuffle TCEs in each fold
    fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng.integers(10),
                                       axis=0).reset_index(drop=True)
    fold_tce_tbl.to_csv(shard_tbls_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

# %% build TFRecord dataset based on the fold TCE tables

data_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_08-17-2021_03-57')
shard_tbls_dir = data_dir / 'shard_tables'

tfrec_dir_root = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                      'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar'
                      '-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/')
# tfrec_dir_root = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
#                       'tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-'
#                       '6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/')

src_tfrec_dir = tfrec_dir_root / \
                'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-' \
                'bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_' \
                'experiment-labels-norm_nopps_secparams_prad_period'
# src_tfrec_dir = tfrec_dir_root / 'tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-' \
#                                  'centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_' \
#                                  'confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps'
# logger.info(f'Source TFRecord directory: {src_tfrec_dir}')

dest_tfrec_dir = data_dir / 'tfrecords'
dest_tfrec_dir.mkdir(exist_ok=True)
logger.info(f'Destination TFRecord directory: {dest_tfrec_dir}')

src_tfrec_tbl = pd.read_csv(src_tfrec_dir / 'shards_tce_tbl.csv')
shard_tbls_fps = sorted(list(shard_tbls_dir.iterdir()))

n_processes = 10
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl) for fold_i, shard_tbl_fp
        in enumerate(shard_tbls_fps)]
async_results = [pool.apply_async(create_shard_fold, job) for job in jobs]
pool.close()

tces_not_found = [async_result.get() for async_result in async_results]
tces_not_found = [tce_not_found[0] for tce_not_found in tces_not_found if len(tce_not_found) > 0]
tces_not_found_df = pd.DataFrame(tces_not_found,
                                 columns=['target_id', 'tce_plnt_num'])
tces_not_found_df.to_csv(data_dir / 'tces_not_found.csv', index=False)

#%% Create CV run

data_shards_fps = list(dest_tfrec_dir.iterdir())
data_shards = [data_shards_fp.name for data_shards_fp in data_shards_fps]

# assert len(data_dir.iterdir()) == num_folds
num_folds = len(data_shards_fps)
# assert num_unit_folds <= num_folds
folds_arr = np.arange(num_folds)

cv_folds_runs = [{'train': data_shards[:fold_i] + data_shards[fold_i + 1:],
                  'test': [data_shards[fold_i]]} for fold_i in folds_arr]

np.save(data_dir / f'cv_folds_runs.npy', cv_folds_runs)

logged_cv_folds = [{dataset: [fold_dataset for fold_dataset in folds_dataset]
                    for dataset, folds_dataset in cv_fold_run.items()} for cv_fold_run in cv_folds_runs]
logger.info(f'CV folds: {logged_cv_folds}')

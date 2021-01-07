from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

#%%

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
n_models = 10
logger.info(f'Number of models in the ensemble: {n_models}')

#%% prepare the shards for CV

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec.csv')
logger.info(f'Reading TCE table {tce_tbl_fp}')
tce_tbl = pd.read_csv(tce_tbl_fp)

# shuffle per target stars
logger.info('Shuffling TCE table per target stars...')
target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
rng.shuffle(target_star_grps)
tce_tbl = pd.concat(target_star_grps).reset_index(drop=True)
# tce_tbl = pd.concat(rng.permutation(tce_tbl.groupby('target_id')))

tce_tbl.to_csv(data_dir / f'{tce_tbl_fp.stem}_shuffled.csv', index=False)

# split TCE table into n fold tables (all TCEs from the same target star should be in the same table)
logger.info(f'Split TCE table into {n_folds} fold TCE tables...')
target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds, axis=0)
for fold_i, target_stars_split in enumerate(target_stars_splits):
    fold_tce_tbl = pd.concat(target_star_grps[target_stars_split[0]:target_stars_split[-1] + 1])
    # shuffle TCEs in each fold
    fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng.integers(10),
                                       axis=0).reset_index(drop=True)
    fold_tce_tbl.to_csv(data_dir / f'{tce_tbl_fp.stem}_fold{fold_i}.csv', index=False)

#%% build TFRecord dataset based on the fold TCE tables

src_tfrec_dir = Path()
dest_tfrec_dir = Path()

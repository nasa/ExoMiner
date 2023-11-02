"""
Script used to split TCE table into training, validation, test, and predict sets.

Input: TCE table
Output: tables for training, validation, test, and predict sets
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

#%% set parameters

# saving directory
destTfrecDir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/data/dataset_splits/kepler_q1q17dr25/'
                    f'split_q1q17dr25_kepler_simulated_inj1-3_scr1-2_inv_{datetime.now().strftime("%m-%d-%Y_%H%M")}')
# source TCE table
experimentTceTbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed_updtstellar_preprocessed.csv')
rnd_seed = 24  # random seed
# split ratio
dataset_frac = {'train': 0.8, 'val': 0.1, 'test': 0.1}

#%% create data set tables

assert sum(dataset_frac.values()) == 1

destTfrecDir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='split_dataset_run')
logger_handler = logging.FileHandler(filename=destTfrecDir / f'split_dataset.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

logger.info(f'Setting random seed to {rnd_seed}')
rng = np.random.default_rng(rnd_seed)

logger.info(f'Using as source table the TCE table {experimentTceTbl_fp}')
experimentTceTbl = pd.read_csv(experimentTceTbl_fp)
experimentTceTbl.sort_values(['target_id', 'tce_plnt_num'], ascending=True, inplace=True)
experimentTceTbl.reset_index(drop=True, inplace=True)

# get TCEs with disposition and without (i.e, `label` set to 'UNK')
predict_tces = experimentTceTbl.loc[experimentTceTbl['label'] == 'UNK']
labeled_tces = experimentTceTbl.loc[experimentTceTbl['label'] != 'UNK']

logger.info(f'TCE disposition count:\n {labeled_tces["label"].value_counts()}')

labeled_tces.to_csv(destTfrecDir / f'{experimentTceTbl_fp.stem}_labeledtces.csv', index=False)
logger.info(f'Number of TCEs in predict set: {len(predict_tces)}')
predict_tces.to_csv(destTfrecDir / f'predictset.csv', index=False)

# shuffle and split per target stars
logger.info('Shuffling at target star level...')
target_star_grps = [df for _, df in labeled_tces.groupby('target_id')]
rng.shuffle(target_star_grps)
labeled_tces = pd.concat(target_star_grps, axis=0).reset_index(drop=True)
targetStars = labeled_tces['target_id'].unique()
numTargetStars = len(targetStars)
logger.info(f'Number of target starts: {numTargetStars}')

logger.info('Splitting into training, validation and test sets...')
logger.info(f'Dataset split: {dataset_frac}')

lastTargetStar = {'train': targetStars[int(dataset_frac['train'] * numTargetStars)],
                  'val': targetStars[int((dataset_frac['train'] + dataset_frac['val']) * numTargetStars)]}

trainIdx = labeled_tces.loc[labeled_tces['target_id'] == lastTargetStar['train']].index[-1] + 1
valIdx = labeled_tces.loc[labeled_tces['target_id'] == lastTargetStar['val']].index[-1] + 1

logger.info(f'Train idx: {trainIdx}\nValidation index: {valIdx}')

datasetTbl = {'train': labeled_tces[:trainIdx],
              'val': labeled_tces[trainIdx:valIdx],
              'test': labeled_tces[valIdx:]}

assert len(np.intersect1d(datasetTbl['train']['target_id'].unique(), datasetTbl['val']['target_id'].unique())) == 0
assert len(np.intersect1d(datasetTbl['train']['target_id'].unique(), datasetTbl['test']['target_id'].unique())) == 0
assert len(np.intersect1d(datasetTbl['val']['target_id'].unique(), datasetTbl['test']['target_id'].unique())) == 0

# shuffle TCEs in each dataset
logger.info('Shuffling TCEs inside each dataset...')
np.random.seed(rnd_seed)
datasetTbl = {dataset: datasetTbl[dataset].iloc[rng.permutation(len(datasetTbl[dataset]))]
              for dataset in datasetTbl}

for dataset in datasetTbl:
    logger.info(f'Saving TCE table for dataset {dataset}...')
    datasetTbl[dataset].to_csv(destTfrecDir / f'{dataset}set.csv', index=False)

for dataset in datasetTbl:
    logger.info(dataset)
    logger.info(datasetTbl[dataset]['label'].value_counts())
    logger.info(f'Number of TCEs in {dataset} set: {len(datasetTbl[dataset])}')

logger.info('Finished splitting data in table.')

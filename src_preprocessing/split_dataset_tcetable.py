"""
Script used to split TCE table into training, validation and test sets.
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# %% create training, validation and test datasets

# saving directory
destTfrecDir = Path(f'/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/'
                    f'split_{datetime.now().strftime("%m-%d-%Y_%H%M")}')
destTfrecDir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='split_dataset_run')
logger_handler = logging.FileHandler(filename=destTfrecDir / f'split_dataset.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

rnd_seed = 24  # random seed
logger.info(f'Setting random seed to {rnd_seed}')
rng = np.random.default_rng(rnd_seed)

# source TCE table
# experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
#                                'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
#                                'noRobovetterKOIs.csv')
# experimentTceTbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'\
#                            'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_'\
#                            'nomissingval_symsecphase_confirmedkoiperiod_sec.csv')
experimentTceTbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
logger.info(f'Using as source table the TCE table {experimentTceTbl_fp}')
experimentTceTbl = pd.read_csv(experimentTceTbl_fp)
experimentTceTbl.sort_values(['target_id', 'tce_plnt_num'], ascending=True, inplace=True)
experimentTceTbl.reset_index(drop=True, inplace=True)

# create unused dataset
unused_tce_tbl = experimentTceTbl.loc[(experimentTceTbl['label'] == 'UNK') |
                                      (experimentTceTbl['tce_rogue_flag'] == 1)]
unused_tce_tbl.to_csv(destTfrecDir / 'predictset.csv', index=False)

# remove rogue TCEs
experimentTceTbl = experimentTceTbl.loc[experimentTceTbl['tce_rogue_flag'] == 0]
logger.info('Removing rogue TCEs...')
# # keep only CONFIRMED KOIs, CFP, CFA and POSSIBLE PLANET KOIs, and non-KOI
# experimentTceTbl = experimentTceTbl.loc[(experimentTceTbl['koi_disposition'] == 'CONFIRMED') |
#                                         (experimentTceTbl['fpwg_disp_status'].isin(['CERTIFIED FP',
#                                                                                     'CERTIFIED FA',
#                                                                                     'POSSIBLE PLANET'])) |
#                                         experimentTceTbl['kepoi_name'].isna()]
# remove unused KOIs
experimentTceTbl = experimentTceTbl.loc[experimentTceTbl['label'] != 'UNK']
logger.info('Removing unused KOIs...')
experimentTceTbl.reset_index(drop=True, inplace=True)

# logger.info(f'Number of TCEs after removing KOIs dispositioned by Robovetter: {len(experimentTceTbl)}')

# # set labels
# experimentTceTbl[['label']] = 'NTP'
# experimentTceTbl.loc[experimentTceTbl['koi_disposition'] == 'CONFIRMED', ['label']] = 'PC'
# experimentTceTbl.loc[(experimentTceTbl['koi_disposition'] != 'CONFIRMED') &
#                      (experimentTceTbl['fpwg_disp_status'] == 'CERTIFIED FP'), ['label']] = 'AFP'

logger.info(f'TCE disposition count:\n {experimentTceTbl["label"].value_counts()}')
logger.info(f'TCE label id count:\n {experimentTceTbl["label_id"].value_counts()}')

experimentTceTbl.to_csv(destTfrecDir / f'{experimentTceTbl_fp.stem}_allTCEs.csv', index=False)

# shuffle per target stars
logger.info('Shuffling at target star level...')
target_star_grps = [df for _, df in experimentTceTbl.groupby('target_id')]
rng.shuffle(target_star_grps)
experimentTceTbl = pd.concat(target_star_grps, axis=0).reset_index(drop=True)
targetStars = experimentTceTbl['target_id'].unique()
numTargetStars = len(targetStars)
logger.info(f'Number of target starts: {numTargetStars}')

logger.info('Splitting into training, validation and test sets...')
dataset_frac = {'train': 0.8, 'val': 0.1, 'test': 0.1}
logger.info(f'Dataset split: {dataset_frac}')

assert sum(dataset_frac.values()) == 1

lastTargetStar = {'train': targetStars[int(dataset_frac['train'] * numTargetStars)],
                  'val': targetStars[int((dataset_frac['train'] + dataset_frac['val']) * numTargetStars)]}

trainIdx = experimentTceTbl.loc[experimentTceTbl['target_id'] == lastTargetStar['train']].index[-1] + 1
valIdx = experimentTceTbl.loc[experimentTceTbl['target_id'] == lastTargetStar['val']].index[-1] + 1

logger.info(f'Train idx: {trainIdx}\nValidation index: {valIdx}')

datasetTbl = {'train': experimentTceTbl[:trainIdx],
              'val': experimentTceTbl[trainIdx:valIdx],
              'test': experimentTceTbl[valIdx:]}

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

# #%% create unused TCE table
#
# tce_tbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_11-17-2021_1255/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_allTCEs.csv')
# tce_tbl = tce_tbl.loc[(tce_tbl['original_label'] == 'UNK') | (tce_tbl['tce_rogue_flag'] == 1)]
# tce_tbl.to_csv(destTfrecDir / 'predictset.csv', index=False)

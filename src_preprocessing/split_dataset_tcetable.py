import numpy as np
import pandas as pd
import os

#%% create training, validation and test datasets

experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                               'noRobovetterKOIs.csv')

destTfrecDir = ''

# shuffle per target stars
targetStars = experimentTceTbl['target_id'].unique()
numTargetStars = len(targetStars)

dataset_frac = {'train': 0.8, 'val': 0.1, 'test': 0.1}

assert sum(dataset_frac.values()) == 1

lastTargetStar = {'train': targetStars[int(dataset_frac['train'] * numTargetStars)],
                  'val': targetStars[int((dataset_frac['train'] + dataset_frac['val']) * numTargetStars)]}

numTces = {'total': len(experimentTceTbl)}
trainIdx = experimentTceTbl.loc[experimentTceTbl['target_id'] == lastTargetStar['train']].index[-1] + 1  # int(numTces['total'] * dataset_frac['train'])
valIdx = experimentTceTbl.loc[experimentTceTbl['target_id'] == lastTargetStar['val']].index[-1] + 1 # int(numTces['total'] * (dataset_frac['train'] + dataset_frac['val']))

print('Train idx: {}\nValidation index: {}'.format(trainIdx, valIdx))

datasetTbl = {'train': experimentTceTbl[:trainIdx],
              'val': experimentTceTbl[trainIdx:valIdx],
              'test': experimentTceTbl[valIdx:]}

assert len(np.intersect1d(datasetTbl['train']['target_id'].unique(), datasetTbl['val']['target_id'].unique())) == 0
assert len(np.intersect1d(datasetTbl['train']['target_id'].unique(), datasetTbl['test']['target_id'].unique())) == 0
assert len(np.intersect1d(datasetTbl['val']['target_id'].unique(), datasetTbl['test']['target_id'].unique())) == 0

# shuffle TCEs in each dataset
np.random.seed(24)
datasetTbl = {dataset: datasetTbl[dataset].iloc[np.random.permutation(len(datasetTbl[dataset]))]
              for dataset in datasetTbl}

for dataset in datasetTbl:
    datasetTbl[dataset].to_csv(os.path.join(destTfrecDir, '{}set.csv'.format(dataset)), index=False)

for dataset in datasetTbl:
    numTces[dataset] = len(datasetTbl[dataset])
    print(datasetTbl[dataset]['label'].value_counts())
    print('Number of TCEs in {} set: {}'.format(dataset, len(datasetTbl[dataset])))

""" Script used to manipulate TFRecords by performing operations such as splitting the dataset into training,
validation and test sets, normalizing features, ...
"""

# 3rd party
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import itertools
import multiprocessing
from astropy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# local
from src_preprocessing.preprocess import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc, \
    centering_and_normalization
from src_preprocessing.tf_util import example_util
from src_preprocessing.utils_manipulate_tfrecords import create_shard, update_labels, plot_features_example

plt.switch_backend('TkAgg')

#%% variables used across the script

tceIdentifier = 'tce_plnt_num'  # TCE identifier

#%% define directories

srcTfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-tps_g301-l31_6tr_spline_gapped1_flux-loe-lwks-centroid-centroidfdl-6stellar-stdts_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-tps_g301-l31_6tr_spline_gapped1_flux-loe-lwks-centroid-centroidfdl-6stellar-stdts_correctprimarygapping_confirmedkoiperiod')
srcTfrecTbls = sorted([file for file in srcTfrecDir.iterdir() if file.suffix == '.csv' and
                       file.stem.startswith('shard')])

srcTfrecTblMerge = None
for srcTfrecTbl in srcTfrecTbls:
    srcTfrecDf = pd.read_csv(srcTfrecTbl)

    if srcTfrecTblMerge is None:
        srcTfrecTblMerge = srcTfrecDf
    else:
        srcTfrecTblMerge = pd.concat([srcTfrecTblMerge, srcTfrecDf])

srcTfrecTblMerge.to_csv(srcTfrecDir / 'merged_shards.csv', index=True)

#%% create new TFRecords based on the original ones

# load train, val and test datasets
# datasetTblDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/old/bug_with_transitduration_amplified/' \
#                 'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/' \
#                 'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment'
datasetTblDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020')
datasetTbl = {dataset: pd.read_csv(datasetTblDir / f'{dataset}set.csv') for dataset in ['train', 'val', 'test']}

# get only TCEs with tce_plnt_num = 1
# datasetTbl = {dataset: datasetTbl[dataset].loc[datasetTbl[dataset]['tce_plnt_num'] == 1] for dataset in datasetTbl}

numTces = {}
for dataset in datasetTbl:
    numTces[dataset] = len(datasetTbl[dataset])
    print(datasetTbl[dataset]['label'].value_counts())
    print(f'Number of TCEs in {dataset} set: {len(datasetTbl[dataset])}')

# defined number of examples per shard
numTcesPerShard = np.min([100, numTces['train'], numTces['val'], numTces['test']])
numShards = {dataset: int(numTces[dataset] / numTcesPerShard) for dataset in datasetTbl}

assert np.all(list(numShards.values()))

print(f'Number of shards per dataset: {numShards}')

# create pairs of shard filename and shard TCE table
shardFilenameTuples = []
for dataset in datasetTbl:
    shardFilenameTuples.extend(list(itertools.product([dataset], range(numShards[dataset]), [numShards[dataset] - 1])))
shardFilenames = ['{}-shard-{:05d}-of-{:05d}'.format(*shardFilenameTuple) for shardFilenameTuple in shardFilenameTuples]

shardTuples = []
for shardFilename in shardFilenames:

    shardFilenameSplit = shardFilename.split('-')
    dataset, shard_i = shardFilenameSplit[0], int(shardFilenameSplit[2])

    if shard_i < numShards[dataset] - 1:
        shardTuples.append((shardFilename,
                            datasetTbl[dataset][shard_i * numTcesPerShard:(shard_i + 1) * numTcesPerShard]))
    else:
        shardTuples.append((shardFilename,
                            datasetTbl[dataset][shard_i * numTcesPerShard:]))

checkNumTcesInDatasets = {dataset: 0 for dataset in datasetTbl}
for shardTuple in shardTuples:
    checkNumTcesInDatasets[shardTuple[0].split('-')[0]] += len(shardTuple[1])
print(checkNumTcesInDatasets)

# srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod'
srcTbl = pd.read_csv(os.path.join(srcTfrecDir / 'merged_shards.csv'), index_col=0)
destTfrecDir = Path(str(srcTfrecDir) + '_starshuffle_experiment')
destTfrecDir.mkdir(exist_ok=True)
omitMissing = True
nProcesses = 15
pool = multiprocessing.Pool(processes=nProcesses)
jobs = [shardTuple + (srcTbl, srcTfrecDir, destTfrecDir, tceIdentifier, omitMissing) for shardTuple in shardTuples]
async_results = [pool.apply_async(create_shard, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

print('TFRecord dataset created.')

# check the number of Examples in the TFRecord shards and that each TCE example for a given dataset is in the TFRecords
tfrecFiles = [file for file in destTfrecDir.iterdir() if 'shard' in file.stem]
countExamples = []  # total number of examples
for tfrecFile in tfrecFiles:

    dataset = tfrecFile.stem.split('-')[0]

    countExamplesShard = 0

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        foundTce = datasetTbl[dataset].loc[(datasetTbl[dataset]['target_id'] == targetIdTfrec) &
                                           (datasetTbl[dataset][tceIdentifier] == tceIdentifierTfrec)]

        if len(foundTce) == 0:
            raise ValueError(f'TCE {targetIdTfrec}-{tceIdentifierTfrec} not found in the dataset tables.')

        countExamplesShard += 1
    countExamples.append(countExamplesShard)


#%% update labels with a given set of dispositions

srcTfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-tps_g301-l31_6tr_spline_gapped1_flux-loe-lwks-centroid-centroidfdl-6stellar-stdts_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-tps_g301-l31_6tr_spline_gapped1_flux-loe-lwks-centroid-centroidfdl-6stellar-stdts_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment')
destTfrecDir = Path(str(srcTfrecDir) + '-labels')
destTfrecDir.mkdir(exist_ok=True)

# dispositions coming from the experiment TCE table
experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                               'nomissingval_rmcandandfpkois_norogues.csv')
omitMissing = True
srcTfrecFiles = [file for file in srcTfrecDir.iterdir() if 'shard' in file.stem]

nProcesses = 15
pool = multiprocessing.Pool(processes=nProcesses)
jobs = [(destTfrecDir, file, experimentTceTbl, tceIdentifier, omitMissing) for file in srcTfrecFiles]
async_results = [pool.apply_async(update_labels, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

print('Labels updated.')

# confirm that the labels are correct
tfrecFiles = [file for file in destTfrecDir.iterdir() if 'shard' in file.stem]
countExamples = []
for tfrecFile in tfrecFiles:

    countExamplesShard = 0

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        tceLabelTbl = experimentTceTbl.loc[(experimentTceTbl['target_id'] == targetIdTfrec) &
                                           (experimentTceTbl[tceIdentifier] == tceIdentifierTfrec)]['label'].values[0]

        if tceLabelTbl != labelTfrec:
            countExamplesShard += 1

    countExamples.append(countExamplesShard)

assert np.sum(countExamples) == 0

#%% compute normalization statistics for scalar parameters, timeseries, ...

tfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/tfrecords_cv_hpo/')

# get only training set TFRecords
tfrecTrainFiles = [file for file in tfrecDir.iterdir() if file.stem.startswith('train-shard')]

# get out-of-transit indices for the local and global views
nr_transit_durations = 6  # 2 * 2.5 + 1  # 2 * 4 + 1  # number of transit durations (2*n+1, n on each side of the transit)
num_bins_loc = 31  # 201
num_bins_glob = 301  # 2001

scalarParams = {
    # stellar parameters
    'tce_steff': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                  'clip_factor': np.nan, 'dtype': 'int'},
    'tce_slogg': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                  'clip_factor': np.nan, 'dtype': 'float'},
    'tce_smet': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                 'clip_factor': np.nan, 'dtype': 'float'},
    'tce_sradius': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                    'clip_factor': np.nan, 'dtype': 'float'},
    'tce_smass': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                  'clip_factor': np.nan, 'dtype': 'float'},
    'tce_sdens': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                  'clip_factor': np.nan, 'dtype': 'float'},
    # secondary
    'tce_maxmes': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                   'clip_factor': np.nan, 'dtype': 'float'},
    'wst_depth': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                  'clip_factor': 20, 'dtype': 'float', 'replace_value': 0},
    # 'tce_albedo': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                'clip_factor': np.nan, 'dtype': 'float'},
    'tce_albedo_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                        'dtype': 'float'},
    # 'tce_ptemp': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #               'clip_factor': np.nan, 'dtype': 'float'},
    'tce_ptemp_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                       'dtype': 'float'},
    # 'wst_robstat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                 'clip_factor': np.nan, 'dtype': 'float'},
    # odd-even
    # 'tce_bin_oedp_stat': {'missing_value': np.nan, 'log_transform': True, 'log_transform_eps': 1e-32,
    #                       'clip_factor': np.nan, 'dtype': 'float'},
    # other diagnostics
    'boot_fap': {'missing_value': -1, 'log_transform': True, 'log_transform_eps': 1e-32,
                 'clip_factor': np.nan, 'dtype': 'float'},
    'tce_cap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float'},
    'tce_hap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float'},
    'tce_rb_tcount0': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': np.nan, 'dtype': 'int'},
    # centroid
    'tce_fwm_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                     'dtype': 'float'},
    'tce_dikco_msky': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': np.nan, 'dtype': 'float'},
    'tce_dicco_msky': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': np.nan, 'dtype': 'float'},
    'tce_dikco_msky_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'float'},
    'tce_dicco_msky_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'float'},
    # flux
    'transit_depth': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float'},
    # 'tce_depth_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                   'clip_factor': np.nan, 'dtype': 'float'},
    # 'tce_duration': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                  'clip_factor': np.nan, 'dtype': 'float'},
    # 'tce_duration_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                      'clip_factor': np.nan, 'dtype': 'float'},
    'tce_period': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                   'clip_factor': np.nan, 'dtype': 'float'},
    # 'tce_period_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                    'clip_factor': np.nan, 'dtype': 'float'},
    # 'tce_max_mult_ev': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
    #                     'clip_factor': np.nan, 'dtype': 'float'},
    'tce_prad': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                 'clip_factor': 20, 'dtype': 'float'}
}

scalarParamsDict = {scalarParam: [] for scalarParam in scalarParams}

# FDL centroid time series normalization statistics parameters
timeSeriesFDLList = ['global_centr_fdl_view', 'local_centr_fdl_view']
timeSeriesFDLDict = {timeSeries: [] for timeSeries in timeSeriesFDLList}

idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(num_bins_loc, nr_transit_durations)  # same for all TCEs

# our centroid time series normalization statistics parameters
centroidList = ['global_centr_view', 'local_centr_view']
centroidDict = {timeSeries: [] for timeSeries in centroidList}

# get data out of the training set TFRecords
for tfrec_i, tfrecFile in enumerate(tfrecTrainFiles):

    print(f'Getting data from {tfrecFile.name} ({tfrec_i / len(tfrecTrainFiles) * 100} %)')

    # iterate through the shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # get scalar parameters data
        # tceScalarParams = example.features.feature['scalar_params'].float_list.value
        # scalarParamsMat.append(tceScalarParams)
        for scalarParam in scalarParams:
            if scalarParams[scalarParam]['dtype'] == 'int':
                scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].int64_list.value[0])
            elif scalarParams[scalarParam]['dtype'] == 'float':
                scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].float_list.value[0])

        # get FDL centroid time series data
        transitDuration = example.features.feature['tce_duration'].float_list.value[0]
        orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
        idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(num_bins_glob, transitDuration, orbitalPeriod)
        for timeSeries in timeSeriesFDLList:
            timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
            if 'glob' in timeSeries:
                timeSeriesFDLDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
            else:
                timeSeriesFDLDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_loc])

        # get centroid time series data
        for timeSeries in centroidList:
            timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
            # centroidMat[timeSeries].extend(timeSeriesTce)
            if 'glob' in timeSeries:
                centroidDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
            else:
                centroidDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_loc])

# save normalization statistics for the scalar parameters (median and robust estimator of std)
# do not use missing values to compute the normalization statistics for bootstrap FA probability
scalarParamsDict = {scalarParam: np.array(scalarParamVals) for scalarParam, scalarParamVals in scalarParamsDict.items()}
scalarNormStats = {scalarParam: {'median': np.nan, 'mad_std': np.nan, 'info': scalarParams[scalarParam]}
                   for scalarParam in scalarParams}
for scalarParam in scalarParams:

    scalarParamVals = scalarParamsDict[scalarParam]

    # remove missing values so that they do not contribute to the normalization statistics
    if not np.isnan(scalarParams[scalarParam]['missing_value']):
        if scalarParam == 'wst_depth':
            scalarParamVals = scalarParamVals[np.where(scalarParamVals > scalarParams[scalarParam]['missing_value'])]
        else:
            scalarParamVals = scalarParamVals[np.where(scalarParamVals != scalarParams[scalarParam]['missing_value'])]

    # log transform the data
    if scalarParams[scalarParam]['log_transform']:

        # add constant value
        if not np.isnan(scalarParams[scalarParam]['log_transform_eps']):
            scalarParamVals += scalarParams[scalarParam]['log_transform_eps']

        scalarParamVals = np.log10(scalarParamVals)

    # compute median as robust estimate of central tendency
    scalarNormStats[scalarParam]['median'] = np.median(scalarParamVals)
    # compute MAD std as robust estimate of deviation from central tendency
    scalarNormStats[scalarParam]['mad_std'] = stats.mad_std(scalarParamVals)

# save normalization statistics into a numpy file
np.save(tfrecDir / 'train_scalarparam_norm_stats.npy', scalarNormStats)

# create additional csv file with normalization statistics
scalarNormStatsDataForDf = {}
for scalarParam in scalarParams:
    scalarNormStatsDataForDf[f'{scalarParam}_median'] = [scalarNormStats[scalarParam]['median']]
    scalarNormStatsDataForDf[f'{scalarParam}_mad_std'] = [scalarNormStats[scalarParam]['mad_std']]
scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)
scalarNormStatsDf.to_csv(tfrecDir / 'train_scalarparam_norm_stats.csv', index=False)

# save normalization statistics for FDL centroid time series
normStatsFDL = {timeSeries:  {
                              # 'values': timeSeriesFDLMat,
                              'oot_median': np.median(timeSeriesFDLDict[timeSeries]),
                              'oot_std': stats.mad_std(timeSeriesFDLDict[timeSeries])
                              }
                for timeSeries in timeSeriesFDLList}
np.save(tfrecDir / 'train_fdlcentroid_norm_stats.npy', normStatsFDL)

# create additional csv file with normalization statistics
normStatsFDLDataForDf = {}
for timeSeries in timeSeriesFDLList:
    normStatsFDLDataForDf[f'{timeSeries}_oot_median'] = [normStatsFDL[timeSeries]['oot_median']]
    normStatsFDLDataForDf[f'{timeSeries}_oot_std'] = [normStatsFDL[timeSeries]['oot_std']]
normStatsFDLDf = pd.DataFrame(data=normStatsFDLDataForDf)
normStatsFDLDf.to_csv(tfrecDir / 'train_fdlcentroid_norm_stats.csv', index=False)

# save normalization statistics for centroid time series
normStatsCentroid = {timeSeries:  {
                                   'median': np.median(centroidDict[timeSeries]),
                                   'std': stats.mad_std(centroidDict[timeSeries]),
                                   'clip_value': 30
                                   # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
                                   #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
                                   }
                     for timeSeries in centroidList}
for timeSeries in centroidList:
    centroidMatClipped = np.clip(centroidDict[timeSeries], a_max=30, a_min=None)
    clipStats = {
        'median_clip': np.median(centroidMatClipped),
        'std_clip': stats.mad_std(centroidMatClipped)
    }
    normStatsCentroid[timeSeries].update(clipStats)
np.save(tfrecDir / 'train_centroid_norm_stats.npy', normStatsCentroid)

# create additional csv file with normalization statistics
normStatsCentroidDataForDf = {}
for timeSeries in centroidList:
    normStatsCentroidDataForDf[f'{timeSeries}_median'] = [normStatsCentroid[timeSeries]['median']]
    normStatsCentroidDataForDf[f'{timeSeries}_std'] = [normStatsCentroid[timeSeries]['std']]
    # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
    normStatsCentroidDataForDf[f'{timeSeries}_clip_value'] = 30
    normStatsCentroidDataForDf[f'{timeSeries}_median_clip'] = [normStatsCentroid[timeSeries]['median_clip']]
    normStatsCentroidDataForDf[f'{timeSeries}_std_clip'] = [normStatsCentroid[timeSeries]['std_clip']]
normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)
normStatsCentroidDf.to_csv(tfrecDir / 'train_centroid_norm_stats.csv', index=False)

#%% normalize examples in the TFRecords


def normalize_examples(destTfrecDir, srcTfrecFile, normStats, auxParams):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  Path, destination TFRecord directory for the normalized data
    :param srcTfrecFile: Path, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :return:
    """

    # get out-of-transit indices for the local views
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(auxParams['num_bins_loc'],
                                                              auxParams['nr_transit_durations'])  # same for all TCEs

    with tf.io.TFRecordWriter(str(destTfrecDir / srcTfrecFile.name)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(srcTfrecFile))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # normalize scalar parameters
            normalizedFeatures = {}
            for scalarParam in normStats['scalar_params']:
                # get the scalar value from the example
                if normStats['scalar_params'][scalarParam]['info']['dtype'] == 'int':
                    scalarParamVal = np.array(example.features.feature[scalarParam].int64_list.value)
                elif normStats['scalar_params'][scalarParam]['info']['dtype'] == 'float':
                    scalarParamVal = np.array(example.features.feature[scalarParam].float_list.value)

                # remove missing values so that they do not contribute to the normalization statistics
                # for wst_depth, replace values smaller than missing_value by the replace value
                if scalarParamVal < normStats['scalar_params'][scalarParam]['info']['missing_value'] and \
                        scalarParam == 'wst_depth':
                    scalarParamVal = normStats['scalar_params'][scalarParam]['info']['replace_value']
                # replace missing value by the median
                elif scalarParamVal == normStats['scalar_params'][scalarParam]['info']['missing_value']:
                    scalarParamVal = normStats['scalar_params'][scalarParam]['median']

                else:  # in the case that the value is not a missing value
                    # log transform the data
                    if normStats['scalar_params'][scalarParam]['info']['log_transform']:
                        # add constant value
                        if not np.isnan(normStats['scalar_params'][scalarParam]['info']['log_transform_eps']):
                            scalarParamVal += normStats['scalar_params'][scalarParam]['info']['log_transform_eps']

                        scalarParamVal = np.log10(scalarParamVal)

                    # clipping the data
                    # if not np.isnan(normStats['scalar_params']['info'][scalarParam]['clip_factor']):
                    #     scalarParamVal = np.clip([scalarParamVal],
                    #                              normStats['scalar_params']['median'][scalarParam] -
                    #                              normStats['scalar_params']['info'][scalarParam]['clip_factor'] *
                    #                              normStats['scalar_params']['mad_std'][scalarParam],
                    #                              normStats['scalar_params']['median'][scalarParam] +
                    #                              normStats['scalar_params']['info'][scalarParam]['clip_factor'] *
                    #                              normStats['scalar_params']['mad_std'][scalarParam]
                    #                              )[0]
                    if not np.isnan(normStats['scalar_params'][scalarParam]['info']['clip_factor']):
                        scalarParamVal = np.clip([scalarParamVal],
                                                 normStats['scalar_params'][scalarParam]['median'] -
                                                 normStats['scalar_params'][scalarParam]['info']['clip_factor'] *
                                                 normStats['scalar_params'][scalarParam]['mad_std'],
                                                 normStats['scalar_params'][scalarParam]['median'] +
                                                 normStats['scalar_params'][scalarParam]['info']['clip_factor'] *
                                                 normStats['scalar_params'][scalarParam]['mad_std']
                                                 )[0]

                # standardization
                # scalarParamVal = (scalarParamVal - normStats['scalar_params']['median'][scalarParam]) /  \
                #                  normStats['scalar_params']['mad_std'][scalarParam]
                scalarParamVal = (scalarParamVal - normStats['scalar_params'][scalarParam]['median']) /  \
                                 normStats['scalar_params'][scalarParam]['mad_std']

                # tceScalarParams[normStats['scalar_params']['info'][scalarParam]['idx']] = scalarParamVal
                normalizedFeatures[f'{scalarParam}_norm'] = [scalarParamVal]

            # normalize FDL centroid time series
            # get out-of-transit indices for the global views
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(auxParams['num_bins_glob'],
                                                                        transitDuration,
                                                                        orbitalPeriod)
            # compute oot global and local flux views std
            glob_flux_view_std = \
                np.std(
                    np.array(example.features.feature['global_'
                                                      'flux_view_'
                                                      'fluxnorm'].float_list.value)[idxs_nontransitcadences_glob],
                    ddof=1)
            loc_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['local_'
                                                 'flux_view_'
                                                 'fluxnorm'].float_list.value)[idxs_nontransitcadences_loc], ddof=1)
            # center and normalize FDL centroid time series
            glob_centr_fdl_view = np.array(example.features.feature['global_centr_fdl_view'].float_list.value)
            glob_centr_fdl_view_norm = \
                centering_and_normalization(glob_centr_fdl_view,
                                            normStats['fdl_centroid']['global_centr_fdl_view']['oot_median'],
                                            normStats['fdl_centroid']['global_centr_fdl_view']['oot_std']
                                            )
            glob_centr_fdl_view_norm *= glob_flux_view_std / \
                                        np.std(glob_centr_fdl_view_norm[idxs_nontransitcadences_glob], ddof=1)
            loc_centr_fdl_view = np.array(example.features.feature['local_centr_fdl_view'].float_list.value)
            loc_centr_fdl_view_norm = \
                centering_and_normalization(loc_centr_fdl_view,
                                            normStats['fdl_centroid']['local_centr_fdl_view']['oot_median'],
                                            normStats['fdl_centroid']['local_centr_fdl_view']['oot_std']
                                            )
            loc_centr_fdl_view_norm *= loc_flux_view_std / np.std(loc_centr_fdl_view_norm[idxs_nontransitcadences_loc],
                                                                  ddof=1)

            # normalize centroid time series
            glob_centr_view = np.array(example.features.feature['global_centr_view'].float_list.value)
            loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)

            # 1) clipping to physically meaningful distance in arcsec
            glob_centr_view_std_clip = np.clip(glob_centr_view,
                                               a_max=normStats['centroid']['global_centr_view']['clip_value'],
                                               a_min=None)
            glob_centr_view_std_clip = centering_and_normalization(glob_centr_view_std_clip,
                                                                   # normStats['centroid']['global_centr_view']['median_'
                                                                   #                                            'clip'],
                                                                   np.median(glob_centr_view_std_clip),
                                                                   normStats['centroid']['global_centr_view']['std_'
                                                                                                              'clip'])

            loc_centr_view_std_clip = np.clip(loc_centr_view, a_max=normStats['centroid']['local_centr_view']['clip_'
                                                                                                              'value'],
                                              a_min=None)
            loc_centr_view_std_clip = centering_and_normalization(loc_centr_view_std_clip,
                                                                   # normStats['centroid']['local_centr_view']['median_'
                                                                   #                                           'clip'],
                                                                  np.median(loc_centr_view_std_clip),
                                                                   normStats['centroid']['local_centr_view']['std_'
                                                                                                             'clip'])

            # 2) no clipping
            glob_centr_view_std_noclip = centering_and_normalization(glob_centr_view,
                                                                     # normStats['centroid']['global_centr_'
                                                                     #                       'view']['median'],
                                                                     np.median(glob_centr_view),
                                                                     normStats['centroid']['global_centr_view']['std'])
            loc_centr_view_std_noclip = centering_and_normalization(loc_centr_view,
                                                                    # normStats['centroid']['local_centr_view']['median'],
                                                                    np.median(loc_centr_view),
                                                                    normStats['centroid']['local_centr_view']['std'])

            # 3) center each centroid individually using their median and divide by the standard deviation of the
            # training set
            glob_centr_view_medind_std = glob_centr_view - np.median(glob_centr_view)
            glob_centr_view_medind_std /= normStats['centroid']['global_centr_view']['std']
            loc_centr_view_medind_std = loc_centr_view - np.median(loc_centr_view)
            loc_centr_view_medind_std /= normStats['centroid']['local_centr_view']['std']

            # add features to the example in the TFRecord
            normalizedFeatures.update({
                # 'scalar_params': tceScalarParams,
                'local_centr_fdl_view_norm': loc_centr_fdl_view_norm,
                'global_centr_fdl_view_norm': glob_centr_fdl_view_norm,
                'global_centr_view_std_clip': glob_centr_view_std_clip,
                'local_centr_view_std_clip': loc_centr_view_std_clip,
                'global_centr_view_std_noclip': glob_centr_view_std_noclip,
                'local_centr_view_std_noclip': loc_centr_view_std_noclip,
                'global_centr_view_medind_std': glob_centr_view_medind_std,
                'local_centr_view_medind_std': loc_centr_view_medind_std
            })

            for normalizedFeature in normalizedFeatures:
                example_util.set_float_feature(example, normalizedFeature, normalizedFeatures[normalizedFeature],
                                               allow_overwrite=True)

            writer.write(example.SerializeToString())


srcTfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/tfrecords_cv_hpo/')
destTfrecDir = Path(str(srcTfrecDir) + '-normalized')
destTfrecDir.mkdir(exist_ok=True)
srcTfrecFiles = [file for file in srcTfrecDir.iterdir() if 'shard' in file.stem and file.suffix != '.csv']

auxParams = {
    'nr_transit_durations': 6,  # 2 * 2.5 + 1,  # 2 * 4 + 1,  # number of transit durations (2*n+1, n on each side of the transit)
    'num_bins_loc': 31,  # 31, 201
    'num_bins_glob': 301  # 301, 2001
}

normStatsDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/tfrecords_cv_hpo/')

# load normalization statistics
normStats = {
    'scalar_params': np.load(normStatsDir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item(),
    'fdl_centroid': np.load(normStatsDir / 'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item(),
    'centroid': np.load(normStatsDir / 'train_centroid_norm_stats.npy', allow_pickle=True).item()
}
# for TPS
# normStats['scalar_params'] = {key: val for key, val in normStats['scalar_params'].items()
#                               if key in ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens',
#                                          'transit_depth', 'tce_duration', 'tce_period', 'tce_max_mult_ev']}
# for TESS
# normStats['scalar_params'] = {key: val for key, val in normStats['scalar_params'].items()
#                               if key in ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens',
#                                          'transit_depth', 'tce_duration', 'tce_period']}
# normStats['scalar_params']['tce_steff']['info']['dtype'] = 'float'

nProcesses = 15
pool = multiprocessing.Pool(processes=nProcesses)
jobs = [(destTfrecDir, file, normStats, auxParams) for file in srcTfrecFiles]
async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
pool.close()

for async_result in async_results:
    async_result.get()

print('Normalization finished.')

#%% Check final preprocessed data

# TFRecord directory
tfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/tfrecords_cv_hpo/tfrecords_cv_hpo-normalized')
# create plot directory if it does not exist
plotDir = tfrecDir / 'plots_all_views'
plotDir.mkdir(exist_ok=True)
# get filepaths to TFRecord files
tfrecFiles = [file for file in tfrecDir.iterdir() if 'shard' in file.stem]

tceIdentifier = 'tce_plnt_num'

# set views to be plotted
views = [
    'global_flux_view',
    'local_flux_view',
    'global_flux_view_fluxnorm',
    'local_flux_view_fluxnorm',
    # 'global_flux_odd_view',
    'local_flux_odd_view',
    'local_flux_odd_view_fluxnorm',
    # 'global_flux_even_view',
    'local_flux_even_view',
    'local_flux_even_view_fluxnorm',
    # 'local_flux_oddeven_view_diff',
    # 'local_flux_oddeven_view_diff_dir',
    # 'global_weak_secondary_view',
    # 'local_weak_secondary_view',
    # 'local_weak_secondary_view_selfnorm',
    # 'local_weak_secondary_view_fluxnorm',
    # 'local_weak_secondary_view_max_flux-wks_norm',
    # centroid
    'global_centr_view',
    'local_centr_view',
    # 'global_centr_view_std_clip',
    # 'local_centr_view_std_clip',
    'global_centr_view_std_noclip',
    'local_centr_view_std_noclip',
    # 'global_centr_view_medind_std',
    # 'local_centr_view_medind_std',
    # 'global_centr_view_medcmaxn',
    # 'local_centr_view_medcmaxn',
    # 'global_centr_view_medcmaxn_dir',
    # 'local_centr_view_medcmaxn_dir',
    # 'global_centr_view_medn',
    # 'local_centr_view_medn',
    'global_centr_fdl_view',
    'local_centr_fdl_view',
    'global_centr_fdl_view_norm',
    'local_centr_fdl_view_norm',
]

# set scalar parameter values to be extracted
scalarParams = [
    # stellar parameters
    'tce_steff',
    'tce_slogg',
    'tce_smet',
    'tce_sradius',
    'tce_smass',
    'tce_sdens',
    # secondary
    'wst_depth',
    'tce_maxmes',
    'tce_albedo_stat',
    # 'tce_albedo',
    'tce_ptemp_stat',
    # 'tce_ptemp',
    # 'wst_robstat',
    # odd-even
    # 'tce_bin_oedp_stat',
    # other parameters
    'boot_fap',
    'tce_cap_stat',
    'tce_hap_stat',
    'tce_rb_tcount0',
    # centroid
    'tce_fwm_stat',
    'tce_dikco_msky',
    'tce_dikco_msky_err',
    'tce_dicco_msky',
    'tce_dicco_msky_err',
    # flux
    # 'tce_max_mult_ev',
    # 'tce_depth_err',
    # 'tce_duration_err',
    # 'tce_period_err',
    'transit_depth',
    'tce_prad',
    'tce_period'
]

# set this to get the normalized scalar parameters
tceOfInterest = '9773869-1'  # (6500206, 2)  # (9773869, 1) AFP with clear in-transit shift, (8937762, 1) nice PC, (8750094, 1)  # (8611832, 1)
scheme = (3, 6)
basename = 'all_views'  # basename for figures
# probPlot = 1.01  # probability threshold for plotting

for tfrecFile in tfrecFiles:
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        # if np.random.random() > probPlot:
        #     continue

        example = tf.train.Example()
        example.ParseFromString(string_record)

        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        if tceIdentifier == 'oi':
            tceIdentifierTfrec = example.features.feature[tceIdentifier].float_list.value[0]
            tceid = str(tceIdentifierTfrec).split('.')
            tceid = f'{tceid[0]}.{tceid[1][:2]}'
        else:
            tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
            tceid = f'{targetIdTfrec}-{tceIdentifierTfrec}'



        if tceid != tceOfInterest:
            continue
        # else:
        #     tceFound = True

        # get scalar parameters
        scalarParamsStr = ''
        for scalarParam_i in range(len(scalarParams)):
            scalar_param_val_norm = example.features.feature[f'{scalarParams[scalarParam_i]}_norm'].float_list.value[0]
            if scalarParams[scalarParam_i] in ['tce_steff', 'tce_rb_tcount0']:
                scalar_param_val = example.features.feature[scalarParams[scalarParam_i]].int64_list.value[0]
            else:
                scalar_param_val = example.features.feature[scalarParams[scalarParam_i]].float_list.value[0]

            if scalarParam_i % 6 == 0 and scalarParam_i != 0:
                scalarParamsStr += '\n'
            if scalarParams[scalarParam_i] in ['boot_fap']:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4E})|'
            elif scalarParams[scalarParam_i] in ['tce_steff', 'tce_rb_tcount0']:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val})|'
            else:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4f})|'

        # get label
        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        # get time series views
        viewsDict = {}
        for view in views:
            viewsDict[view] = np.array(example.features.feature[view].float_list.value)

        # plot features
        plot_features_example(viewsDict, scalarParamsStr, tceid, labelTfrec, plotDir, scheme, basename=basename,
                              display=False)
        aaa


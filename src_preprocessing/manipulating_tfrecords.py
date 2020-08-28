import pandas as pd
import os
import tensorflow as tf
import numpy as np
import itertools
import multiprocessing
from astropy import stats
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from src_preprocessing.preprocess import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc, \
    centering_and_normalization
from src_preprocessing.tf_util import example_util

#%% define directories

srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband'
srcTfrecTbls = sorted([os.path.join(srcTfrecDir, file) for file in os.listdir(srcTfrecDir)
                       if file.endswith('.csv') and file.startswith('shard')])

srcTfrecTblMerge = None
for srcTfrecTbl in srcTfrecTbls:
    srcTfrecDf = pd.read_csv(srcTfrecTbl)

    if srcTfrecTblMerge is None:
        srcTfrecTblMerge = srcTfrecDf
    else:
        srcTfrecTblMerge = pd.concat([srcTfrecTblMerge, srcTfrecDf])

srcTfrecTblMerge.to_csv(os.path.join(srcTfrecDir, 'merged_shards.csv'), index=True)

#%% create new TFRecords based on the original ones

# load train, val and test datasets
datasetTblDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/old/bug_with_transitduration_amplified/' \
                'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/' \
                'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment'
datasetTbl = {dataset: pd.read_csv(os.path.join(datasetTblDir, '{}set.csv'.format(dataset)))
              for dataset in ['train', 'val', 'test']}

# get only TCEs with tce_plnt_num = 1
# datasetTbl = {dataset: datasetTbl[dataset].loc[datasetTbl[dataset]['tce_plnt_num'] == 1] for dataset in datasetTbl}

numTces = {}
for dataset in datasetTbl:
    numTces[dataset] = len(datasetTbl[dataset])
    print(datasetTbl[dataset]['label'].value_counts())
    print('Number of TCEs in {} set: {}'.format(dataset, len(datasetTbl[dataset])))

tceIdentifier = 'tce_plnt_num'  # TCE identifier

# defined number of examples per shard
numTcesPerShard = np.min([100, numTces['train'], numTces['val'], numTces['test']])
numShards = {dataset: int(numTces[dataset] / numTcesPerShard) for dataset in datasetTbl}

assert np.all(list(numShards.values()))

print('Number of shards per dataset:', numShards)

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


def create_shard(shardFilename, shardTbl, srcTbl, srcTfrecDir, destTfrecDir, tceIdentifier='tce_plnt_num',
                 omitMissing=True):
    """ Create a TFRecord file (shard) based on a set of existing TFRecord files.

    :param shardFilename: str, shard filename
    :param shardTbl: pandas DataFrame, shard TCE table
    :param srcTbl: pandas DataFrame, source TCE table
    :param srcTfrecDir: str, filepath to directory with the source TFRecords
    :param destTfrecDir: str, filepath to directory in which to save the new TFRecords
    :param tceIdentifier: str, used to identify TCEs. Either `tce_plnt_num` or `oi`
    :param omitMissing: bool, omit missing TCEs in teh source TCE table
    :return:
    """

    # with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, shardFilename)) as writer:
    with tf.io.TFRecordWriter(os.path.join(destTfrecDir, shardFilename)) as writer:

        # iterate through TCEs in the shard TCE table
        for tce_i, tce in shardTbl.iterrows():

            # check if TCE is in the source TFRecords TCE table
            foundTce = srcTbl.loc[(srcTbl['target_id'] == tce['target_id']) &
                                  (srcTbl[tceIdentifier] == tce[tceIdentifier])]['shard']

            if len(foundTce) > 0:

                tceIdx = foundTce.index[0]
                tceFoundInTfrecordFlag = False

                # record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(srcTfrecDir, foundTce.values[0]))
                tfrecord_dataset = tf.data.TFRecordDataset(os.path.join(srcTfrecDir, foundTce.values[0]))

                for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                # for string_i, string_record in enumerate(record_iterator):
                    # index in the source TFRecords TCE table follow the writing order in the source TFRecords, so it
                    # can be used to access a specific TCE
                    if string_i == tceIdx:
                        example = tf.train.Example()
                        example.ParseFromString(string_record)

                        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
                        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
                        assert targetIdTfrec == tce['target_id'] and tceIdentifierTfrec == tce[tceIdentifier]

                        tceFoundInTfrecordFlag = True
                        break

                if not tceFoundInTfrecordFlag:
                    raise ValueError('TCE for shard {} not found in the TFRecords:'
                                     ' {}-{}.'.format(shardFilename, tce['target_id'], tce[tceIdentifier]))

                writer.write(example.SerializeToString())

            else:
                if omitMissing:  # omit missing TCEs in the source TCE table
                    print('TCE for shard {} not found in the TFRecords merged table:'
                          ' {}-{}.'.format(shardFilename, tce['target_id'], tce[tceIdentifier]))
                    continue

                raise ValueError('TCE for shard {} not found in the TFRecords merged table:'
                                 ' {}-{}.'.format(shardFilename, tce['target_id'], tce[tceIdentifier]))


srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband'
srcTbl = pd.read_csv(os.path.join(srcTfrecDir, 'merged_shards.csv'), index_col=0)
destTfrecDir = srcTfrecDir + '_starshuffle_experiment'
os.makedirs(destTfrecDir, exist_ok=True)
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
tfrecFiles = [os.path.join(destTfrecDir, file) for file in os.listdir(destTfrecDir) if 'shard' in file]
countExamples = []  # total number of examples
tceIdentifier = 'tce_plnt_num'
for tfrecFile in tfrecFiles:

    dataset = tfrecFile.split('/')[-1].split('-')[0]

    countExamplesShard = 0

    # iterate through the source shard
    # record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecFile)

    for string_record in tfrecord_dataset.as_numpy_iterator():
    # for string_i, string_record in enumerate(record_iterator):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        foundTce = datasetTbl[dataset].loc[(datasetTbl[dataset]['target_id'] == targetIdTfrec) &
                                           (datasetTbl[dataset][tceIdentifier] == tceIdentifierTfrec)]

        if len(foundTce) == 0:
            raise ValueError('TCE {}-{}'.format(targetIdTfrec, tceIdentifierTfrec))

        countExamplesShard += 1
    countExamples.append(countExamplesShard)


#%% update labels with a given set of dispositions

def update_labels(destTfrecDir, srcTfrecFile, tceTbl, tceIdentifier, omitMissing=True):
    """ Update example label field in the TFRecords.

    :param destTfrecDir: str, destination TFRecord directory with data following the updatedlabels given
    :param srcTfrecFile: str, source TFRecord directory
    :param tceTbl: pandas DataFrame, TCE table with labels for the TCEs
    :param tceIdentifier: str, TCE identifier in the TCE table. Either `tce_plnt_num` or `oi`, it depends also on the
    columns in the table
    :param omitMissing: bool, if True it skips missing TCEs in the TCE table
    :return:
    """

    # with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:
    with tf.io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:

        # iterate through the source shard
        # record_iterator = tf.python_io.tf_record_iterator(path=srcTfrecFile)
        tfrecord_dataset = tf.data.TFRecordDataset(srcTfrecFile)

        for string_record in tfrecord_dataset.as_numpy_iterator():
        # for string_i, string_record in enumerate(record_iterator):

            example = tf.train.Example()
            example.ParseFromString(string_record)

            tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
            targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

            foundTce = tceTbl.loc[(tceTbl['target_id'] == targetIdTfrec) &
                                  (tceTbl[tceIdentifier] == tceIdentifierTfrec)]

            if len(foundTce) > 0:
                tceLabel = foundTce['label'].values[0]
                example_util.set_feature(example, 'label', [tceLabel], allow_overwrite=True)
                writer.write(example.SerializeToString())
            else:
                if omitMissing:
                    print('TCE {}-{} not found in the TCE table'.format(targetIdTfrec, tceIdentifierTfrec))
                    continue

                raise ValueError('TCE {}-{} not found in the TCE table'.format(targetIdTfrec, tceIdentifierTfrec))


srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment'
destTfrecDir = srcTfrecDir + '-labels'
os.makedirs(destTfrecDir, exist_ok=True)

# dispositions coming from the experiment TCE table
experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                               'noRobovetterKOIs.csv')
tceIdentifier = 'tce_plnt_num'
omitMissing = True
srcTfrecFiles = [os.path.join(srcTfrecDir, file) for file in os.listdir(srcTfrecDir) if 'shard' in file]

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
tfrecFiles = [os.path.join(destTfrecDir, file) for file in os.listdir(destTfrecDir) if 'shard' in file]
countExamples = []
for tfrecFile in tfrecFiles:

    countExamplesShard = 0

    # iterate through the source shard
    # record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecFile)

    for string_record in tfrecord_dataset.as_numpy_iterator():
    # for string_i, string_record in enumerate(record_iterator):

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

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment'

# get only training set TFRecords
tfrecTrainFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if file.startswith('train-shard')]

tceIdentifier = 'tce_plnt_num'

# scalar parameters are also present in the TCE table, since that was where they came from, but to keep things more
# flexible, we will also take them from the TFRecords
# scalarParams = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens', 'wst_robstat',
#                 'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']
# scalarParams = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat',
#                 'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat']
scalarParams = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat',
                'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat', 'tce_rb_tcount0']
numScalarParams = len(scalarParams)
scalarParamsMat = []

# FDL centroid time series normalization statistics parameters
timeSeriesFDLList = ['global_centr_fdl_view', 'local_centr_fdl_view']
timeSeriesFDLMat = {timeSeries: [] for timeSeries in timeSeriesFDLList}
# get out-of-transit indices for the local and global views
nr_transit_durations = 2 * 4 + 1  # number of transit durations (2*n+1, n on each side of the transit)
num_bins_loc = 201
num_bins_glob = 2001
idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(num_bins_loc, nr_transit_durations)  # same for all TCEs

# our centroid time series normalization statistics parameters
centroidList = ['global_centr_view', 'local_centr_view']
centroidMat = {timeSeries: [] for timeSeries in centroidList}

for tfrec_i, tfrecFile in enumerate(tfrecTrainFiles):

    print('Getting data from {} ({} %)'.format(tfrecFile.split('/')[-1], tfrec_i / len(tfrecTrainFiles) * 100))

    # iterate through the shard
    # record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecFile)

    for string_record in tfrecord_dataset.as_numpy_iterator():
    # for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # get scalar parameters data
        tceScalarParams = example.features.feature['scalar_params'].float_list.value

        scalarParamsMat.append(tceScalarParams)

        # get FDL centroid time series data
        transitDuration = example.features.feature['tce_duration'].float_list.value[0]
        orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
        idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(num_bins_glob, transitDuration, orbitalPeriod)
        for timeSeries in timeSeriesFDLList:
            timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
            if 'glob' in timeSeries:
                timeSeriesFDLMat[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
            else:
                timeSeriesFDLMat[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_loc])

        # get centroid time series data
        for timeSeries in centroidList:
            timeSeriesTce = example.features.feature[timeSeries].float_list.value
            centroidMat[timeSeries].extend(timeSeriesTce)

# save normalization statistics for the scalar parameters (median and robust estimator of std)
# do not use missing values to compute the normalization statistics for bootstrap FA probability
scalarParamsMat = np.array(scalarParamsMat)

# transform bootstrap FA probability
bfapArr = scalarParamsMat[:, 7]
bfapArrValid = bfapArr[np.where(bfapArr != -1)]  # leave out missing values (-1)
bfapArrValid += 1e-32  # np.min(bfapArrValid[np.where(bfapArrValid > 0)])  # 1  # add 1 so that range becomes [1, 2]
bfapArrValid = np.log10(bfapArrValid)  # log transform
bfapNormStats = {'median': np.median(bfapArrValid), 'std': stats.mad_std(bfapArrValid)}

# transform binoedp stat
binoedpArr = scalarParamsMat[:, 6]
binoedpArr += 1e-32
binoedpArr = np.log10(binoedpArr)
binoedpNormStats = {'median': np.median(binoedpArr), 'std': stats.mad_std(binoedpArr)}

scalarNormStats = {'values': scalarParamsMat,
                   'median': np.median(scalarParamsMat, axis=0),
                   'std': stats.mad_std(scalarParamsMat, axis=0),
                   'params': scalarParams}

# overwrite stats for bfap and binoedp
scalarNormStats['median'][7] = bfapNormStats['median']
scalarNormStats['std'][7] = bfapNormStats['std']

scalarNormStats['median'][6] = binoedpNormStats['median']
scalarNormStats['std'][6] = binoedpNormStats['std']

np.save(os.path.join(tfrecDir, 'train_scalarparam_norm_stats.npy'), scalarNormStats)

# create additional csv file with normalization statistics
scalarNormStatsDataForDf = {}
for scalar_i in range(len(scalarParams)):
    scalarNormStatsDataForDf['{}_median'.format(scalarParams[scalar_i])] = [scalarNormStats['median'][scalar_i]]
    scalarNormStatsDataForDf['{}_std'.format(scalarParams[scalar_i])] = [scalarNormStats['std'][scalar_i]]
scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)
scalarNormStatsDf.to_csv(os.path.join(tfrecDir, 'train_scalarparam_norm_stats.csv'), index=False)

# save normalization statistics for FDL centroid time series
normStatsFDL = {timeSeries:  {
                              # 'values': timeSeriesFDLMat,
                              'oot_median': np.median(timeSeriesFDLMat[timeSeries]),
                              'oot_std': stats.mad_std(timeSeriesFDLMat[timeSeries])
                              }
                for timeSeries in timeSeriesFDLList}
np.save(os.path.join(tfrecDir, 'train_fdlcentroid_norm_stats.npy'), normStatsFDL)

# create additional csv file with normalization statistics
normStatsFDLDataForDf = {}
for timeSeries in timeSeriesFDLList:
    normStatsFDLDataForDf['{}_oot_median'.format(timeSeries)] = [normStatsFDL[timeSeries]['oot_median']]
    normStatsFDLDataForDf['{}_oot_std'.format(timeSeries)] = [normStatsFDL[timeSeries]['oot_std']]
normStatsFDLDf = pd.DataFrame(data=normStatsFDLDataForDf)
normStatsFDLDf.to_csv(os.path.join(tfrecDir, 'train_fdlcentroid_norm_stats.csv'), index=False)

# save normalization statistics for centroid time series
normStatsCentroid = {timeSeries:  {
                                   # 'values': timeSeriesFDLMat,
                                   'median': np.median(centroidMat[timeSeries]),
                                   'std': stats.mad_std(centroidMat[timeSeries]),
                                   'clip_value': 30
                                   # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
                                   #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
                                   }
                     for timeSeries in centroidList}
for timeSeries in centroidList:
    centroidMatClipped = np.clip(centroidMat[timeSeries], a_max=30, a_min=None)
    clipStats = {
        'median_clip': np.median(centroidMatClipped),
        'std_clip': stats.mad_std(centroidMatClipped)
    }
    normStatsCentroid[timeSeries].update(clipStats)
np.save(os.path.join(tfrecDir, 'train_centroid_norm_stats.npy'), normStatsCentroid)

# create additional csv file with normalization statistics
normStatsCentroidDataForDf = {}
for timeSeries in centroidList:
    normStatsCentroidDataForDf['{}_median'.format(timeSeries)] = [normStatsCentroid[timeSeries]['median']]
    normStatsCentroidDataForDf['{}_std'.format(timeSeries)] = [normStatsCentroid[timeSeries]['std']]
    # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
    normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = 30
    normStatsCentroidDataForDf['{}_median_clip'.format(timeSeries)] = [normStatsCentroid[timeSeries]['median_clip']]
    normStatsCentroidDataForDf['{}_std_clip'.format(timeSeries)] = [normStatsCentroid[timeSeries]['std_clip']]
normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)
normStatsCentroidDf.to_csv(os.path.join(tfrecDir, 'train_centroid_norm_stats.csv'), index=False)

#%% normalize examples in the TFRecords


def normalize_examples(destTfrecDir, srcTfrecFile, normStats, normParams):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  str, destination TFRecord directory for the normalized data
    :param srcTfrecFile: str, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :return:
    """

    # get out-of-transit indices for the local views
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(normParams['num_bins_loc'],
                                                              normParams['nr_transit_durations'])  # same for all TCEs

    # with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:
    with tf.io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:

        # iterate through the source shard
        # record_iterator = tf.python_io.tf_record_iterator(path=srcTfrecFile)
        tfrecord_dataset = tf.data.TFRecordDataset(srcTfrecFile)

        for string_record in tfrecord_dataset.as_numpy_iterator():
        # for string_i, string_record in enumerate(record_iterator):

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # normalize scalar parameters
            tceScalarParams = np.array(example.features.feature['scalar_params'].float_list.value)
            # TODO: normalization statistics perhaps should be computed using the TCE table and after that missing
            #       would be replaced; stellar parameters would have to be updated in the TFRecords
            # log transform with constant translation of bfap and oedp stat
            tceScalarParams[6] += 1e-32
            assert tceScalarParams[6] > 0
            tceScalarParams[6] = np.log10(tceScalarParams[6])

            if tceScalarParams[7] == -1:  # check if bootstrap FA probability value is missing (-1)
                tceScalarParams[7] = normStats['scalar_params']['median'][7]
            else:
                tceScalarParams[7] += 1e-32
                assert tceScalarParams[7] > 0
                tceScalarParams[7] = np.log10(tceScalarParams[7])

            # clip ghost diagnostic by +- 20 * std
            tceScalarParams[-3] = np.clip(tceScalarParams[-3],
                                          -20 * normStatsScalar['std'][-3],
                                          20 * normStatsScalar['std'][-3])
            # hap
            tceScalarParams[-2] = np.clip(tceScalarParams[-2],
                                          -20 * normStatsScalar['std'][-2],
                                          20 * normStatsScalar['std'][-2])

            # median centering and std normalization
            tceScalarParams = (tceScalarParams - normStats['scalar_params']['median']) / \
                              normStats['scalar_params']['std']

            # # for TPS - does not have DV diagnostics (bfap, oedpstat, ghost, ...)
            # tceScalarParams = (tceScalarParams - normStats['scalar_params']['median'][np.array([0, 1, 2, 3, 8, 9])]) / \
            #                   normStats['scalar_params']['std'][np.array([0, 1, 2, 3, 8, 9])]

            # normalize FDL centroid time series
            # get out-of-transit indices for the global views
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(normParams['num_bins_glob'],
                                                                        transitDuration,
                                                                        orbitalPeriod)
            # compute oot global and local flux views std
            glob_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['global_flux_view'].float_list.value)[idxs_nontransitcadences_glob],
                    ddof=1)
            loc_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['local_flux_view'].float_list.value)[idxs_nontransitcadences_loc],
                    ddof=1)
            # center and normalize FDL centroid time series
            glob_centr_fdl_view = np.array(example.features.feature['global_centr_fdl_view'].float_list.value)
            glob_centr_fdl_view = centering_and_normalization(glob_centr_fdl_view,
                                                              normStats['global_centr_fdl_view']['oot_median'],
                                                              normStats['global_centr_fdl_view']['oot_std']
                                                              )
            glob_centr_fdl_view *= glob_flux_view_std
            loc_centr_fdl_view = np.array(example.features.feature['local_centr_fdl_view'].float_list.value)
            loc_centr_fdl_view = centering_and_normalization(loc_centr_fdl_view,
                                                             normStats['local_centr_fdl_view']['oot_median'],
                                                             normStats['local_centr_fdl_view']['oot_std']
                                                             )
            loc_centr_fdl_view *= loc_flux_view_std

            # normalize centroid time series; first clip time series to a maximum value for each view
            glob_centr_view = np.array(example.features.feature['global_centr_view'].float_list.value)
            loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)

            # 1) clipping to physically meaningful distance in arcsec
            glob_centr_view_std_clip = np.clip(glob_centr_view, a_max=normStats['global_centr_view']['clip_value'],
                                               a_min=None)
            glob_centr_view_std_clip = centering_and_normalization(glob_centr_view_std_clip,
                                                                   normStats['global_centr_view']['median_clip'],
                                                                   normStats['global_centr_view']['std_clip'])

            loc_centr_view_std_clip = np.clip(loc_centr_view, a_max=normStats['local_centr_view']['clip_value'],
                                              a_min=None)
            loc_centr_view_std_clip = centering_and_normalization(loc_centr_view_std_clip,
                                                                   normStats['local_centr_view']['median_clip'],
                                                                   normStats['local_centr_view']['std_clip'])

            # 2) no clipping
            glob_centr_view_std_noclip = centering_and_normalization(glob_centr_view,
                                                                   normStats['global_centr_view']['median'],
                                                                   normStats['global_centr_view']['std'])
            loc_centr_view_std_noclip = centering_and_normalization(loc_centr_view,
                                                                   normStats['local_centr_view']['median'],
                                                                   normStats['local_centr_view']['std'])


            # add features to the example in the TFRecord
            normalizedFeatures = {'scalar_params': tceScalarParams,
                                  'global_centr_fdl_view': glob_centr_fdl_view,
                                  'local_centr_fdl_view': loc_centr_fdl_view,
                                  'global_centr_view_std_clip': glob_centr_view_std_clip,
                                  'local_centr_view_std_clip': loc_centr_view_std_clip,
                                  'global_centr_view_std_noclip': glob_centr_view_std_noclip,
                                  'local_centr_view_std_noclip': loc_centr_view_std_noclip
                                  }

            for normalizedFeature in normalizedFeatures:
                example_util.set_float_feature(example, normalizedFeature, normalizedFeatures[normalizedFeature],
                                               allow_overwrite=True)

            # orbitalPeriod = -1 + 2 * (orbitalPeriod - 0.500309) / (720.2339999999999 - 0.500309)
            # example_util.set_float_feature(example, 'tce_period_norm', [orbitalPeriod])

            writer.write(example.SerializeToString())


srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment-labels'
destTfrecDir = srcTfrecDir + '-norm'
os.makedirs(destTfrecDir, exist_ok=True)
srcTfrecFiles = [os.path.join(srcTfrecDir, file) for file in os.listdir(srcTfrecDir) if 'shard' in file]
srcTfrecFiles = [file for file in srcTfrecFiles if not file.endswith('.csv')]

normParams = {'nr_transit_durations': 2 * 4 + 1,  # number of transit durations (2*n+1, n on each side of the transit)
              'num_bins_loc': 201,
              'num_bins_glob': 2001
              }

# load normalization statistics
normStats = {}
normStatsDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment'
# scalar parameters normalization statistics
normStatsScalar = np.load(os.path.join(normStatsDir, 'train_scalarparam_norm_stats.npy'), allow_pickle=True).item()
# FDL centroid normalization statistics
normStatsFDLCentroid = np.load(os.path.join(normStatsDir, 'train_fdlcentroid_norm_stats.npy'), allow_pickle=True).item()
# Centroid normalization statistics
normStatsCentroid = np.load(os.path.join(normStatsDir, 'train_centroid_norm_stats.npy'), allow_pickle=True).item()
normStats['scalar_params'] = {key: normStatsScalar[key] for key in normStatsScalar if key not in 'values'}
normStats.update(normStatsFDLCentroid)
normStats.update(normStatsCentroid)

nProcesses = 15
pool = multiprocessing.Pool(processes=nProcesses)
jobs = [(destTfrecDir, file, normStats, normParams) for file in srcTfrecFiles]
async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

print('Normalization finished.')

#%% Check final preprocessed data

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment-labels-norm'
plotDir = os.path.join(tfrecDir, 'plots_all_views')
os.makedirs(plotDir, exist_ok=True)
tfrecFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if 'shard' in file]

tceIdentifier = 'tce_plnt_num'
views = [
    'global_flux_view',
    'local_flux_view',
    # 'global_flux_odd_view',
    'local_flux_odd_view',
    # 'global_flux_even_view',
    'local_flux_even_view',
    # 'local_flux_oddeven_view_diff',
    # 'local_flux_oddeven_view_diff_dir',
    'global_weak_secondary_view',
    'local_weak_secondary_view',
    'global_centr_view',
    'local_centr_view',
    'global_centr_view_std_clip',
    'local_centr_view_std_clip',
    'global_centr_view_std_noclip',
    'local_centr_view_std_noclip',
    'global_centr_view_medcmaxn',
    'local_centr_view_medcmaxn',
    'global_centr_view_medcmaxn_dir',
    'local_centr_view_medcmaxn_dir',
    # 'global_centr_view_medn',
    # 'local_centr_view_medn',
    'global_centr_fdl_view',
    'local_centr_fdl_view',
]
scalarParams = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat',
                'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat', 'tce_rb_tcount0']
# scalarParams = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens']

tceOfInterest = (8937762, 1)  # (8750094, 1)  # (8611832, 1)
scheme = (5, 4)
basename = 'all_views'  # basename for figures
# probPlot = 1.01  # probability threshold for plotting

for tfrecFile in tfrecFiles:

    # record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecFile)

    for string_record in tfrecord_dataset.as_numpy_iterator():
    # for string_i, string_record in enumerate(record_iterator):

        # if np.random.random() > probPlot:
        #     continue

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        if targetIdTfrec != tceOfInterest[0] or tceIdentifierTfrec != tceOfInterest[1]:
            continue
        # else:
        #     tceFound = True

        scalarParamsTfrec = np.array(example.features.feature['scalar_params'].float_list.value)
        scalarParamsStr = ''
        for scalarParam_i in range(len(scalarParams)):
            # print('{}: {}'.format(scalarParams[scalarParam_i], scalarParamsTfrec[scalarParam_i]))
            if scalarParam_i == 5:
                scalarParamsStr += '\n'
            # if scalarParams[scalarParam_i] == 'boot_fap':
            #     scalarParamsStr += '{}={:.4E}  '.format(scalarParams[scalarParam_i], scalarParamsTfrec[scalarParam_i])
            # else:
            scalarParamsStr += '{}={:.4f}  '.format(scalarParams[scalarParam_i], scalarParamsTfrec[scalarParam_i])

        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        viewsDict = {}
        for view in views:
            viewsDict[view] = np.array(example.features.feature[view].float_list.value)

        f, ax = plt.subplots(scheme[0], scheme[1], figsize=(22, 18))
        k = 0
        views_list = list(viewsDict.keys())
        for i in range(scheme[0]):
            for j in range(scheme[1]):
                if k < len(views_list):
                    ax[i, j].plot(viewsDict[views_list[k]])
                    # ax[i, j].scatter(np.arange(len(viewsDict[views_list[k]])), viewsDict[views_list[k]], s=5)
                    ax[i, j].set_title(views_list[k], pad=20)
                if i == scheme[0] - 1:
                    ax[i, j].set_xlabel('Bin number')
                if j == 0:
                    ax[i, j].set_ylabel('Amplitude')
                k += 1

        f.suptitle('TCE {} {} {}\n{}'.format(targetIdTfrec, tceIdentifierTfrec, labelTfrec, scalarParamsStr))
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(os.path.join(plotDir, '{}_{}_{}_{}.png'.format(targetIdTfrec, tceIdentifierTfrec, labelTfrec,
                                                                   basename)))
        # f.tight_layout(rect=[0, 0.03, 1, 0.95])
        aaa
        plt.close()

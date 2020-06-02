import pandas as pd
import os
import tensorflow as tf
import numpy as np
import itertools
import multiprocessing
from astropy import stats
import matplotlib.pyplot as plt

from src_preprocessing.preprocess import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc, \
    centering_and_normalization
from src_preprocessing.tf_util import example_util

#%% define directories

srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar'
destTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment'
os.makedirs(destTfrecDir, exist_ok=True)

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

#%% create training, validation and test datasets

experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                               'noRobobvetterKOIs.csv')

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

#%% create new TFRecords based on the original ones

# load train, val and test datasets
datasetTblDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/' \
                'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment'
datasetTbl = {dataset: pd.read_csv(os.path.join(datasetTblDir, '{}set.csv'.format(dataset)))
              for dataset in ['train', 'val', 'test']}
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

    with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, shardFilename)) as writer:

        # iterate through TCEs in the shard TCE table
        for tce_i, tce in shardTbl.iterrows():

            # check if TCE is in the source TFRecords TCE table
            foundTce = srcTbl.loc[(srcTbl['target_id'] == tce['target_id']) &
                                  (srcTbl[tceIdentifier] == tce[tceIdentifier])]['shard']

            if len(foundTce) > 0:

                record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(srcTfrecDir, foundTce.values[0]))
                tceIdx = foundTce.index[0]

                tceFoundInTfrecordFlag = False

                for string_i, string_record in enumerate(record_iterator):
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


srcTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar/merged_shards.csv', index_col=0)
srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar'
destTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment'
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
    # iterate through the source shard
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    countExamplesShard = 0
    for string_i, string_record in enumerate(record_iterator):
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

    with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:
        # iterate through the source shard
        record_iterator = tf.python_io.tf_record_iterator(path=srcTfrecFile)

        for string_i, string_record in enumerate(record_iterator):
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


srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment'
destTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment-labels'
os.makedirs(destTfrecDir, exist_ok=True)

# dispositions coming from the experiment TCE table
experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                               'noRobobvetterKOIs.csv')
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
    # iterate through the source shard
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)
    countExamplesShard = 0
    for string_i, string_record in enumerate(record_iterator):
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

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment'
tfrecTrainFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if file.startswith('train-shard')]

tceIdentifier = 'tce_plnt_num'

# scalar parameters are also present in the TCE table, since that was where they came from, but to keep things more
# flexible, we will also take them from the TFRecords

# scalarParams = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens', 'wst_robstat',
#                 'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']
scalarParams = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat',
                'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat']
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

# FDL centroid time series normalization statistics parameters
centroidList = ['global_centr_view', 'local_centr_view']
centroidMat = {timeSeries: [] for timeSeries in centroidList}

for tfrec_i, tfrecFile in enumerate(tfrecTrainFiles):

    print('Getting data from {} ({} %)'.format(tfrecFile.split('/')[-1], tfrec_i / len(tfrecTrainFiles) * 100))

    # iterate through the shard
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # tceLabel = example.features.feature['label'].bytes_list.value[0].decode("utf-8")
        #
        # if tceLabel != 'NTP':
        #     continue

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

# save normalization statistics for the scalar parameters
scalarParamsMat = np.array(scalarParamsMat)
scalarNormStats = {'values': scalarParamsMat,
                   'median': np.median(scalarParamsMat, axis=0),
                   'std': stats.mad_std(scalarParamsMat, axis=0),
                   'params': scalarParams}
np.save(os.path.join(tfrecDir, 'train_scalarparam_norm_stats.npy'), scalarNormStats)
# create additional csv file with normalization statistics
scalarNormStatsDataForDf = {}
for scalar_i in range(len(scalarParams)):
    scalarNormStatsDataForDf['{}_median'.format(scalarParams[scalar_i])] = [scalarNormStats['median'][scalar_i]]
    scalarNormStatsDataForDf['{}_std'.format(scalarParams[scalar_i])] = [scalarNormStats['std'][scalar_i]]
scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)
scalarNormStatsDf.to_csv(os.path.join(tfrecDir, 'train_scalarparam_norm_stats.csv'), index=False)

# save normalization statistics for FDL centroid time series
normStatsFDL = {timeSeries:  {# 'values': timeSeriesFDLMat,
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
normStatsCentroid = {timeSeries:  {# 'values': timeSeriesFDLMat,
                                   'median': np.median(centroidMat[timeSeries]),
                                   'std': stats.mad_std(centroidMat[timeSeries]),
                                   'clip_value': np.percentile(centroidMat[timeSeries], 75) +
                                                 1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
                                   }
                     for timeSeries in centroidList}
np.save(os.path.join(tfrecDir, 'train_centroid_norm_stats.npy'), normStatsCentroid)
# create additional csv file with normalization statistics
normStatsCentroidDataForDf = {}
for timeSeries in centroidList:
    normStatsCentroidDataForDf['{}_median'.format(timeSeries)] = [normStatsCentroid[timeSeries]['median']]
    normStatsCentroidDataForDf['{}_std'.format(timeSeries)] = [normStatsCentroid[timeSeries]['std']]
    normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)
normStatsCentroidDf.to_csv(os.path.join(tfrecDir, 'train_centroid_norm_stats.csv'), index=False)

# # plot boxplots of centroid local and global views in the training set
# print('Generating boxplots...')
# f, ax = plt.subplots()
# ax.boxplot([centroidMat['global_centr_view'], centroidMat['local_centr_view']], bootstrap=None, meanline=True,
#            showmeans=True)
# ax.set_title('Non-normalized centroid time series')
# ax.set_yscale('log')
# ax.set_ylabel('Value')
# ax.set_xticklabels(['Global view', 'Local view'])
# f.savefig('/home/msaragoc/Downloads/hist_nonnormalized_centroids.png')
# # plt.show()

#%% Checking values for centroid views

# print(len(np.where(centroidMat['global_centr_view'], np.percentile(centroidMat['global_centr_view'], 75))))

badTces = []
thr = 1000  # 5844.65576171875  # 99.9 percentile

label = 'AFP'
# centroidDict = {'global_centr_view': [], 'local_centr_view': []}

for tfrec_i, tfrecFile in enumerate(tfrecTrainFiles):

    print('Getting data from {} ({} %)'.format(tfrecFile.split('/')[-1], tfrec_i / len(tfrecTrainFiles) * 100))

    # iterate through the shard
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceLabel = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        if tceLabel != label:
            continue

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        # get centroid time series data

        timeSeriesTce = np.array(example.features.feature['global_centr_view'].float_list.value)
        # centroidDict['global_centr_view'].append(np.min(timeSeriesTce))

        if np.any(np.array(timeSeriesTce) > thr):
            badTces.append((targetIdTfrec, tceIdentifierTfrec))

        # timeSeriesTce = np.array(example.features.feature['local_centr_view'].float_list.value)
        # centroidDict['local_centr_view'].append(np.min(timeSeriesTce))

print(len(badTces))
print(badTces)

# numOutliers = {}
# for timeSeries in centroidDict:
#     q75 = np.percentile(centroidDict[timeSeries], 75)
#     q25 = np.percentile(centroidDict[timeSeries], 25)
#
#     iqr = q75 - q25
#
#     numOutliers[timeSeries] = len(np.where(centroidDict[timeSeries] > 1.5 * iqr)[0])
#
# # plot boxplots of centroid local and global views in the training set
# print('Generating boxplots...')
# f, ax = plt.subplots()
# ax.boxplot([centroidDict['global_centr_view'], centroidDict['local_centr_view']], bootstrap=None, meanline=True,
#            showmeans=True)
# ax.set_title('Non-normalized centroid time series\n{} Num outliers = {}'.format(label, numOutliers[timeSeries]))
# ax.set_yscale('log')
# ax.set_ylabel('Median value')
# ax.set_xticklabels(['Global view', 'Local view'])
# f.savefig('/home/msaragoc/Downloads/hist_nonnormalized_centroids_{}.png'.format(label))
# plt.show()

#%% normalize examples in the TFRecords


def normalize_examples(destTfrecDir, srcTfrecFile, normStats):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  str, destination TFRecord directory for the normalized data
    :param srcTfrecFile: str, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :return:
    """

    # get out-of-transit indices for the local views
    nr_transit_durations = 2 * 4 + 1  # number of transit durations (2*n+1, n on each side of the transit)
    num_bins_loc = 201
    num_bins_glob = 2001
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(num_bins_loc, nr_transit_durations)  # same for all TCEs

    with tf.python_io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:
        # iterate through the source shard
        record_iterator = tf.python_io.tf_record_iterator(path=srcTfrecFile)

        for string_i, string_record in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # normalize scalar parameters
            tceScalarParams = np.array(example.features.feature['scalar_params'].float_list.value)
            # print(example.features.feature['target_id'].int64_list.value, tceScalarParams)
            # aaaaa
            tceScalarParams = (tceScalarParams - normStats['scalar_params']['median']) / \
                              normStats['scalar_params']['std']
            # tceScalarParams = (tceScalarParams - normStats['scalar_params']['median'][np.array([0,1,2,3,8,9])]) / \
            #                   normStats['scalar_params']['std'][np.array([0,1,2,3,8,9])]

            # normalize FDL centroid time series
            # get out-of-transit indices for the global views
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(num_bins_glob, transitDuration, orbitalPeriod)
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
            glob_centr_view[np.where(glob_centr_view > normStats['global_centr_view']['clip_value'])] = \
                normStats['global_centr_view']['clip_value']
            glob_centr_view = centering_and_normalization(glob_centr_view,
                                                          normStats['global_centr_view']['median'],
                                                          normStats['global_centr_view']['std'])
            loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)
            loc_centr_view[np.where(loc_centr_view > normStats['local_centr_view']['clip_value'])] = \
                normStats['local_centr_view']['clip_value']
            loc_centr_view = centering_and_normalization(loc_centr_view,
                                                         normStats['local_centr_view']['median'],
                                                         normStats['local_centr_view']['std'])

            normalizedFeatures = {'scalar_params': tceScalarParams,
                                  'global_centr_fdl_view': glob_centr_fdl_view,
                                  'local_centr_fdl_view': loc_centr_fdl_view,
                                  'global_centr_view': glob_centr_view,
                                  'local_centr_view': loc_centr_view
                                  }

            for normalizedFeature in normalizedFeatures:
                example_util.set_float_feature(example, normalizedFeature, normalizedFeatures[normalizedFeature],
                                               allow_overwrite=True)

            writer.write(example.SerializeToString())


srcTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment-labels'
destTfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_experiment-labels-norm'
os.makedirs(destTfrecDir, exist_ok=True)
srcTfrecFiles = [os.path.join(srcTfrecDir, file) for file in os.listdir(srcTfrecDir) if 'shard' in file]

# load normalization statistics
normStats = {}
normStatsDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment'
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
jobs = [(destTfrecDir, file, normStats) for file in srcTfrecFiles]
async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

print('Normalization finished.')

#%% Check final preprocessed data

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment-labels-norm'
plotDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/plots_after_norm'
os.makedirs(plotDir, exist_ok=True)
tfrecFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if 'shard' in file]

tceIdentifier = 'tce_plnt_num'
views = ['global_centr_view', 'local_centr_view', 'global_centr_view_medcmaxn', 'local_centr_view_medcmaxn',
         'global_centr_view_medn', 'local_centr_view_medn', 'global_centr_fdl_view', 'local_centr_fdl_view',
         'global_flux_view', 'local_flux_view', 'global_flux_odd_view', 'local_flux_odd_view', 'global_flux_even_view',
         'local_flux_even_view', 'global_weak_secondary_view', 'local_weak_secondary_view']

tceOfInterest = (5130369, 1)
for tfrecFile in tfrecFiles:

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        if targetIdTfrec != tceOfInterest[0] or tceIdentifierTfrec != tceOfInterest[1]:
            continue
        else:
            tceFound = True

        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        tceScalarParams = np.array(example.features.feature['scalar_params'].float_list.value)

        for view in views:
            viewTimeseries = example.features.feature[view].float_list.value
            f, ax = plt.subplots()
            ax.scatter(np.arange(len(viewTimeseries)), viewTimeseries, c='b', s=5)
            ax.set_xlabel('Bin Number')
            ax.set_ylabel('Amplitude')
            ax.set_title('{}'.format(view))
            f.savefig(os.path.join(plotDir, '{}.png'.format(view)))

        print('Scalar parameters: ', tceScalarParams)

        if tceFound:
            aaaaa

#%% normalize scalar parameters in TFRecords using computed statistics based on the experiment TCE table
#
# srcDir = ''
# destDirFp = ''
#
# srcTfrecs = [os.path.join(srcDir, file) for file in os.listdir(srcDir) if 'shard' in file]
#
# os.makedirs(destDirFp, exist_ok=True)
#
# scalarNormStats = np.load(os.path.join(newTfrecDir, 'train_scalarparam_norm_stats.npy'), allow_pickle=True).item()
#
# for srcTfrec in srcTfrecs:
#
#     record_iterator = tf.python_io.tf_record_iterator(path=srcTfrec)
#
#     with tf.python_io.TFRecordWriter(os.path.join(destDirFp, os.path.basename(srcTfrec))) as writer:
#         for string_record in enumerate(record_iterator):
#             example = tf.train.Example()
#             example.ParseFromString(string_record)
#
#             scalarParams = np.array(example.features.feature['scalar_params'].float_list.value)
#
#             scalarParams = (scalarParams - scalarNormStats['median']) / scalarNormStats['std']
#
#             example_util.set_feature(example, 'scalar_params', list(scalarParams), allow_overwrite=True)
#
#             writer.write(example.SerializeToString())
#
#
# #%% normalize FDL centroid time-series in the TFRecords using computed normalization statistics based on the experiment
# # TCE table
#
# srcDir = ''
# destDirFp = ''
#
# srcTfrecs = [os.path.join(srcDir, file) for file in os.listdir(srcDir) if 'shard' in file]
#
# os.makedirs(destDirFp, exist_ok=True)
#
# timeSeriesList = ['global_view_centr_fdl', 'local_view_centr_fdl']
# timeSeriesNormStats = np.load(os.path.join(newTfrecDir, 'train_fdlcentroid_norm_stats.npy'), allow_pickle=True).item()
#
# # get out-of-transit indices for the local and global views
# nr_transit_durations = 2 * 4 + 1  # number of transit durations (2*n+1, n on each side of the transit)
# num_bins_loc = 201
# num_bins_glob = 2001
# idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(num_bins_loc, nr_transit_durations)  # same for all TCEs
#
# for srcTfrec in srcTfrecs:
#
#     record_iterator = tf.python_io.tf_record_iterator(path=srcTfrec)
#
#     with tf.python_io.TFRecordWriter(os.path.join(destDirFp, os.path.basename(srcTfrec))) as writer:
#         for string_record in enumerate(record_iterator):
#             example = tf.train.Example()
#             example.ParseFromString(string_record)
#
#             glob_flux_view = np.array(example.features.feature['global_view'].float_list.value)
#             loc_flux_view = np.array(example.features.feature['local_view'].float_list.value)
#
#             transitDuration = example.features.feature['tce_duration'].float_list.value[0]
#             orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
#             idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(num_bins_glob, transitDuration, orbitalPeriod)
#
#             for timeSeries in timeSeriesList:
#                 timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
#                 if 'global' in timeSeries:
#                     fluxOotStd = np.std(glob_flux_view[idxs_nontransitcadences_glob], ddof=1)
#                     timeSeriesTce = (timeSeriesTce - timeSeriesNormStats[timeSeries]['oot_median']) /\
#                                     timeSeriesNormStats[timeSeries]['oot_std'] * fluxOotStd
#                 else:
#                     fluxOotStd = np.std(loc_flux_view[idxs_nontransitcadences_loc], ddof=1)
#                     timeSeriesTce = (timeSeriesTce - timeSeriesNormStats[timeSeries]['oot_median']) /\
#                                     timeSeriesNormStats[timeSeries]['oot_std'] * fluxOotStd
#
#                 # timeSeriesTce_meanoot = example.features.feature['{}_meanoot'.format(timeSeries)].float_list.value[0]
#                 # timeSeriesTce_meanoot = (timeSeriesTce_meanoot - timeSeriesNormStats[timeSeries]['oot_median']) / \
#                 #                         timeSeriesNormStats[timeSeries]['oot_std'] * fluxOotStd
#                 # timeSeriesTce_stdoot = example.features.feature['{}_stdoot'.format(timeSeries)].float_list.value[0]
#                 # timeSeriesTce_stdoot = timeSeriesTce_stdoot / timeSeriesNormStats[timeSeries]['oot_std'] * \
#                 #                        fluxOotStd
#                 #
#                 example_util.set_float_feature(example, timeSeries, timeSeriesTce, allow_overwrite=True)
#                 # example_util.set_float_feature(example, '{}_meanoot'.format(timeSeries), [timeSeriesTce_meanoot],
#                 #                                allow_overwrite=True)
#                 # example_util.set_float_feature(example, '{}_stdoot'.format(timeSeries), [timeSeriesTce_stdoot],
#                 #                                allow_overwrite=True)
#
#             writer.write(example.SerializeToString())

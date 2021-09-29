"""
Update labels for items in a TFRecord dataset given a set of dispositions. Iterates through examples in each TFRecord
file and updates their dispositions (`label` feature). It assumes examples can be uniquely identified by 'target_id' and
`tceIdentifier`.
"""

# 3rd party
from pathlib import Path
import multiprocessing
import tensorflow as tf
import numpy as np
import pandas as pd

# local
from src_preprocessing.tf_util import example_util


def update_labels(destTfrecDir, srcTfrecFile, tceTbl, tceIdentifier, omitMissing=True):
    """ Update example label field in the TFRecords.

    :param destTfrecDir: Path, destination TFRecord directory with data following the updated labels given
    :param srcTfrecFile: Path, source TFRecord directory
    :param tceTbl: pandas DataFrame, TCE table with labels for the TCEs
    :param tceIdentifier: str, TCE identifier in the TCE table. Either `tce_plnt_num` or `oi`, it depends also on the
    columns in the table
    :param omitMissing: bool, if True it skips missing TCEs in the TCE table
    :return:
    """

    with tf.io.TFRecordWriter(str(destTfrecDir / srcTfrecFile.name)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(srcTfrecFile))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
            if tceIdentifier == 'tce_plnt_num':
                targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
            else:
                # targetIdTfrec = example.features.feature['target_id'].float_list.value[0]
                targetIdTfrec = round(example.features.feature['target_id'].float_list.value[0], 2)

            foundTce = tceTbl.loc[(tceTbl['target_id'] == targetIdTfrec) &
                                  (tceTbl[tceIdentifier] == tceIdentifierTfrec)]

            if len(foundTce) > 0:
                tceLabel = foundTce['label'].values[0]
                example_util.set_feature(example, 'label', [tceLabel], allow_overwrite=True)
                writer.write(example.SerializeToString())
            else:
                if omitMissing:
                    print(f'TCE {targetIdTfrec}-{tceIdentifierTfrec} not found in the TCE table.')
                    continue

                raise ValueError(f'TCE {targetIdTfrec}-{tceIdentifierTfrec} not found in the TCE table')


# %%

# input parameters
# source TFRecord directory
srcTfrecDir = Path(
    '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_starshuffle_experiment')
tceIdentifier = 'tce_plnt_num'  # unique identifier for TCEs (+ 'target_id')
destTfrecDirName = '-labels'  # destination TFRecords directory name
destTfrecDir = srcTfrecDir.parent / f'{srcTfrecDir.name}_{destTfrecDirName}'
destTfrecDir.mkdir(exist_ok=True)
nProcesses = 15  # number of processes used in parallel
omitMissing = True  # skip missing TCEs in the source TFRecords

# dispositions coming from the experiment TCE table
experimentTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                               'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                               'nomissingval_rmcandandfpkois_norogues.csv')

# get source TFRecord file paths
srcTfrecFiles = [file for file in srcTfrecDir.iterdir() if 'shard' in file.stem]

pool = multiprocessing.Pool(processes=nProcesses)
jobs = [(destTfrecDir, file, experimentTceTbl, tceIdentifier, omitMissing) for file in srcTfrecFiles]
async_results = [pool.apply_async(update_labels, job) for job in jobs]
pool.close()
for async_result in async_results:
    async_result.get()

print('Labels updated.')

# %% confirm that the labels are correct

# get destination TFRecord file paths
tfrecFiles = [file for file in destTfrecDir.iterdir() if 'shard' in file.stem]
countExamples = []
for tfrecFile in tfrecFiles:

    countExamplesShard = 0

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        if tceIdentifier == 'tce_plnt_num':
            tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        else:
            tceIdentifierTfrec = round(example.features.feature[tceIdentifier].float_list.value[0], 2)
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        tceLabelTbl = experimentTceTbl.loc[(experimentTceTbl['target_id'] == targetIdTfrec) &
                                           (experimentTceTbl[tceIdentifier] == tceIdentifierTfrec)]['label'].values[0]

        if tceLabelTbl != labelTfrec:
            countExamplesShard += 1

    countExamples.append(countExamplesShard)

assert np.sum(countExamples) == 0

"""
Create new TFRecord files based on a set of original TFRecord files and data set tables. Script used to create
training, validation, test and predict TFRecord datasets. E.g., use as source directory the TFRecords created from
preprocessing light curve data (Kepler, TESS, ...).

The original TFRecords are named 'shard-xxx-of-xxx...'. The TFRecord files created for the training, validation,
test, and predict sets records are identified by their prefix: '{dataset}-shard-xxx...'

Data set prefixes: 'train', 'val', 'test', 'predict'

Input:
    - source TFRecord data set (shard files)
    - shard TCE table

The shard TCE table should contain the following columns:
    - (FIRST COLUMN) order of example in the TFRecord file (i.e., shard described in the 'shard' column).
    - 'uid': Unique identifier for the example.
    - 'shard': shard filename where the example is stored.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import multiprocessing
import tensorflow as tf
import itertools
import numpy as np
import yaml

# local
from src_preprocessing.lc_preprocessing.utils_manipulate_tfrecords import create_shard
from utils.utils_dataio import is_yamlble

if __name__ == '__main__':

    # get the configuration parameters
    path_to_yaml = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/config_create_new_tfrecords.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # source TFRecord directory
    srcTfrecDir = Path(config['src_tfrec_dir'])

    # load data set TCE tables
    datasetTblDir = Path(config['split_tbls_dir'])
    datasetTbl = {dataset: pd.read_csv(datasetTblDir / f'{dataset}set.csv') for dataset in config['datasets']}

    # load merged shards table
    srcTbl = pd.read_csv(config['shards_tbl_fp'], index_col=0)

    # create destination directory
    destTfrecDir = Path(config['dest_tfrec_dir'])
    destTfrecDir.mkdir(exist_ok=True)

    # get number of items per dataset table
    numTces = {}
    for dataset in datasetTbl:
        numTces[dataset] = len(datasetTbl[dataset])
        print(datasetTbl[dataset]['label'].value_counts())
        print(f'Number of TCEs in {dataset} set: {len(datasetTbl[dataset])}')

    # define number of examples per shard
    numTcesPerShard = np.min([config['n_examples_per_shard']] + list(numTces.values()))
    numShards = {dataset: int(numTces[dataset] / numTcesPerShard) for dataset in datasetTbl}

    assert np.all(list(numShards.values()))

    print(f'Number of shards per dataset: {numShards}')

    # create pairs of shard filename and shard TCE table
    shardFilenameTuples = []
    for dataset in datasetTbl:
        shardFilenameTuples.extend(
            list(itertools.product([dataset], range(numShards[dataset]), [numShards[dataset] - 1])))
    shardFilenames = ['{}-shard-{:05d}-of-{:05d}'.format(*shardFilenameTuple) for shardFilenameTuple in
                      shardFilenameTuples]

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

    pool = multiprocessing.Pool(processes=config['nProcesses'])
    jobs = [shardTuple + (srcTbl, srcTfrecDir, destTfrecDir, config['omitMissing'], config['verbose'])
            for shardTuple in shardTuples]
    async_results = [pool.apply_async(create_shard, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    print('TFRecord dataset created.')

    # check the number of Examples in the TFRecord shards and that each TCE example for a given dataset is in the
    # TFRecords
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

            example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

            foundTce = datasetTbl[dataset].loc[datasetTbl[dataset]['uid'] == example_uid]

            if len(foundTce) == 0:
                raise ValueError(f'Example {example_uid} not found in the dataset tables.')

            countExamplesShard += 1
        countExamples.append(countExamplesShard)

    print(f'Finished checking for examples in the dataset tables that are missing from the TFRecord shards.')

    json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
    with open(destTfrecDir / 'config_create_new_tfrecords.yaml', 'w') as preproc_run_file:
        yaml.dump(json_dict, preproc_run_file)

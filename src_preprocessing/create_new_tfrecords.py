"""
Create new TFRecord files based on a set of previous TFRecord files. Script used to create training, validation,
test and predict TFRecord datasets. E.g., use as source directory the TFRecords created from preprocessing light curve
data (Kepler, TESS, ...). These original TFRecords will be named 'shard-xxx-of-xxx...'. TFRecords fed as input to the
model are selected as training, validation, test and predict records depending on their prefix: '{dataset}-shard-xxx...'

dataset = {'train', 'val', 'test', 'predict'}

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
from src_preprocessing.utils_manipulate_tfrecords import create_shard
# from paths import path_main
from utils.utils_dataio import is_yamlble

if __name__ == '__main__':

    # get the configuration parameters
    path_to_yaml = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/config_create_new_tfrecords.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # %%

    # source TFRecord directory
    srcTfrecDir = Path(config['src_tfrec_dir'])

    # unique identifier for TCEs (+ 'target_id')
    # tceIdentifier = 'tce_plnt_num'

    # %% load train, val and test datasets; the keys of the `datasetTbl` dictionary determines the prefix name of the new
    # TFRecords files

    datasetTblDir = Path(config['split_tbls_dir'])
    datasetTbl = {dataset: pd.read_csv(datasetTblDir / f'{dataset}set.csv') for dataset in config['datasets']}


    # Filter out items from the datasets that we do not want to have in the final TFRecords

    # # for AUM experiments
    # datasetTbl['train'] = pd.concat([datasetTbl['train'], datasetTbl['val'], datasetTbl['test']], axis=0,
    #                                 ignore_index=True)
    # del datasetTbl['test']
    # del datasetTbl['val']
    # datasetTbl['predict'] = pd.concat([datasetTbl['predict'],
    #                                    datasetTbl['train'].loc[datasetTbl['train']['label'] == 'NTP']],
    #                                   axis=0, ignore_index=True)
    # datasetTbl['train'] = datasetTbl['train'].loc[datasetTbl['train']['label'].isin(['PC', 'AFP'])]
    # for dataset in datasetTbl:
    #     datasetTbl[dataset]['uid'] = \
    #         datasetTbl[dataset][['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
    #                                                                                           x['tce_plnt_num']),
    #                                                                  axis=1)

    # get only TCEs with tce_plnt_num = 1
    # datasetTbl = {dataset: datasetTbl[dataset].loc[datasetTbl[dataset]['tce_plnt_num'] == 1] for dataset in datasetTbl}

    # get TCEs not used for training nor evaluation
    # datasetTbl = {'predict': pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-22-2021/exofop_toilists_nomissingpephem_sectors.csv')
    #               }
    # # remove rogue TCEs
    # datasetTbl['predict'] = datasetTbl['predict'].loc[datasetTbl['predict']['tce_rogue_flag'] == 0]
    # # remove CONFIRMED KOIs
    # datasetTbl['predict'] = datasetTbl['predict'].loc[datasetTbl['predict']['koi_disposition'] != 'CONFIRMED']
    # # remove CFP and CFA KOIs
    # datasetTbl['predict'] = datasetTbl['predict'].loc[~((datasetTbl['predict']['fpwg_disp_status'] == 'CERTIFIED FP') |
    #                         (datasetTbl['predict']['fpwg_disp_status'] == 'CERTIFIED FA'))]
    # # remove non-KOIs
    # datasetTbl['predict'] = datasetTbl['predict'].loc[~(datasetTbl['predict']['kepoi_name'].isna())]

    # %% Create new TFRecords by setting the number of items per TFRecord

    # input parameters
    srcTbl = pd.read_csv(srcTfrecDir / 'merged_shards.csv', index_col=0)
    destTfrecDir = srcTfrecDir.parent / f'{srcTfrecDir.name}-{config["destTfrecDirName"]}'
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
    jobs = [shardTuple + (srcTbl, srcTfrecDir, destTfrecDir, config['omitMissing']) for shardTuple in shardTuples]
    async_results = [pool.apply_async(create_shard, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    print('TFRecord dataset created.')

    # %% check the number of Examples in the TFRecord shards and that each TCE example for a given dataset is in the
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

            # if tceIdentifier == 'tce_plnt_num':
            #     tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
            # else:
            #     tceIdentifierTfrec = round(example.features.feature[tceIdentifier].float_list.value[0], 2)
            #
            # targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

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

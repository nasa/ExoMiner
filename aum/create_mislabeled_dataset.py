"""
Create a mislabeled noise dataset.
"""

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import yaml
import shutil

# local
from src_preprocessing.tf_util import example_util
from utils.utils_dataio import is_yamlble


def switch_labels_tfrecord_shard_from_table(src_data_fp, dest_data_dir, flip_tbl, noise_label='MISLABELED'):
    """ Switch labels for examples in TFRecord shard to pseudo new class based on a table.

    :param src_data_fp: Path, file path to source TFRecord shard
    :param dest_data_dir: Path, destination data directory to store new TFRecord files
    :param flip_tbl: pandas DataFrame, table used to flip labels
    :param noise_label: str, label given to examples that switch to the pseudo new class

    :return:
    """

    with tf.io.TFRecordWriter(str(dest_data_dir / src_data_fp.name)) as writer:
        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # check if example is in flip table
            target_id = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]

            tce_found = flip_tbl.loc[(flip_tbl['target_id'] == target_id) & (flip_tbl['tce_plnt_num'] == tce_plnt_num)]

            if len(tce_found) == 0:
                flip = False
            else:
                flip = tce_found['label'].values[0] == noise_label

            if flip:
                # overwrite label
                example_util.set_bytes_feature(example, 'label', [noise_label], allow_overwrite=True)

            writer.write(example.SerializeToString())


def create_tfrec_dataset_tbl(tfrec_fps, features):
    """ Creates TFRecord dataset table by iterating through examples in the TFRecord shards. The table contains the
    example ID ('target_id' + 'tce_plnt_num'), its label ('label'), and the TFRecord shard name it belongs to
    ('shard_name').

    :param tfrec_fps: list, file paths to shards in TFRecord dataset
    :param features: dict, features to extract for each example from the TFRecord dataset
    :return:
        tfrec_tbl: pandas DataFrame, TFRecord dataset table
    """
    data_to_tbl = []
    for tfrec_fp in tfrec_fps:

        tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

        for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            feats_example = []
            for feature_name in features:

                if features[feature_name]['dtype'] == 'int64':
                    feat_example = example.features.feature[feature_name].int64_list.value[0]
                elif features[feature_name]['dtype'] == 'float':
                    feat_example = example.features.feature[feature_name].float_list.value[0]
                elif features[feature_name]['dtype'] == 'str':
                    feat_example = example.features.feature[feature_name].bytes_list.value[0].decode('utf-8')
                else:
                    raise ValueError(f'Data type {features[feature_name]["dtype"]} not expected.')

                feats_example.append(feat_example)

            feats_example += [tfrec_fp.name, example_i]

            data_to_tbl.append(feats_example)

    tfrec_tbl = pd.DataFrame(data=data_to_tbl, columns=list(features.keys()) + ['shard_name', 'example_i'])

    return tfrec_tbl


def create_flip_tbl(tfrec_tbl, rnd_seed, switch_frac, noise_label='MISLABELED'):
    """ Creates table from TFRecord dataset table where a fraction of PCs and AFPs is randomly selected and their labels
    are switched to noise class label.

    :param tfrec_tbl: pandas DataFrame, TFRecord dataset table
    :param rnd_seed: float, random seed used to instantiate rng that controls random selection of examples whose labels
    :param noise_label: str, label given to examples that switch to the pseudo new class
    :param switch_frac: dict, fraction of each category to switch to the mislabeled pseudo class
    are switched
    :return:
        flip_tbl: pandas DataFrame, table used to flip labels
    """

    rng = np.random.default_rng(seed=rnd_seed)

    # choose examples from each category to switch to noise class
    for switch_frac_cat in switch_frac:
        idx_cat_switch = tfrec_tbl.loc[tfrec_tbl['label'] ==
                                       switch_frac_cat].sample(frac=switch_frac[switch_frac_cat],
                                                               replace=False,
                                                               random_state=rng.bit_generator).index
        tfrec_tbl.loc[idx_cat_switch, 'label'] = noise_label

    return tfrec_tbl


if __name__ == '__main__':

    path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/aum/config_dataset.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # initialize rng
    rng = np.random.default_rng(seed=config['rnd_seed'])

    # results directory
    res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/') / \
              f'run_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    res_dir.mkdir(exist_ok=True)

    # save the YAML file with training-evaluation parameters that are YAML serializable
    json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
    with open(res_dir / 'dataset_params.yaml', 'w') as yml_file:
        yaml.dump(json_dict, yml_file)

    # create TFRecord dataset table
    tfrec_dir = Path(config['src_tfrec_dir'])
    tfrec_fps = [fp for fp in tfrec_dir.iterdir() if fp.suffix != '.csv' and 'shard' in fp.stem]

    features = {
        'target_id': {'dtype': 'int64'},
        'tce_plnt_num': {'dtype': 'int64'},
        'label': {'dtype': 'str'},

    }
    tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=features)
    tfrec_tbl.to_csv(tfrec_dir / 'tfrec_tbl.csv', index=False)

    tfrec_tbl['label_changed_to_mislabeled'] = False
    tfrec_tbl['label_changed_to_other_class'] = False
    tfrec_tbl['label_before'] = tfrec_tbl['label']

    # get only training set examples
    train_set = tfrec_tbl.loc[tfrec_tbl['shard_name'].str.contains('train')]
    # get only PCs and AFPs to build training set
    train_set = train_set.loc[train_set['label'] != 'NTP']

    # # inject stochastic noise in the PC and AFP populations in the training set; switch labels for some PCs and AFP
    # train_set['label_changed_to_other_class'] = False
    #
    # switch_pcs = train_set.loc[train_set['label'] == 'PC'].sample(frac=config['mislabeling_rate'],
    #                                                               replace=False,
    #                                                               random_state=rng.bit_generator)
    # switch_afps = train_set.loc[train_set['label'] == 'AFP'].sample(frac=config['mislabeling_rate'],
    #                                                                 replace=False,
    #                                                                 random_state=rng.bit_generator)
    # train_set.loc[switch_pcs.index, 'label'] = 'AFP'
    # train_set.loc[switch_pcs.index, 'label_changed_to_other_class'] = True
    # train_set.loc[switch_afps.index, 'label'] = 'PC'
    # train_set.loc[switch_afps.index, 'label_changed_to_other_class'] = True
    # train_set.to_csv(res_dir / 'train_set_labels_switched.csv', index=False)

    # get PCs and AFPs from the training set to build mislabeled pseudo-class
    pc_train_set = train_set.loc[train_set['label'] == 'PC']
    afp_train_set = train_set.loc[train_set['label'] == 'AFP']
    n_pcs_trainset, n_afps_trainset = len(pc_train_set), len(afp_train_set)
    # # randomly sample same number of AFPs as PCs
    # afp_train_set_chosen = afp_train_set.sample(n=len(pc_train_set),
    #                                                                 replace=False,
    #                                                                 random_state=rng.bit_generator)
    # afp_train_set_not_chosen = afp_train_set.loc[~afp_train_set.index.isin(afp_train_set_chosen.index)]
    # afp_train_set = afp_train_set_chosen.copy(deep=True)

    # split these categories into N splits
    pc_train_set_split = np.array_split(pc_train_set, config['n_splits'])
    afp_train_set_split = np.array_split(afp_train_set, config['n_splits'])

    # create different iterations by combining these splits
    # each combination has their examples' labels switched to MISLABELED pseudo-class
    train_set_splits = [pd.concat(split_comb)
                        for split_comb in itertools.product(pc_train_set_split, afp_train_set_split)]

    # create the datasets for the different iterations by combining the PC/AFP splits with the rest of the training set
    n_runs = len(train_set_splits)
    dataset_runs = [train_set.copy(deep=True) for run_i in range(n_runs)]
    for run_i, comb in enumerate(train_set_splits):
        # dataset_runs[run_i].loc[comb.index, 'label'] = config['noise_label']  # set examples to mislabeled class
        # dataset_runs[run_i].loc[comb.index, 'label_changed_to_mislabeled'] = True

        switch_pcs = dataset_runs[run_i].loc[dataset_runs[run_i]['label'] == 'PC'].sample(
            frac=config['mislabeling_rate'],
            replace=False,
            random_state=rng.bit_generator)
        switch_afps = dataset_runs[run_i].loc[dataset_runs[run_i]['label'] == 'AFP'].sample(
            frac=config['mislabeling_rate'] * n_pcs_trainset / n_afps_trainset,
            # adjust label noise injection rate to account for more AFPs than PCs
            replace=False,
            random_state=rng.bit_generator)
        dataset_runs[run_i].loc[switch_pcs.index, 'label'] = 'AFP'
        dataset_runs[run_i].loc[switch_pcs.index, 'label_changed_to_other_class'] = True
        dataset_runs[run_i].loc[switch_afps.index, 'label'] = 'PC'
        dataset_runs[run_i].loc[switch_afps.index, 'label_changed_to_other_class'] = True

    # save each dataset run as csv file
    tbl_res_dir = res_dir / 'trainset_runs'
    tbl_res_dir.mkdir(exist_ok=True)
    for run_i, dataset_run in enumerate(dataset_runs):
        dataset_run.to_csv(tbl_res_dir / f'trainset_run{run_i}.csv', index=False)

    # create TFRecord dataset from these splits
    tfrecord_res_dir = res_dir / 'tfrecords'
    tfrecord_res_dir.mkdir(exist_ok=True)
    train_tfrec_fps = [fp for fp in tfrec_fps if fp.name.startswith('train')]
    dataset_runs = [dataset_run.set_index(keys=['shard_name', 'example_i']) for dataset_run in dataset_runs]
    for run_i, dataset_run in enumerate(dataset_runs):

        print(f'Creating training set for run {run_i}...')

        dataset_run_dir = tfrecord_res_dir / f'run{run_i}'
        dataset_run_dir.mkdir(exist_ok=True)

        # # copy validation and test TFRecord shards
        # for fp in val_test_tfrec_fps:
        #     shutil.copy(fp, dataset_run_dir / fp.name)

        # update examples' labels for the training set
        for fp in train_tfrec_fps:

            with tf.io.TFRecordWriter(str(dataset_run_dir / fp.name)) as writer:
                # iterate through the source shard
                tfrecord_dataset = tf.data.TFRecordDataset(str(fp))

                for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                    if label == 'NTP':  # exclude NTPs from training set
                        continue

                    # keep original label
                    example_util.set_bytes_feature(example, 'original_label', [label])

                    # get example from dataset table run
                    example_in_tbl = dataset_run.loc[(fp.stem, example_i)]

                    # overwrite original label with label from dataset table run
                    example_util.set_bytes_feature(example, 'label', [example_in_tbl['label']], allow_overwrite=True)

                    writer.write(example.SerializeToString())

        # create predict shards with NTPs from original training set that are not used
        for fp_i, fp in enumerate(train_tfrec_fps):

            with tf.io.TFRecordWriter(str(dataset_run_dir /
                                          f'predict-shard-ntp-{str(fp_i).zfill(5)}-of'
                                          f'-{str(len(train_tfrec_fps)).zfill(5)}')) as writer:

                # iterate through the source shard
                tfrecord_dataset = tf.data.TFRecordDataset(str(fp))

                for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                    if label == 'NTP':
                        writer.write(example.SerializeToString())

    # validation and test datasets
    val_test_tfrec_fps = [fp for fp in tfrec_fps if fp.name.startswith('val') or fp.name.startswith('test')]
    # get only validation and test set examples
    val_test_set = tfrec_tbl.loc[(tfrec_tbl['shard_name'].str.contains('val')) |
                                 (tfrec_tbl['shard_name'].str.contains('test'))]
    # # get only PCs and AFPs
    # val_test_set = val_test_set.loc[val_test_set['label'] != 'NTP']
    # inject stochastic noise in the PC and AFP populations in the validation and test sets; switch labels for some PCs and AFP
    # val_test_set['label_changed_to_other_class'] = False

    switch_pcs = val_test_set.loc[val_test_set['label'] == 'PC'].sample(frac=config['mislabeling_rate'], replace=False,
                                                                        random_state=rng.bit_generator)
    switch_afps = val_test_set.loc[val_test_set['label'] == 'AFP'].sample(frac=config['mislabeling_rate'],
                                                                          replace=False,
                                                                          random_state=rng.bit_generator)
    val_test_set.loc[switch_pcs.index, 'label'] = 'AFP'
    val_test_set.loc[switch_pcs.index, 'label_changed_to_other_class'] = True
    val_test_set.loc[switch_afps.index, 'label'] = 'PC'
    val_test_set.loc[switch_afps.index, 'label_changed_to_other_class'] = True
    val_test_set.to_csv(res_dir / 'val_test_sets_labels_switched.csv', index=False)
    val_test_set = val_test_set.set_index(keys=['shard_name', 'example_i'])
    for run_i, dataset_run in enumerate(dataset_runs):

        print(f'Creating val and test datasets for run {run_i}...')

        dataset_run_dir = tfrecord_res_dir / f'run{run_i}'
        dataset_run_dir.mkdir(exist_ok=True)

        # # copy validation and test TFRecord shards
        # for fp in val_test_tfrec_fps:
        #     shutil.copy(fp, dataset_run_dir / fp.name)

        # update examples' labels for the validation and test sets
        for fp in val_test_tfrec_fps:

            with tf.io.TFRecordWriter(str(dataset_run_dir / fp.name)) as writer:
                # iterate through the source shard
                tfrecord_dataset = tf.data.TFRecordDataset(str(fp))

                for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    if val_test_set.loc[(fp.stem, example_i), 'label_changed_to_other_class']:
                        label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                        # keep original label
                        example_util.set_bytes_feature(example, 'original_label', [label])

                        # get example from dataset table run
                        example_in_tbl = val_test_set.loc[(fp.stem, example_i)]

                        # overwrite original label with label from dataset table run
                        example_util.set_bytes_feature(example, 'label', [example_in_tbl['label']],
                                                       allow_overwrite=True)

                    writer.write(example.SerializeToString())

    # predict (unlabeled) dataset
    predict_tfrec_fps = [fp for fp in tfrec_fps if fp.name.startswith('predict')]
    for run_i, dataset_run in enumerate(dataset_runs):

        print(f'Creating predict set for run {run_i}...')

        dataset_run_dir = tfrecord_res_dir / f'run{run_i}'
        dataset_run_dir.mkdir(exist_ok=True)

        # copy predict TFRecord shards
        for fp in predict_tfrec_fps:
            shutil.copy(fp, dataset_run_dir / fp.name)

    # # create flip table from TFRecord dataset table
    # print('Creating flip table...')
    # res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum')
    # tfrec_tbl = pd.read_csv(tfrec_dir / 'tfrec_tbl.csv')
    # tfrec_tbl = tfrec_tbl.loc[tfrec_tbl['label'] != 'UNK']  # remove unlabeled data
    # flip_tbl = create_flip_tbl(tfrec_tbl, rnd_seed=2, switch_frac={'PC': 0.1, 'AFP': 0.1}, noise_label='MISLABELED')
    # flip_tbl.to_csv(res_dir / 'flip_tbl.csv', index=False)
    #
    # # create new TFRecord dataset using flip table to switch labels of chosen examples to the mislabeled class
    # print('Creating TFRecord dataset')
    # dest_data_dir = tfrec_dir.parent / f'{tfrec_dir.name}_aum'
    # dest_data_dir.mkdir(exist_ok=True)
    # for tfrec_fp in tfrec_fps:
    #     print(f'Creating TFRecord shard {tfrec_fp}')
    #     switch_labels_tfrecord_shard_from_table(tfrec_fp, dest_data_dir, flip_tbl, noise_label='MISLABELED')

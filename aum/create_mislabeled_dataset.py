"""
Create mislabeled noise
"""

# 3rd party
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import shutil

# local
from src_preprocessing.tf_util import example_util


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
                    try:
                        feat_example = example.features.feature[feature_name].bytes_list.value[0].decode('utf-8')
                    except:
                        aaa
                else:
                    raise ValueError('Data type not expected')

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
        idx_cat_switch = tfrec_tbl.loc[tfrec_tbl['label'] == switch_frac_cat].sample(frac=switch_frac[switch_frac_cat],
                                                                                     replace=False,
                                                                                     random_state=rng.bit_generator).index
        tfrec_tbl.loc[idx_cat_switch, 'label'] = noise_label

    return tfrec_tbl


if __name__ == '__main__':

    # results directory
    res_dir = Path(
        '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/') / f'run_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    res_dir.mkdir(exist_ok=True)

    # create TFRecord dataset table
    tfrec_dir = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
    tfrec_fps = [fp for fp in tfrec_dir.iterdir() if fp.suffix != '.csv' and 'shard' in fp.stem]

    features = {
        'target_id': {'dtype': 'int64'},
        'tce_plnt_num': {'dtype': 'int64'},
        'label': {'dtype': 'str'},

    }
    tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=features)
    tfrec_tbl.to_csv(tfrec_dir / 'tfrec_tbl.csv', index=False)

    tfrec_tbl['label_changed'] = False
    tfrec_tbl['label_before'] = tfrec_tbl['label']

    # get only training set examples
    train_set = tfrec_tbl.loc[tfrec_tbl['shard_name'].str.contains('train')]
    # get PCs and AFPs from the training set
    pc_train_set = train_set.loc[train_set['label'] == 'PC']
    afp_train_set = train_set.loc[train_set['label'] == 'AFP']

    # split these categories into N splits
    n_splits = 3
    # TODO: do I want to shuffle them beforehand? No, it's already shuffled
    pc_train_set_split = np.array_split(pc_train_set, n_splits)
    afp_train_set_split = np.array_split(afp_train_set, n_splits)

    # create different iterations by combining these splits
    # each combination has their examples' labels switched to MISLABELED pseudo-class
    train_set_splits = [pd.concat(split_comb)
                        for split_comb in itertools.product(pc_train_set_split, afp_train_set_split)]

    # create the datasets for the different iterations by combining the PC/AFP splits with the rest of the training set
    noise_label = 'MISLABELED'
    n_runs = len(train_set_splits)
    dataset_runs = [train_set.copy(deep=True) for run_i in range(n_runs)]
    for run_i, comb in enumerate(train_set_splits):
        dataset_runs[run_i].loc[comb.index, 'label'] = noise_label  # set examples to mislabeled class
        dataset_runs[run_i].loc[comb.index, 'label_changed'] = True

    # save each dataset run as csv file
    tbl_res_dir = res_dir / 'trainset_runs'
    tbl_res_dir.mkdir(exist_ok=True)
    for run_i, dataset_run in enumerate(dataset_runs):
        dataset_run.to_csv(tbl_res_dir / f'trainset_run{run_i}.csv', index=False)

    # create TFRecord dataset from these splits
    tfrecord_res_dir = res_dir / 'tfrecords'
    tfrecord_res_dir.mkdir(exist_ok=True)
    val_test_tfrec_fps = [fp for fp in tfrec_fps if not fp.name.startswith('train')]
    train_tfrec_fps = [fp for fp in tfrec_fps if fp.name.startswith('train')]
    dataset_runs = [dataset_run.set_index(keys=['shard_name', 'example_i']) for dataset_run in dataset_runs]
    for run_i, dataset_run in enumerate(dataset_runs):

        print(f'Creating dataset for run {run_i}...')

        dataset_run_dir = tfrecord_res_dir / f'run{run_i}'
        dataset_run_dir.mkdir(exist_ok=True)

        # copy validation and test TFRecord shards
        for fp in val_test_tfrec_fps:
            shutil.copy(fp, dataset_run_dir / fp.name)

        # update examples' labels for the training set
        for fp in train_tfrec_fps:

            with tf.io.TFRecordWriter(str(dataset_run_dir / fp.name)) as writer:
                # iterate through the source shard
                tfrecord_dataset = tf.data.TFRecordDataset(str(fp))

                for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    example_in_tbl = dataset_run.loc[(fp.stem, example_i)]

                    label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")
                    # keep original label
                    example_util.set_bytes_feature(example, 'original_label', [label])

                    # overwrite label
                    example_util.set_bytes_feature(example, 'label', [example_in_tbl['label']], allow_overwrite=True)

                    # # check if example is in flip table
                    # target_id = example.features.feature['target_id'].int64_list.value[0]
                    # tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]
                    #
                    # tce_found = dataset_run.loc[
                    #     (dataset_run['target_id'] == target_id) & (flip_tbl['tce_plnt_num'] == tce_plnt_num)]
                    #
                    # if len(tce_found) == 0:
                    #     flip = False
                    # else:
                    #     flip = tce_found['label'].values[0] == noise_label
                    #
                    # if flip:

                    writer.write(example.SerializeToString())

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

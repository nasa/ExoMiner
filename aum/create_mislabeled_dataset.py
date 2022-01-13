"""

"""

# 3rd party
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np

# local
from src_preprocessing.tf_util import example_util


def switch_labels_tfrecord_shard_from_table(src_data_fp, dest_data_dir, flip_tbl):
    """ Switch labels for examples in TFRecord shard to pseudo new class based on a table.

    :param src_data_fp: Path, file path to source TFRecord shard
    :param dest_data_dir: Path, destination data directory to store new TFRecord files
    :param flip_tbl: pandas DataFrame, table used to flip labels
    :return:
    """

    noise_label = 'MISLABELED'  # label given to examples that switch to the pseudo new class

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

            flip = len(tce_found) == 1

            if flip:
                # overwrite label
                example_util.set_bytes_feature(example, 'label', [noise_label], allow_overwrite=True)

            writer.write(example.SerializeToString())


def create_tfrec_dataset_tbl(tfrec_fps):
    """ Creates TFRecord dataset table by iterating through examples in the TFRecord shards. The table contains the
    example ID ('target_id' + 'tce_plnt_num'), its label ('label'), and the TFRecord shard name it belongs to
    ('shard_name').

    :param tfrec_fps: list, file paths to shards in TFRecord dataset
    :return:
        tfrec_tbl: pandas DataFrame, TFRecord dataset table
    """
    data_to_tbl = []
    for tfrec_fp in tfrec_fps:

        tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():
            example = tf.train.Example()
            example.ParseFromString(string_record)

            target_id = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]
            label = example.features.feature['label'].bytes_list.value[0].decode('utf-8')

            data_to_tbl.append([target_id, tce_plnt_num, label, tfrec_fp.name])

    tfrec_tbl = pd.DataFrame(data=data_to_tbl, columns=['target_id', 'tce_plnt_num', 'label', 'shard_name'])

    return tfrec_tbl


def create_flip_tbl(tfrec_tbl, rnd_seed):
    """ Creates table from TFRecord dataset table where a fraction of PCs and AFPs is randomly selected and their labels
    are switched to noise class label.

    :param tfrec_tbl: pandas DataFrame, TFRecord dataset table
    :param rnd_seed: float, random seed used to instantiate rng that controls random selection of examples whose labels
    are switched
    :return:
        flip_tbl: pandas DataFrame, table used to flip labels
    """

    noise_label = 'MISLABELED'  # label given to examples that switch to the pseudo new class

    tfrec_tbl['flip'] = 'false'
    rng = np.random.default_rng(seed=rnd_seed)

    # choose PCs to switch to noise class
    idx_pcs_switch = tfrec_tbl.loc[tfrec_tbl['label'] == 'PC'].sample(frac=0.1, replace=False, random_state=rng.bit_generator).index
    tfrec_tbl.loc[idx_pcs_switch, 'label'] = noise_label

    # choose AFPs to switch to noise class
    idx_afps_switch = tfrec_tbl.loc[tfrec_tbl['label'] == 'AFP'].sample(frac=0.1, replace=False, random_state=rng.bit_generator).index
    tfrec_tbl.loc[idx_afps_switch, 'label'] = noise_label

    return tfrec_tbl


if __name__ == '__main__':

    # create TFRecord dataset table
    tfrec_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
    # tfrec_fps = [fp for fp in tfrec_dir.iterdir() if fp.suffix != '.csv']
    #
    # tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps)
    # tfrec_tbl.to_csv(tfrec_dir / 'tfrec_tbl.csv', index=False)

    # create flip table from TFRecord dataset table
    res_dir = Path('')
    tfrec_tbl = pd.read_csv(tfrec_dir / 'tfrec_tbl.csv')
    tfrec_tbl = tfrec_tbl.loc[tfrec_tbl['label'] != 'UNK']  # remove unlabeled data
    flip_tbl = create_flip_tbl(tfrec_tbl, rnd_seed=2)
    flip_tbl.to_csv(res_dir / 'flip_tbl.csv', index=False)








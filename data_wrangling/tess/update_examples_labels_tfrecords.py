""" Update labels for examples in the TFRecords. """

# 3rd party
import tensorflow as tf
import pandas as pd
from pathlib import Path

# local
from src_preprocessing.tf_util import example_util

#%%

src_tfrecs_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/tess/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_data/tfrecordstesss1s40-dv_g301-l31_5tr_spline_nongapped_all_features_phases_8-1-2022_1624_updtlablesrenamedfeats')
dest_tfrecs_dir = src_tfrecs_dir.parent / f'{src_tfrecs_dir.name}_toidv'
dest_tfrecs_dir.mkdir(exist_ok=True)

src_tfrecs_fps = [fp for fp in src_tfrecs_dir.iterdir() if fp]

tce_tbl_fp = Path('/Users/msaragoc/Downloads/toi-tce_matching_dv/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv_final.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
tce_tbl.set_index('uid', inplace=True)

label_map = {
    # TESS
    'KP': 'T-KP',
    'CP': 'T-CP',
    'NTP': 'T-NTP',
    'FA': 'T-FA',
    'FP': 'T-FP',
    'EB': 'T-EB',
    'UNK': 'T-UNK',
    'APC': 'T-APC',
    'PC': 'T-PC'
    # # Kepler
    # 'PC': 'K-PC',
    # 'AFP': 'K-AFP',
    # 'NTP': 'K-NTP',
}

label_dict = {'uid': [], 'label_old': [], 'label_new': []}
for src_tfrecs_fp in src_tfrecs_fps:

    with tf.io.TFRecordWriter(str(dest_tfrecs_dir / src_tfrecs_fp.name)) as writer:
        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrecs_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")
            example_label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            label_dict['uid'].append(example_uid)
            label_dict['label_old'].append(example_label)

            example_label_new = tce_tbl.loc[example_uid, 'label']
            label_dict['label_new'].append(example_label_new)

            example_util.set_bytes_feature(example, 'label', [label_map[example_label_new]], allow_overwrite=True)

            writer.write(example.SerializeToString())


label_tbl = pd.DataFrame(label_dict)
label_tbl.to_csv(dest_tfrecs_dir / 'label_change.csv', index=False)

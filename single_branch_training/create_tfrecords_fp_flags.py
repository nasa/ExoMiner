"""
Create new TFRecord data set with FP flags to be used as labels for training single-branch models.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import pandas as pd
# import numpy as np

# local
from src_preprocessing.tf_util import example_util

#%%

src_tfrec_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/cv_keplerq1q17dr25-dv_03-07-2023_1053/tfrecords/eval/')
dest_tfrec_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/cv_keplerq1q17dr25-dv_03-07-2023_1053/tfrecords/eval_fpflags/')
dest_tfrec_dir.mkdir(exist_ok=True)

robovetter_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/kplr_dr25_obs_robovetter_output.csv')
robovetter_tbl = pd.read_csv(robovetter_tbl_fp)

branch_flag_cols = [
    'odd_even_flag',
    'sec_flag',
    'centroid_flag',
    'not_transit_like_flag'
]
examples_in_tfrecords = {'uid': []}
examples_in_tfrecords.update({col: [] for col in branch_flag_cols})
for src_tfrec_fp in [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-')
                                                             and not fp.name.endswith('.csv')]:

    print(f'Iterating over TFRecord file {src_tfrec_fp}...')

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

                example_in_tbl = robovetter_tbl.loc[robovetter_tbl['uid'] == example_uid]
                if len(example_in_tbl) == 0:
                    raise ValueError(f'Example {example_uid} not found in the Robovetter table {robovetter_tbl_fp}.')

                examples_in_tfrecords['uid'].append(example_uid)
                for col in branch_flag_cols:
                    example_util.set_bytes_feature(example, col, [example_in_tbl[col].values[0]])
                    examples_in_tfrecords[col].append(example_in_tbl[col].values[0])

                writer.write(example.SerializeToString())

examples_in_tfrecords = pd.DataFrame(examples_in_tfrecords)
examples_in_tfrecords.to_csv(dest_tfrec_dir / 'examples_in_tfrecords.csv', index=False)

for col in branch_flag_cols:
    print(examples_in_tfrecords[col].value_counts())


#%% check new tfrecord data set

tfrec_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/cv_keplerq1q17dr25-dv_03-07-2023_1053/tfrecords/eval_fpflags')

for src_tfrec_fp in [fp for fp in tfrec_dir.iterdir() if fp.name.startswith('shard-') and not fp.name.endswith('.csv')]:

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")
        example_flag = example.features.feature['odd_even_flag'].bytes_list.value[0].decode("utf-8")
        aaaa


#%%

label_map = {
    '0': 0,
    '1': 1,
}
table_initializer = tf.lookup.KeyValueTensorInitializer(keys=list(label_map.keys()),
                                                        values=list(label_map.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32)

label_to_id = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

data_fields = {'odd_even_flag': tf.io.FixedLenFeature((1, ), tf.string)}
parsed_features = tf.io.parse_single_example(serialized=string_record, features=data_fields)
label_id = label_to_id.lookup(parsed_features['odd_even_flag'])


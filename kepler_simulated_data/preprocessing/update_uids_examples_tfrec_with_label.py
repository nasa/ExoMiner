"""
Update TCE uids to include label so that the ids are actually unique.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path

# local
from src_preprocessing.tf_util import example_util

#%%

src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/agg_src_data')
src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']
print(f'Iterating through {len(src_tfrec_fps)} TFRecord source files.')

dest_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/agg_src_data_updated_uids_with_labels')
dest_tfrec_dir.mkdir(exist_ok=True)

for src_tfrec_fp_i, src_tfrec_fp in enumerate(src_tfrec_fps):

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode('utf-8')
                example_label = example.features.feature['label'].bytes_list.value[0].decode('utf-8').lower()

                example_util.set_bytes_feature(example, 'uid', [f'{example_uid}-{example_label}'], allow_overwrite=True)

                writer.write(example.SerializeToString())

print(f'Finished.')

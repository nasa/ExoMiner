"""
Create data set with only INJ1 and planet examples.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import shutil

#%%

dest_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25obs_planets_sim_inj1_2-22-2024_1115')
dest_tfrec_dir.mkdir(exist_ok=True)

# #%% Copy INJ1 TFRecord files to the destination directory
#
# src_inj1_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/agg_src_data_updated_uids_with_labels')
# src_inj1_tfrec_fps = [fp for fp in src_inj1_tfrec_dir.iterdir() if fp.name.endswith('inj1')]
#
# print(f'Found {len(src_inj1_tfrec_fps)} shards for INJ1 examples in {src_inj1_tfrec_dir}.')
#
# for src_inj1_tfrec_fp in src_inj1_tfrec_fps:
#     shutil.copy(src_inj1_tfrec_fp, dest_tfrec_dir / src_inj1_tfrec_fp.name)

#%% Copy planets examples to TFRecord files written to destination directory

src_planet_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/cv_kepler_q1q17dr25_obs_11-22-2023_2356/tfrecords/eval')
src_planet_tfrec_fps = [fp for fp in src_planet_tfrec_dir.iterdir()]
print(src_planet_tfrec_fps)

planets_dict = {'uid': []}
for src_tfrec_fp_i, src_tfrec_fp in enumerate(src_planet_tfrec_fps):

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode('utf-8')
                example_label = example.features.feature['label'].bytes_list.value[0].decode('utf-8')

                if example_label == 'PC':
                    planets_dict['uid'].append(example_uid)

                    writer.write(example.SerializeToString())

print(f'Finished. Added {len(planets_dict["uid"])} to the data set.')

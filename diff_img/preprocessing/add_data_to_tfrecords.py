"""
Add difference image data from the NumPy files to TFRecords. Creates new TFRecord directory with added difference image
data as features.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np

# local
from src_preprocessing.tf_util import example_util

#%%

src_tfrec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois')
src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-')]

dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_adddiffimg'
dest_tfrec_dir.mkdir(exist_ok=True)

# Kepler - single NumPy file
diff_img_data_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/09-07-2023_1515/diffimg_preprocess.npy')
diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()

# TESS - one NumPy file per sector run
# aggregate all dictionaries
# might be memory intensive as number of sector runs increases...
diff_img_data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/09-20-2023_1502')
diff_img_data = {}
for fp in diff_img_data_dir.iterdir():
    diff_img_data_sector_run = np.load(fp / 'diffimg_preprocess.npy', allow_pickle=True).item()
    diff_img_data.update(diff_img_data_sector_run)


for src_tfrec_fp in src_tfrec_fps:

    print(f'Iterating through shard {src_tfrec_fp}...')

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

                if example_uid not in diff_img_data:
                    print(f'Example {example_uid} not found in the difference image data.')

                # add difference and oot images
                for feature_name in ['diff_imgs', 'oot_imgs']:
                    diff_feature_example = np.array(diff_img_data[example_uid]['cropped_imgs']['diff_imgs'])
                    example_util.set_tensor_feature(example, feature_name, diff_feature_example)

                # add target subpixel position
                target_subpx_pos = np.vstack([diff_img_data[example_uid]['cropped_imgs']['sub_x'],
                                              diff_img_data[example_uid]['cropped_imgs']['sub_y']])
                example_util.set_tensor_feature(example, 'target_subpx_pos', target_subpx_pos)

                # add quality metrics
                example_util.set_float_feature(example, 'qmetric',
                                               diff_img_data[example_uid]['cropped_imgs']['quality'])

                # add image numbers (i.e., sampled quarters/sector runs)
                example_util.set_float_feature(example, 'img_number',
                                               diff_img_data[example_uid]['cropped_imgs']['imgs_numbers'])

                # diff_imgs_example_tfrecord = tf.reshape(tf.io.parse_tensor(example.features.feature['diff_img'].bytes_list.value[0], tf.float32), (5, 11, 11)).numpy()

                writer.write(example.SerializeToString())

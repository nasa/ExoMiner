"""
Add difference image data from the NumPy files to TFRecords. Creates new TFRecord directory with added difference image
data as features.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd

# local
from src_preprocessing.tf_util import example_util

#%%

src_tfrec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois')
src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-')]

# Kepler - single NumPy file
diff_img_data_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/09-07-2023_1515/diffimg_preprocess.npy')
diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()

# # TESS - one NumPy file per sector run
# # aggregate all dictionaries
# # might be memory intensive as number of sector runs increases...
# diff_img_data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/09-20-2023_1502')
# diff_img_data = {}
# # get file paths to directories with difference image data in NumPy files
# sector_dirs = [fp for fp in diff_img_data_dir.iterdir() if fp.is_dir() and 'sector' in fp.name]
# for fp in sector_dirs:
#     diff_img_data_sector_run = np.load(fp / 'diffimg_preprocess.npy', allow_pickle=True).item()
#     diff_img_data.update(diff_img_data_sector_run)

dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_adddiffimg'
dest_tfrec_dir.mkdir(exist_ok=True)

#%% keep uids of examples without difference image data

examples_not_found_dict = {'uid': [], 'filename': [], 'example_i': []}
for src_tfrec_fp in src_tfrec_fps:

    print(f'Iterating through shard {src_tfrec_fp}...')

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

                if example_uid not in diff_img_data:
                    if len(examples_not_found_dict['uid']) % 50 == 0:
                        print(f'Example {example_uid} not found in the difference image data.')
                    examples_not_found_dict['uid'].append(example_uid)
                    examples_not_found_dict['filename'].append(src_tfrec_fp.name)
                    examples_not_found_dict['example_i'].append(example_i)
                    continue

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

                writer.write(example.SerializeToString())

examples_not_found_df = pd.DataFrame(examples_not_found_dict)
examples_not_found_df.to_csv(dest_tfrec_dir / 'examples_without_diffimg_data.csv', index=False)

print(f'Examples without difference image data: {len(examples_not_found_df)}.')
print('Finished adding difference image data to TFRecords.')

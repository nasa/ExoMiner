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

# %% set file paths

src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_merged_8-27-2024_1123_pgram_maxpower')
src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']
print(f'Found {len(src_tfrec_fps)} source TFRecord files.')

# # Kepler - single NumPy file
# diff_img_data_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/11-17-2023_1205/diffimg_preprocess.npy')
# diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()

# TESS - one NumPy file per sector run
# aggregate all dictionaries
# might be memory intensive as number of sector runs increases...
diff_img_data_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tess/2min/dv/diff_img/preprocessed/11-20-2023_1127')
diff_img_data = {}
# get file paths to directories with difference image data in NumPy files
sector_dirs = [fp for fp in diff_img_data_dir.iterdir() if fp.is_dir() and 'sector' in fp.name]
for fp in sector_dirs:
    diff_img_data_sector_run = np.load(fp / 'diffimg_preprocess.npy', allow_pickle=True).item()
    diff_img_data.update(diff_img_data_sector_run)

dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_adddiffimg'
dest_tfrec_dir.mkdir(exist_ok=True)

# %% keep uids of examples without difference image data

examples_not_found_dict = {'uid': [], 'filename': [], 'example_i': [], 'label': []}
for src_tfrec_fp_i, src_tfrec_fp in enumerate(src_tfrec_fps):

    print(f'Iterating through shard {src_tfrec_fp} ({src_tfrec_fp_i + 1}/{len(src_tfrec_fps)})...')

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

        for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

            example = tf.train.Example()
            example.ParseFromString(string_record)

            example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")
            example_label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            if example_uid not in diff_img_data:
                if len(examples_not_found_dict['uid']) % 50 == 0:
                    print(f'Example {example_uid} not found in the difference image data.')
                examples_not_found_dict['uid'].append(example_uid)
                examples_not_found_dict['filename'].append(src_tfrec_fp.name)
                examples_not_found_dict['example_i'].append(example_i)
                examples_not_found_dict['label'].append(example_label)

                continue

            # add difference features
            for suffix_str in ['', '_tc']:
                for img_name in ['diff_imgs', 'oot_imgs', 'target_imgs']:
                    img_data = np.array(diff_img_data[example_uid]['images'][f'{img_name}{suffix_str}'])
                    example_util.set_tensor_feature(example, f'{img_name}{suffix_str}', img_data)

                for pixel_feature_name in ['pixel', 'subpixel']:
                    pixel_feature_data = np.vstack(
                        [diff_img_data[example_uid]['target_position'][f'{pixel_feature_name}_x{suffix_str}'],
                         diff_img_data[example_uid]['target_position'][f'{pixel_feature_name}_y{suffix_str}']])

                    example_util.set_tensor_feature(example, f'{pixel_feature_name}{suffix_str}',
                                                    pixel_feature_data)

            example_util.set_float_feature(example, 'quality', diff_img_data[example_uid]['quality'])
            example_util.set_float_feature(example, 'images_numbers',
                                           diff_img_data[example_uid]['images_numbers'])

            writer.write(example.SerializeToString())

examples_not_found_df = pd.DataFrame(examples_not_found_dict)
examples_not_found_df.to_csv(dest_tfrec_dir / 'examples_without_diffimg_data.csv', index=False)

print(f'Number of examples without difference image data: {len(examples_not_found_df)}.')

print(f'{examples_not_found_df["label"].value_counts()}')

print('Finished adding difference image data to TFRecords.')

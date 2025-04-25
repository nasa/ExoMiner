"""
Add difference image data from the NumPy files to TFRecords. Creates new TFRecord directory with added difference image
data as features.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import multiprocessing
import os

# local
from src_preprocessing.tf_util import example_util

logger = logging.getLogger(__name__)


def write_diff_img_data_to_tfrec_files(src_tfrec_fps, dest_tfrec_dir, diff_img_data):
    """ Write difference image data to a set of TFRecord files under filepaths in `src_tfrec_fps` to a new dataset in
    `dest_tfrec_dir`.

        Args:
            src_tfrec_fps: list of Path objects, source TFRecord filepaths
            dest_tfrec_dir: Path, destination TFRecord
            diff_img_data: dict, preprocessed difference image data

        Returns: examples_not_found_df, pandas DataFrame with examples with no difference image data found

    """

    pid = os.getpid()
    logging.basicConfig(filename=dest_tfrec_dir / 'logs' / f'write_diff_img_data_tfrec_{pid}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        force=True
                        )

    examples_not_found_df_lst = []
    for src_tfrec_fp_i, src_tfrec_fp in enumerate(src_tfrec_fps):

        logger.info(f'Iterating through shard {src_tfrec_fp} ({src_tfrec_fp_i + 1}/{len(src_tfrec_fps)})...')

        dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

        examples_not_found_df = write_diff_img_data_to_tfrec_file(src_tfrec_fp, dest_tfrec_fp, diff_img_data, logger)

        examples_not_found_df_lst.append(examples_not_found_df)

    examples_not_found_df = pd.concat(examples_not_found_df_lst, axis=0, ignore_index=True)

    return examples_not_found_df


def write_diff_img_data_to_tfrec_file(src_tfrec_fp, dest_tfrec_fp, diff_img_data, logger=None):
    """ Write difference image data to a TFRecord file under filepath `src_tfrec_fp` to a new dataset in
        `dest_tfrec_dir`.

        Args:
            src_tfrec_fp: Path, source TFRecord filepath
            dest_tfrec_fp: Path, destination TFRecord
            diff_img_data: dict, preprocessed difference image data
            logger: logger

        Returns: examples_not_found_df pandas DataFrame with examples with no difference image data found

    """

    examples_not_found_dict = {'uid': [], 'filename': [], 'example_i': [], 'label': []}

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
                    if logger:
                        logger.info(f'Example {example_uid} ({example_label}) not found in the difference image data.')
                    else:
                        print(f'Example {example_uid} ({example_label}) not found in the difference image data.')
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

    return examples_not_found_df


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    n_procs = 36
    n_jobs = 108

    # %% set file paths
    src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055_data/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_11-25-2024_1055')
    src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']

    # # Kepler - single NumPy file
    # diff_img_data_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/11-17-2023_1205/diffimg_preprocess.npy')
    # diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()

    # TESS - one NumPy file per sector run
    # aggregate all dictionaries
    # might be memory intensive as number of sector runs increases...
    diff_img_data_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/preprocessed_data/tess/ffi/dv/diff_img/preprocessed_data')
    diff_img_data = {}

    # # get file paths to directories with difference image data in NumPy files
    # sector_dirs = [fp for fp in diff_img_data_dir.iterdir() if fp.is_dir() and 'sector' in fp.name]
    # for fp in sector_dirs:
    #     diff_img_data_sector_run = np.load(fp / 'diffimg_preprocess.npy', allow_pickle=True).item()
    #     diff_img_data.update(diff_img_data_sector_run)

    # get file paths to difference image NumPy files for the needed sector runs
    diff_img_fps = list(diff_img_data_dir.rglob('*.npy'))
    print(f'Found {len(diff_img_fps)} difference image files.')
    n_jobs = min(len(diff_img_fps), n_jobs)
    for fp in diff_img_fps:
        diff_img_data_sector_run = np.load(fp, allow_pickle=True).item()
        diff_img_data.update(diff_img_data_sector_run)

    dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_adddiffimg'
    dest_tfrec_dir.mkdir(exist_ok=True)

    # set logger
    log_dir = dest_tfrec_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    # create logger
    logging.basicConfig(filename=log_dir / f'write_diff_img_data_to_tfrecs_main.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        )

    logger.info(f'Found {len(src_tfrec_fps)} source TFRecord files.')

    # parallel processing
    src_tfrec_fps_jobs = np.array_split(src_tfrec_fps, n_jobs)
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(src_tfrec_fps_job, dest_tfrec_dir, diff_img_data)
            for src_tfrec_fps_job in src_tfrec_fps_jobs]
    async_results = [pool.apply_async(write_diff_img_data_to_tfrec_files, job) for job in jobs]
    pool.close()

    examples_not_found_df_lst = []
    for async_result in async_results:
        examples_not_found_df_lst.append(async_result.get())

    examples_not_found_df = pd.concat(examples_not_found_df_lst, axis=0, ignore_index=True)
    examples_not_found_df.to_csv(dest_tfrec_dir / 'examples_without_diffimg_data.csv', index=False)

    logger.info(f'Number of examples without difference image data: {len(examples_not_found_df)}.')

    logger.info(f'{examples_not_found_df["label"].value_counts()}')

    logger.info('Finished adding difference image data to TFRecords.')

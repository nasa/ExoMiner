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
import yaml

# local
from src_preprocessing.tf_util import example_util

logger = logging.getLogger(__name__)


def write_diff_img_data_to_tfrec_files(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fps, shards_tbl,
                                       n_examples_shard=300):
    """ Write difference image data to a set of TFRecord files under `src_tfrec_dir` to a new dataset in
    `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord
            dest_tfrec_dir: Path, destination TFRecord
            diff_img_data_fps: list, list of Path objects for the NumPy files containing preprocessed image data
            shards_tbl: pandas DataFrame, table containing shard filename 'shard' and example position in shard
                'example_i_tfrec' for the source shards in `src_tfrec_dir`
            n_examples_shard: int, number of examples per shard

        Returns: examples_found_df, pandas DataFrame with the examples that were added to the new TFRecord dataset

    """

    pid = os.getpid()
    logging.basicConfig(filename=dest_tfrec_dir / 'logs' / f'write_diff_img_data_tfrec_{pid}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        force=True
                        )

    # examples_not_found_df_lst = []
    examples_found_df_lst = []
    for diff_img_data_i, diff_img_data_fp in enumerate(diff_img_data_fps):

        logger.info(f'Iterating through Difference Image data NumPy file {diff_img_data_fp} '
                    f'({diff_img_data_i + 1}/{len(diff_img_data_fps)})...')

        examples_added_df = write_diff_img_data_to_tfrec_file(src_tfrec_dir,
                                                              dest_tfrec_dir,
                                                              diff_img_data_fp,
                                                              shards_tbl,
                                                              n_examples_shard,
                                                              logger)

        # examples_not_found_df_lst.append(examples_not_found_df)
        examples_found_df_lst.append(examples_added_df)

    examples_found_df = pd.concat(examples_found_df_lst, axis=0, ignore_index=True)

    return examples_found_df


def add_diff_img_data_to_tfrec_example(example, tce_diff_img_data, imgs_fields):
    """ Add  difference image data to an example in a TFRecord file.

        Args:
            example: TFRecord example
            tce_diff_img_data: dict, difference image data
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset

        Returns: example, with added difference image data
    """

    # add difference features
    for suffix_str in ['', '_tc']:
        for img_name in imgs_fields:
            if img_name not in tce_diff_img_data['images']:
                raise ValueError(f'Image {img_name} not found in the difference image data. '
                                 f'Check `imgs_fields` in YAML configuration file and adapt the variable accordingly.\n'
                                 f'Found images: {tce_diff_img_data["images"].keys()}.')
            img_data = np.array(tce_diff_img_data['images'][f'{img_name}{suffix_str}'])
            example_util.set_tensor_feature(example, f'{img_name}{suffix_str}', img_data)

        for pixel_feature_name in ['pixel', 'subpixel']:
            pixel_feature_data = np.vstack(
                [tce_diff_img_data['target_position'][f'{pixel_feature_name}_x{suffix_str}'],
                 tce_diff_img_data['target_position'][f'{pixel_feature_name}_y{suffix_str}']])

            example_util.set_tensor_feature(example, f'{pixel_feature_name}{suffix_str}',
                                            pixel_feature_data)

    example_util.set_float_feature(example, 'quality', tce_diff_img_data['quality'])
    example_util.set_float_feature(example, 'images_numbers',
                                   tce_diff_img_data['images_numbers'])

    return example

def write_diff_img_data_to_tfrec_file(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fp, shards_tbl, imgs_fields,
                                      n_examples_shard=300, logger=None):
    """ Write preprocessed difference image data in NumPy file `diff_im_data_fp` to TFRecord files under directory
        `src_tfrec_dir` to a new dataset in `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord dataset directory
            dest_tfrec_dir: Path, destination TFRecord dataset directory
            diff_img_data_fp: Path, filepath to preprocessed difference image data
            shards_tbl: pandas DataFrame, table containing shard filename 'shard' and example position in shard
                'example_i_tfrec' for the source shards in `src_tfrec_dir`
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            n_examples_shard: int, number of examples per shard
            logger: logger

        Returns: examples_added_df, pandas DataFrame with the examples that were added to the new TFRecord dataset
    """

    logger.info(f'Reading difference image data in {diff_img_data_fp}...')
    diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()
    logger.info(f'Read difference image data in {diff_img_data_fp}: Found {len(diff_img_data)} TCEs.')

    dest_shard_suffix = 0
    dest_tfrec_fp = dest_tfrec_dir / f'shard-{diff_img_data_fp.parent.stem}_{dest_shard_suffix}'
    examples_lst = []
    examples_added_dict = {field : [] for field in ['uid', 'label']}
    for tce_i, (tce_uid, tce_diff_img_data) in enumerate(diff_img_data.items()):

        if tce_i % 500 == 0:
            logger.info(f'Iterated over {tce_i + 1} TCEs out of {len(diff_img_data)} in {diff_img_data_fp}...')

        # get information on storage location of example for the corresponding TCE
        idx_tce_uid_shards_tbl = shards_tbl['uid'] == tce_uid
        if idx_tce_uid_shards_tbl.sum() == 0:
            logger.info(f'TCE {tce_uid} not found in shards table.')
            continue

        src_shard_example_filename, src_shard_example_i = (shards_tbl.loc[idx_tce_uid_shards_tbl, 'shard'].values[0],
                                                           shards_tbl.loc[idx_tce_uid_shards_tbl,
                                                           'example_i_tfrec'].values[0])

        # get source TFRecord shard
        src_shard_example_fp = src_tfrec_dir / src_shard_example_filename

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_shard_example_fp))
        for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

            if example_i != src_shard_example_i:
                continue
            else:
                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")
                example_label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                if example_uid != tce_uid:
                    raise ValueError(f'TCE ID {example_uid} found in TFRecord dataset does not match expected ID: '
                                     f'{tce_uid}.')
                try:
                    example_with_diffimg_data = add_diff_img_data_to_tfrec_example(example, tce_diff_img_data, imgs_fields)
                    examples_lst.append(example_with_diffimg_data)
                    examples_added_dict['uid'].append(example_uid)
                    examples_added_dict['label'].append(example_label)
                    break
                except ValueError as e:
                    logger.info(f'Caught an error for TCE {example_uid} while adding difference image data to example: '
                                f'{e}')

        # write examples to TFRecord file
        if len(examples_lst) == n_examples_shard or tce_i == len(diff_img_data) - 1:

            logger.info(f'Writing examples with difference image data to {dest_tfrec_fp}')
            with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:
                for example in examples_lst:
                    writer.write(example.SerializeToString())

            # reset list
            examples_lst = []
            dest_shard_suffix += 1
            dest_tfrec_fp = dest_tfrec_dir / f'shard-{diff_img_data_fp.parent.stem}_{dest_shard_suffix}'

    examples_added_df = pd.DataFrame(examples_added_dict)

    logger.info(f'Iterated over all TCEs in {diff_img_data_fp}. Added difference image data to '
                f'{len(examples_added_df)} TCEs.')

    return examples_added_df


def write_diff_img_data_to_tfrec_files_main(config_fp, src_tfrec_dir=None, src_diff_img_fp=None):
    """ Wrapper for `write_diff_img_data_to_tfrec_files()`.

    Args:
        config_fp: str, path to config file
        src_tfrec_dir: str, path to source TFRecord directory
        src_diff_img_fp: str, path to source difference image directory

    Returns:

    """

    with open(config_fp, 'r') as file:
        config = yaml.unsafe_load(file)

    if src_tfrec_dir is not None:
        config['src_tfrec_dir'] = Path(src_tfrec_dir)
    if src_diff_img_fp is not None:
        config['src_diff_img_fp'] = Path(src_diff_img_fp)

    # get shard filepaths
    src_tfrec_fps = [fp for fp in config['src_tfrec_dir'].iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']

    # table that has information on the location of examples across shards in TFRecord dataset directory
    shards_tbl_fp = config['src_tfrec_dir'] / 'shards_tbl.csv'

    # load shards table
    shards_tbl = pd.read_csv(shards_tbl_fp)

    # get filepaths to difference image data NumPy files
    diff_img_fps = list(config['src_diff_img_fp'].rglob('*.npy'))

    # set number of jobs to number of files
    n_jobs = min(config['n_jobs'], len(diff_img_fps))

    # create destination directory
    dest_tfrec_dir = config['src_tfrec_dir'].parent / f'{config["src_tfrec_dir"].name}_diffimg'
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

    logger.info(f'`Images to be added from preprocessed difference image to TFRecord dataset`: '
                f'{config["imgs_fields"]}.')

    logger.info(f'Found {len(src_tfrec_fps)} source TFRecord files.')
    logger.info(f'Found {len(diff_img_fps)} difference image NumPy files.')

    # split shards across jobs
    src_diff_img_fps_jobs = np.array_split(diff_img_fps, n_jobs)
    jobs = [(src_tfrec_dir, dest_tfrec_dir, src_diff_img_fps_job, shards_tbl, config['imgs_fields'],
             config['n_examples_shard'])
            for src_diff_img_fps_job in src_diff_img_fps_jobs]

    # parallel processing
    if config['parallel_processing']:
        pool = multiprocessing.Pool(processes=config['n_processes'])
        async_results = [pool.apply_async(write_diff_img_data_to_tfrec_files, job) for job in jobs]
        pool.close()
        pool.join()
        examples_found_df_lst = [async_result.get() for async_result in async_results]
    else:
        # sequential
        examples_found_df_lst = [write_diff_img_data_to_tfrec_files(*job) for job in jobs]

    # aggregate tables of examples found into a single one
    logger.info('Aggregating tables of examples added to new TFRecord directory with difference image data...')
    examples_found_df = pd.concat(examples_found_df_lst, axis=0, ignore_index=True)

    # get examples not found
    examples_not_found_df = shards_tbl.loc[~shards_tbl['uid'].isin(examples_found_df['uid'])]
    examples_not_found_df.to_csv(dest_tfrec_dir / 'examples_without_diffimg_data.csv', index=False)
    logger.info(f'Number of examples without difference image data: {len(examples_not_found_df)}.')
    logger.info(f'{examples_not_found_df["label"].value_counts()}')

    logger.info('Finished adding difference image data to TFRecords.')


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    config_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/src_preprocessing/diff_img/preprocessing/config_add_diff_img_tfrecords.yaml'

    write_diff_img_data_to_tfrec_files_main(config_fp)

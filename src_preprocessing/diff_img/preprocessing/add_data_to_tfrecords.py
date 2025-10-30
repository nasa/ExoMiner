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
import argparse

# local
from src_preprocessing.tf_util import example_util


def add_diff_img_data_to_tfrec_example(example, tce_diff_img_data, imgs_fields):
    """ Add difference image data to an example in a TFRecord file.

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


def parse_uid(serialized_example):
    """Parse only TCE unique IDs 'uid' from the examples in the TFRecord datset.

    :param TF serialized_example: serialized TFRecord example 
    :return tuple: tuple of uid and serialized example
    """
    
    # define the feature spec for just the UuidID
    feature_spec = {
        'uid': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
    
    return parsed_features['uid'], serialized_example

def make_filter_by_uid_fn(chosen_uids):
    """Create filter for TFRecord dataset that excludes TCE examples whose uid is not included in `chosen_uids`.

    :param TF Tensor chosen_uids: chosen uids to filter examples
    :return: filtering function
    """
    
    def filter_uid_tf(uid):
        """Filter function of examples in TFRecord dataset based on 'uid' feature

        :param Tensor string uid: chosen uid
        :return Tensor bool: boolean values that check for uid inclusion
        """
        
        return tf.reduce_any(tf.equal(uid, chosen_uids))
    
    return filter_uid_tf


def write_diff_img_data_to_tfrec_file(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fp, imgs_fields, n_examples_shard=300, logger=None):
    """ Write preprocessed difference image data in NumPy file `diff_im_data_fp` to TFRecord files under directory
        `src_tfrec_dir` to a new dataset in `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord dataset directory
            dest_tfrec_dir: Path, destination TFRecord dataset directory
            diff_img_data_fp: Path, filepath to preprocessed difference image data
            # shards_tbl: pandas DataFrame, table containing shard filename 'shard' and example position in shard
            #     'example_i_tfrec' for the source shards in `src_tfrec_dir`
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            n_examples_shard: int, number of examples per shard
            logger: logger

        Returns: examples_added_df, pandas DataFrame with the examples that were added to the new TFRecord dataset
    """

    # load data dictionary with different image data for the sector run
    logger.info(f'Reading difference image data in {diff_img_data_fp}...')
    diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()
    logger.info(f'Read difference image data in {diff_img_data_fp}: Found {len(diff_img_data)} TCEs.')

    # get filepaths to TFRecord files
    src_tfrec_fps = list(src_tfrec_dir.glob('shard-*'))
    
    # convert list of TCE uids into a TF tensor
    uids_tensor = tf.constant(list(diff_img_data.keys()), dtype=tf.string)

    dataset = tf.data.TFRecordDataset(src_tfrec_fps)
    # parse only uids, keep rest of example serialized
    dataset = dataset.map(parse_uid, num_parallel_calls=tf.data.AUTOTUNE)
    # filter examples based on chosen uids from differnece image data
    filter_uids_fn = make_filter_by_uid_fn(uids_tensor)
    dataset = dataset.filter(lambda uid, _: filter_uids_fn(uid))
    # batch dataset
    batched_dataset = dataset.batch(n_examples_shard)
    
    # iterate over examples in batched TFRecord dataset and add difference image data to examples and write them to new TFRecord files in destination directory
    examples_added_dict = {field : [] for field in ['uid', 'label']}
    for batch_i, batch in enumerate(batched_dataset):
        
        dest_tfrec_fp = dest_tfrec_dir / f'shard-{diff_img_data_fp.parent.stem}_{batch_i}'
        
        logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Iterated over batch {batch_i} and writing into {str(dest_tfrec_fp)}...')
        
        batch_uids, batch_serialized = batch
        batch_examples_cnt = 0  # count examples successfully added to new TFRecord dataset
        
        with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:
            for batch_example_i, (example_uid, serialize_example) in enumerate(zip(batch_uids, batch_serialized)):
                
                if batch_example_i % 50 == 0:
                    logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Batch {batch_i}: Iterating over example {batch_example_i} in batch...')
                
                example_uid_str = example_uid.numpy().decode('utf-8')

                if example_uid_str not in diff_img_data:
                    raise ValueError(f'TCE ID {example_uid_str} found in TFRecord dataset was not found in {diff_img_data_fp} (currently at batch {batch_i} TCE {batch_example_i + 1}).')
        
                example_proto = tf.train.Example()
                example_proto.ParseFromString(serialize_example.numpy())
                
                # add diff image data
                try:
                    updated_example = add_diff_img_data_to_tfrec_example(example_proto, diff_img_data[example_uid_str], imgs_fields)
                    examples_added_dict['uid'].append(example_uid_str)
                    examples_added_dict['label'].append(example_proto.features.feature['label'].bytes_list.value[0].decode('utf-8'))
                except ValueError as e:
                    logger.info(f'Caught an error for TCE {example_uid_str} while adding difference image data to example (currently at batch {batch_i} TCE {batch_example_i + 1}):\n{e}\nSkipping...')
                    continue
                
                writer.write(updated_example.SerializeToString())
                batch_examples_cnt += 1
        
        logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Wrote {batch_examples_cnt}/{batch_example_i + 1} TCEs into batch {batch_i} at {str(dest_tfrec_fp)}.')

    examples_added_df = pd.DataFrame(examples_added_dict)

    logger.info(f'Iterated over all TCEs in {diff_img_data_fp}.\nAdded difference image data to '
                f'{len(examples_added_df)}/{len(diff_img_data)} TCEs.')

    return examples_added_df

def write_diff_img_data_to_tfrec_files(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fps, imgs_fields, n_examples_shard=300):
    """ Write difference image data to a set of TFRecord files under `src_tfrec_dir` to a new dataset in
    `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord
            dest_tfrec_dir: Path, destination TFRecord
            diff_img_data_fps: list, list of Path objects for the NumPy files containing preprocessed image data
            # shards_tbl: pandas DataFrame, table containing shard filename 'shard' and example position in shard
            #     'example_i_tfrec' for the source shards in `src_tfrec_dir`
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            n_examples_shard: int, number of examples per shard

        Returns: examples_found_df, pandas DataFrame with the examples that were added to the new TFRecord dataset

    """

    tf.config.set_visible_devices([], 'GPU')
    
    pid = os.getpid()

    logger = logging.getLogger(name=f'add_diff_img_data_to_tfrec_files_{pid}')
    logger_handler = logging.FileHandler(filename=dest_tfrec_dir / 'logs' /
                                                  f'add_diff_img_data_to_tfrec_files_{pid}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Started adding difference image data to TFRecord dataset...')

    # examples_not_found_df_lst = []
    examples_found_df_lst = []
    for diff_img_data_i, diff_img_data_fp in enumerate(diff_img_data_fps):

        logger.info(f'Iterating through Difference Image data NumPy file {diff_img_data_fp} '
                    f'({diff_img_data_i + 1}/{len(diff_img_data_fps)})...')

        examples_added_df = write_diff_img_data_to_tfrec_file(src_tfrec_dir,
                                                              dest_tfrec_dir,
                                                              diff_img_data_fp,
                                                              imgs_fields,
                                                              n_examples_shard,
                                                              logger=logger,
                                                              )

        # examples_not_found_df_lst.append(examples_not_found_df)
        examples_found_df_lst.append(examples_added_df)

    examples_found_df = pd.concat(examples_found_df_lst, axis=0, ignore_index=True)

    return examples_found_df


def write_diff_img_data_to_tfrec_files_main(config_fp, src_tfrec_dir=None, src_diff_img_fp=None):
    """ Wrapper for `write_diff_img_data_to_tfrec_files()`.

    Args:
        config_fp: str, path to config file
        src_tfrec_dir: str, path to source TFRecord directory
        src_diff_img_fp: str, path to source difference image directory

    Returns:

    """

    with open(config_fp, 'r') as file:
        config = yaml.safe_load(file)

    if src_tfrec_dir is not None:
        config['src_tfrec_dir'] = Path(src_tfrec_dir)
    else:
        src_tfrec_dir = Path(config['src_tfrec_dir'])
    if src_diff_img_fp is not None:
        config['src_diff_img_fp'] = Path(src_diff_img_fp)
    else:
        src_diff_img_fp = Path( config['src_diff_img_fp'])

    # get shard filepaths
    src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and
                     fp.suffix != '.csv']

    # table that has information on the location of examples across shards in TFRecord dataset directory
    shards_tbl_fp = src_tfrec_dir / 'shards_tbl.csv'

    # load shards table
    shards_tbl = pd.read_csv(shards_tbl_fp)

    # get filepaths to difference image data NumPy files
    diff_img_fps = list(src_diff_img_fp.rglob('*.npy'))

    # set number of jobs to number of files
    n_jobs = min(config['n_jobs'], len(diff_img_fps))

    # create destination directory
    dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_diffimg'
    dest_tfrec_dir.mkdir(exist_ok=True)
    
    # save yaml file to destination TFRecord dataset
    with open(dest_tfrec_dir / config_fp.name, 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    # set logger
    log_dir = dest_tfrec_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    # create logger
    logger = logging.getLogger(name='add_diff_img_data_to_tfrec_files_main')
    logger_handler = logging.FileHandler(filename=log_dir / 'add_diff_img_data_to_tfrec_files_main.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Started adding difference image data to TFRecord dataset...')

    logger.info(f'Images to be added from preprocessed difference image to TFRecord dataset: '
                f'{config["imgs_fields"]}.')

    logger.info(f'Found {len(src_tfrec_fps)} source TFRecord files.')
    logger.info(f'Found {len(diff_img_fps)} difference image NumPy files.')

    # split difference image files across jobs
    src_diff_img_fps_jobs = np.array_split(diff_img_fps, n_jobs)
    jobs = [(src_tfrec_dir, dest_tfrec_dir, src_diff_img_fps_job, config['imgs_fields'], config['n_examples_shard'])
            for src_diff_img_fps_job in src_diff_img_fps_jobs]

    # parallel processing
    if config['parallel_processing']:
        pool = multiprocessing.Pool(processes=config['n_processes'])
        async_results = [pool.apply_async(write_diff_img_data_to_tfrec_files, job) for job in jobs]
        pool.close()
        pool.join()
        examples_found_df_lst = []
        for async_result in async_results:
            try:
                examples_found_df_lst.append(async_result.get())
            except Exception as e:
                print(f'Error in multiprocessing job: {e}')
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
    if 'label' in examples_not_found_df:
        logger.info(f'\n{examples_not_found_df["label"].value_counts()}')

    logger.info('Finished adding difference image data to TFRecords.')


if __name__ == '__main__':

    multiprocessing.set_start_method('spawn') 

    tf.config.set_visible_devices([], 'GPU')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file')
    args = parser.parse_args()

    write_diff_img_data_to_tfrec_files_main(Path(args.config_fp))

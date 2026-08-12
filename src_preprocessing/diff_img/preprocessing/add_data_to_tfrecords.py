"""
Add difference image data from the NumPy files to TFRecords. Creates new TFRecord directory with added difference image
data as features.

Source TFRecord dataset: TFRecord files under `src_tfrec_dir` with filenames that match "shard-*" (files that start with that pattern but whose suffix is .csv are filtered) 
and that contain features:
- uid: str feature for TCE unique ID.

Preprocessed difference image data: NumPy files under `src_diff_img_fp` with filenames that match "*.npy" and that are structure as:
- TCE uid (str)
    - images
        - imgs_fields[0]{suffix_str}: NumPy array (e.g., difference image); name in accordance with config YAML file
        - imgs_fields[1]{suffix_str}: ...
        - ...
    - target_position
        - pixel_x{suffix_str}
        - pixel_y{suffix_str}
        - subpixel_x{suffix_str}
        - subpixel_y{suffix_str}
    - quality: quality metrics of sampled quarters/sectors
    - images_number: list of sampled quarters/sectors

suffix_str is in {'_tc', ''}, and means data was centered on target pixel position vs original location
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
from tqdm import tqdm
from functools import partial

# local
from src_preprocessing.tf_util import example_util
from src_preprocessing.utils_manipulate_tfrecords import parse_uid, make_filter_by_uid_fn


def add_diff_img_data_to_tfrec_example(example, tce_diff_img_data, imgs_fields, allow_overwrite=False):
    """ Add difference image data to an example in a TFRecord file.

        Args:
            example: TFRecord example
            tce_diff_img_data: dict, difference image data
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            allow_overwrite: bool, if True it will overwrite existing difference image data in the examples in the TFRecord dataset

        Returns: example, with added difference image data
    """

    # add difference features
    for suffix_str in ['', '_tc']:
        for img_name in imgs_fields:
            if f'{img_name}{suffix_str}' not in tce_diff_img_data['images']:
                raise ValueError(f'Image {img_name}{suffix_str} not found in the difference image data. '
                                 f'Check `imgs_fields` in YAML configuration file and adapt the variable accordingly.\n'
                                 f'Found images: {tce_diff_img_data["images"].keys()}.')
            img_data = np.array(tce_diff_img_data['images'][f'{img_name}{suffix_str}'])
            example_util.set_float_feature(example, f'{img_name}{suffix_str}', img_data.flatten(), allow_overwrite=allow_overwrite)

        for pixel_feature_name in ['pixel', 'subpixel']:
            pixel_feature_data = np.vstack(
                [tce_diff_img_data['target_position'][f'{pixel_feature_name}_x{suffix_str}'],
                 tce_diff_img_data['target_position'][f'{pixel_feature_name}_y{suffix_str}']])

            example_util.set_float_feature(example, f'{pixel_feature_name}{suffix_str}', pixel_feature_data.flatten(), allow_overwrite=allow_overwrite)

    example_util.set_float_feature(example, 'quality', tce_diff_img_data['quality'], allow_overwrite=allow_overwrite)
    example_util.set_float_feature(example, 'images_numbers', tce_diff_img_data['images_numbers'], allow_overwrite=allow_overwrite)

    return example


def write_diff_img_data_to_tfrec_file(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fp, imgs_fields, n_examples_shard=300, logger=None, allow_overwrite=False):
    """ Write preprocessed difference image data in NumPy file `diff_im_data_fp` to TFRecord files under directory
        `src_tfrec_dir` to a new dataset in `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord dataset directory
            dest_tfrec_dir: Path, destination TFRecord dataset directory
            diff_img_data_fp: Path, filepath to preprocessed difference image data
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            n_examples_shard: int, number of examples per shard
            logger: logger
            allow_overwrite: bool, if True it will overwrite existing difference image data in the examples in the TFRecord dataset

        Returns: examples_failed_df, pandas DataFrame with the examples that failed to be added
    """
    
    if logger is None:
        pid = os.getpid()

        logger = logging.getLogger(name=f'add_diff_img_data_to_tfrec_files_{pid}')
        logger_handler = logging.FileHandler(filename=dest_tfrec_dir / 'logs' / 'add_diff_img_data_to_tfrec_files_{pid}.log', mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started adding difference image data in {str(diff_img_data_fp)} to TFRecord dataset...')
        logger.info(f'Reading difference image data in {diff_img_data_fp}...')

    # load data dictionary with different image data for the sector run
    logger.info(f'Reading difference image data in {diff_img_data_fp}...')
    diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()
    logger.info(f'Read difference image data in {diff_img_data_fp}: Found {len(diff_img_data)} TCEs.')

    # get filepaths to TFRecord files
    src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']
    
    dataset_uids_only = tf.data.TFRecordDataset(src_tfrec_fps).map(parse_uid, num_parallel_calls=tf.data.AUTOTUNE)
    tfrec_uids = set()
    for uid, _ in dataset_uids_only:
        tfrec_uids.add(uid.numpy().decode('utf-8'))
    n_examples_src_tfrec = len(tfrec_uids)
    logger.info(f'Found {n_examples_src_tfrec} TCEs in TFRecord dataset in {str(src_tfrec_dir)}.')
    diff_uids = set(diff_img_data.keys())
    # missing_in_diffimg = tfrec_uids - diff_uids         # present in TFRecord but absent in diff_image dict
    missing_in_tfrec = diff_uids  - tfrec_uids        # present in diff_image dict but absent in TFRecord
    present_in_both = tfrec_uids & diff_uids

    # convert list of TCE uids into a TF tensor
    uids_tensor = tf.constant(sorted(list(present_in_both)), dtype=tf.string)

    # create TFRecord dataset object from files
    dataset = tf.data.TFRecordDataset(src_tfrec_fps)
    # parse only uids, keep rest of example serialized
    dataset = dataset.map(parse_uid, num_parallel_calls=tf.data.AUTOTUNE)
    # filter examples based on chosen uids from difference image data
    filter_uids_fn = make_filter_by_uid_fn(uids_tensor)
    dataset = dataset.filter(lambda uid, _: filter_uids_fn(uid))
    # batch dataset
    batched_dataset = dataset.batch(n_examples_shard)
    total_batches = int(np.ceil(len(present_in_both) / n_examples_shard))
    
    successes = []
    failures = []  # list of dicts: uid, present_in_tfrec, present_in_diffimg, added=False, reason
    # iterate over examples in batched TFRecord dataset and add difference image data to examples and write them to new TFRecord files in destination directory
    for batch_i, batch in tqdm(enumerate(batched_dataset), total=total_batches, desc=f"[{diff_img_data_fp.parent.stem}] Processing batches (approx. total batch number)"):
        
        dest_tfrec_fp = dest_tfrec_dir / f'shard-{diff_img_data_fp.parent.stem}_{batch_i}'
        
        logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Iterated over batch {batch_i} and writing into {str(dest_tfrec_fp)}...')
        
        batch_uids, batch_serialized = batch
        batch_examples_cnt = 0  # count examples successfully added to new TFRecord dataset
        
        with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:
            for batch_example_i, (example_uid, serialize_example) in enumerate(zip(batch_uids, batch_serialized)):
                
                if batch_example_i % 50 == 0:
                    logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Batch {batch_i}: Iterating over example {batch_example_i} in batch...')
                
                example_uid_str = example_uid.numpy().decode('utf-8')

                # if example_uid_str not in diff_img_data:
                #     raise ValueError(f'TCE ID {example_uid_str} found in TFRecord dataset was not found in {diff_img_data_fp} (currently at batch {batch_i} TCE {batch_example_i + 1}).')
        
                example_proto = tf.train.Example()
                example_proto.ParseFromString(serialize_example.numpy())
                
                # add diff image data
                try:
                    updated_example = add_diff_img_data_to_tfrec_example(example_proto, diff_img_data[example_uid_str], imgs_fields, allow_overwrite)
                    
                except ValueError as e:
                    
                    failures.append({
                        'uid': example_uid_str,
                        'present_in_tfrec': True,
                        'present_in_diffimg': True,
                        'added': False,
                        'reason': f'error_adding: {e}'
                    })

                    logger.info(f'Caught an error for TCE {example_uid_str} while adding difference image data to example (currently at batch {batch_i} TCE {batch_example_i + 1}):\n{e}\nSkipping...')
                    continue
                
                writer.write(updated_example.SerializeToString())
                successes.append(example_uid_str)
                batch_examples_cnt += 1
        
        logger.info(f'[{diff_img_data_fp.name} | {len(diff_img_data)} TCEs] Wrote {batch_examples_cnt}/{batch_example_i + 1} TCEs into batch {batch_i} at {str(dest_tfrec_fp)}.')

    logger.info(f'Finished writing data to new TFRecord dataset in {dest_tfrec_dir}. Creating table with TCEs that failed to be added...')
    
    # build table with failed examples
    failures_missing_tfrec = pd.DataFrame({
        'uid': list(missing_in_tfrec),
        'present_in_tfrec': False,
        'present_in_diffimg': True,
        'added': False,
        'reason': 'missing_in_tfrec',
    })
    failures_error_adding = pd.DataFrame(failures) if failures else pd.DataFrame(
        columns=['uid', 'present_in_tfrec', 'present_in_diffimg', 'added', 'reason']
    )

    examples_failed_df = pd.concat(
        [failures_missing_tfrec, failures_error_adding],
        ignore_index=True
    )

    # report summary in logs
    n_success = len(successes)
    n_failed = len(examples_failed_df)
    logger.info(f'SUMMARY: successes={n_success}, failures={n_failed}, total_diffimg={diff_uids}, '
                f'intersection_processed={len(present_in_both)}')
    logger.info(f'Failure breakdown: missing_in_tfrec={len(missing_in_tfrec)}, error_adding={len(failures)}')

    return examples_failed_df


def write_diff_img_data_to_tfrec_files(src_tfrec_dir, dest_tfrec_dir, diff_img_data_fps, imgs_fields, n_examples_shard=300, allow_overwrite=False):
    """ Write difference image data to a set of TFRecord files under `src_tfrec_dir` to a new dataset in
    `dest_tfrec_dir`.

        Args:
            src_tfrec_dir: Path, source TFRecord
            dest_tfrec_dir: Path, destination TFRecord
            diff_img_data_fps: list, list of Path objects for the NumPy files containing preprocessed image data
            imgs_fields: list, list of images in preprocessed difference image data to be added to the TFRecord dataset
            n_examples_shard: int, number of examples per shard
            allow_overwrite: bool, if True it will overwrite existing difference image data in the examples in the TFRecord dataset

        Returns: examples_failed_df, pandas DataFrame with the examples that failed to be added

    """
    
    pid = os.getpid()

    logger = logging.getLogger(name=f'add_diff_img_data_to_tfrec_files_{pid}')
    logger_handler = logging.FileHandler(filename=dest_tfrec_dir / 'logs' /
                                                  f'add_diff_img_data_to_tfrec_files_{pid}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Started adding difference image data to TFRecord dataset...')

    examples_failed_df_lst = []
    for diff_img_data_i, diff_img_data_fp in enumerate(diff_img_data_fps):

        logger.info(f'Iterating through Difference Image data NumPy file {diff_img_data_fp} '
                    f'({diff_img_data_i + 1}/{len(diff_img_data_fps)})...')

        examples_failed_df = write_diff_img_data_to_tfrec_file(src_tfrec_dir,
                                                              dest_tfrec_dir,
                                                              diff_img_data_fp,
                                                              imgs_fields,
                                                              n_examples_shard,
                                                              logger=logger,
                                                              allow_overwrite=allow_overwrite,
                                                              )

        examples_failed_df_lst.append(examples_failed_df)

    examples_failed_df = pd.concat(examples_failed_df_lst, axis=0, ignore_index=True)

    return examples_failed_df


def create_table_failed_examples(examples_failed_df, save_fp, logger=None, metadata=None, shards_tbl_src_missing=None):
    """Create table with examples that failed because they were not found in either the source TFRecord dataset or in the difference image data, or they
        were in both but failed to be added.

    :param pandas DataFrame examples_failed_df: examples with difference image data
    :param Path save_fp: save filepath
    :param Python logger logger: logger, defaults to None
    :param dict metadata: additional metadata, defaults to None
    :param shards_tbl_src_missing: pandas DataFrame, including column 'uid' that contains the TCEs' UIDs for the examples in the source TFRecord dataset that could not be added
    """
    
    if shards_tbl_src_missing is not None:
        failures_missing_diff = pd.DataFrame({
            'uid': shards_tbl_src_missing['uid'].to_list(),
            'present_in_tfrec': True,
            'present_in_diffimg': False,
            'added': False,
            'reason': 'missing_in_diffimg',
        })
        
        examples_failed_df = pd.concat([examples_failed_df, failures_missing_diff], axis=0)
        
    examples_failed_df.attrs['Description'] = 'This table contains auxliary information regarding the TCEs in the difference image data that either did not show up in the TFRecord dataset or could not be added due to an error.'
    examples_failed_df.attrs['TFRecord Directory with Preprocessed Data'] = str(save_fp.parent)
    examples_failed_df.attrs['Table Filepath'] = str(save_fp)
    examples_failed_df.attrs['Creation Date'] = pd.Timestamp.now().isoformat()
    examples_failed_df.attrs['Created By'] = 'src_preprocessing/diff_img/preprocessing/add_data_to_tfrecords.py'
    if metadata:
        for k, v in metadata.items():
            examples_failed_df.attrs[k] = v

    with open(save_fp, "w") as f:
        for key, value in examples_failed_df.attrs.items():
            f.write(f"# {key}: {value}\n")
        examples_failed_df.to_csv(f, index=False)
    
    if logger:
        logger.info(f'Number of examples without difference image data: {len(examples_failed_df)}.')
    
    if 'label' in examples_failed_df.columns and logger:
        logger.info(f'\n{examples_failed_df["label"].value_counts()}')


def report_progress(result, completed_jobs, total_jobs, logger):
    """Callback function to report progress."""
    completed_jobs[0] += 1
    logger.info(f'Job completed. {completed_jobs[0]}/{total_jobs} jobs finished.')

def handle_error(e, logger):
    """Error callback function to handle errors."""
    logger.error(f'Error in multiprocessing job: {e}')
    raise e
        
def write_diff_img_data_to_tfrec_files_main(config_fp, src_tfrec_dir=None, src_diff_img_fp=None):
    """ Wrapper for `write_diff_img_data_to_tfrec_files()`.

    Args:
        config_fp: str, path to config file
        src_tfrec_dir: str, path to source TFRecord directory
        src_diff_img_fp: str, path to source difference image directory

    Returns:

    """

    if isinstance(config_fp, str):
        config_fp = Path(config_fp)

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
    src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-') and fp.suffix != '.csv']

    # get filepaths to difference image data NumPy files
    diff_img_fps = list(src_diff_img_fp.rglob('*.npy'))

    # set number of jobs to number of files
    n_jobs = min(config['n_jobs'], len(diff_img_fps))

    # create destination directory
    dest_tfrec_dir = Path(config.get('dest_tfrec_dir') or src_tfrec_dir.parent / f'{src_tfrec_dir.name}_diffimg')
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

    logger.info(f'Images to be added from preprocessed difference image to TFRecord dataset: {config["imgs_fields"]}.')

    logger.info(f'Found {len(src_tfrec_fps)} source TFRecord files.')
    logger.info(f'Found {len(diff_img_fps)} difference image NumPy files.')

    # split difference image files across jobs
    src_diff_img_fps_jobs = np.array_split(diff_img_fps, n_jobs)
    jobs = [(src_tfrec_dir, dest_tfrec_dir, src_diff_img_fps_job, config['imgs_fields'], config['n_examples_shard'], config['overwrite'])
            for src_diff_img_fps_job in src_diff_img_fps_jobs]

    # parallel processing
    if config['parallel_processing']:
        
        with multiprocessing.Manager() as manager:
            completed_jobs = manager.list([0])  # shared counter for completed jobs
            total_jobs = len(jobs)

            report_progress_with_logger = partial(report_progress, completed_jobs=completed_jobs, total_jobs=total_jobs, logger=logger)
            handle_error_with_logger = partial(handle_error, logger=logger)

            with multiprocessing.Pool(processes=config['n_processes']) as pool:
                async_results = [
                    pool.apply_async(
                        write_diff_img_data_to_tfrec_files,
                        job,
                        callback=report_progress_with_logger,
                        error_callback=handle_error_with_logger
                    )
                    for job in jobs
                ]

                examples_failed_df_lst = []
                for async_result in async_results:
                    try:
                        examples_failed_df_lst.append(async_result.get())  # This will raise any exceptions from the worker process
                    except Exception as e:
                        logger.error(f'Error in multiprocessing job: {e}')
                        raise
    else:
        # sequential
        examples_failed_df_lst = []
        for job_i, job in enumerate(jobs):
            try:
                logger.info(f'Running job {job_i} ({job_i + 1}/{len(jobs)} jobs).')
                examples_failed_df_lst.append(write_diff_img_data_to_tfrec_files(*job))
            except Exception as e:
                logger.error(f'Error in sequential job {job_i}: {e}')
                raise e

    logger.info('Finished adding difference image data to TFRecords.')

    # aggregate tables of examples found into a single one
    logger.info('Aggregating tables of difference image data examples that failed to be added to the new TFRecord directory...')
    examples_failed_df = pd.concat(examples_failed_df_lst, axis=0, ignore_index=True)

    logger.info('Creating auxiliary table...')
    
    # save information on examples that failed
    shards_tbl_src_fp = src_tfrec_dir / 'shards_tbl.csv'
    if shards_tbl_src_fp.exists():
        shards_tbl_src = pd.read_csv(shards_tbl_src_fp)
    else:
        logger.info(f'Shards table for source TFRecord directory not found in {str(shards_tbl_src_fp)}. Attempting to create it')
        try:
            shards_tbl_src = create_table_for_tfrecord_dataset(src_tfrec_fps, {'uid': 'str'}, delete_corrupted_tfrec_files=False, verbose=False, logger=None)
        except Exception as e:
            logger.warning(f'Could not create shards table for source TFRecord directory {str(src_tfrec_dir)}: {e}')
            shards_tbl_src = None
    
    # create shards table for source and destination tfrecord directories to find examples in source tfrecord that were not added       
    try:
        shards_tbl_dest = create_table_for_tfrecord_dataset(src_tfrec_fps, {'uid': 'str'}, delete_corrupted_tfrec_files=False, verbose=False, logger=None)
        shards_tbl_src_missing = shards_tbl_src.loc[~shards_tbl_src['uid'].isin(shards_tbl_dest['uid'])]
    except Exception as e:
            logger.warning(f'Could not create shards table for destination TFRecord directory {str(dest_tfrec_dir)}: {e}')
            shards_tbl_src_missing = None
    
    data_metadata_extra = {'Source TFRecord Directory': str(src_tfrec_dir),
                           'Difference Image Data Directory': str(src_diff_img_fp),
                           }
    create_table_failed_examples(examples_failed_df, dest_tfrec_dir / 'examples_failed.csv', logger, data_metadata_extra, shards_tbl_src_missing=shards_tbl_src_missing)

    logger.info('Finished creating auxiliary table.')


if __name__ == '__main__':

    multiprocessing.set_start_method('spawn') 

    tf.config.set_visible_devices([], 'GPU')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file')
    args = parser.parse_args()

    write_diff_img_data_to_tfrec_files_main(Path(args.config_fp))

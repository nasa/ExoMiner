"""
Create TFRecord dataset for hyperparameter optimization that includes only TCEs in targets that are not shared with FFI
data.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import pandas as pd
import multiprocessing
import numpy as np


def create_shards_for_hpo_dataset(src_data_shards_fps, dest_dir, target_2min_shards_tbl):
    """ Create shards for TFRecord dataset that includes only examples in table `target_2min_shards_tbl`.

    Args:
        src_data_shards_fps: list of Paths, source shards
        dest_dir: Path, destination directory for new shards
        target_2min_shards_tbl: pandas DataFrame, includes TCEs to be added to new TFRecord dataset

    Returns:
    """

    for src_data_shard_fp_i, src_data_shard_fp in enumerate(src_data_shards_fps):

        print(f'Iterating over {src_data_shard_fp.name} ({src_data_shard_fp_i + 1}/{len(src_data_shards_fps)})...')

        dest_data_shard_fp = dest_dir / src_data_shard_fp.name

        with (tf.io.TFRecordWriter(str(dest_data_shard_fp)) as writer):

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_shard_fp))

            examples_src_shard_df = target_2min_shards_tbl.loc[target_2min_shards_tbl['shard'] ==
                                                               src_data_shard_fp.name]
            if len(examples_src_shard_df) == 0:
                print(f'No examples in {src_data_shard_fp} to be added to destination TFRecord dataset.')

            for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                if example_i in examples_src_shard_df['example_i_tfrec'].to_list():  # example found

                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    example_uid = example.features.feature['uid'].bytes_list.value[0].decode('utf-8')

                    if (examples_src_shard_df['uid'] == example_uid).any():

                        writer.write(example.SerializeToString())


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    n_procs = 36  # number of processes for parallel processing
    parallel_processing = True
    # set directories
    src_data_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_data/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_agg_diffimg')
    dest_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_data/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_agg_diffimg_targetsnotshared')
    # get shards table for FFI data
    ffi_shards_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_data/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_agg_diffimg/shards_tbl.csv')

    # load FFI shards table
    ffi_shards_tbl = pd.read_csv(ffi_shards_tbl_fp)

    # load 2-min shards table
    src_2min_shards_tbl = pd.read_csv(src_data_dir / 'shards_tbl.csv')

    dest_dir.mkdir(exist_ok=True)

    # filter 2-min shards table for TCEs whose targets are not shared with FFI TCEs and that are labeled (i.e., != UNK)
    target_2min_shards_tbl = src_2min_shards_tbl.loc[
        ~src_2min_shards_tbl['target_id'].isin(ffi_shards_tbl['target_id'])]
    target_2min_shards_tbl = target_2min_shards_tbl.loc[target_2min_shards_tbl['label'] != 'UNK']
    print(f'Copying {len(target_2min_shards_tbl)} examples to {dest_dir}.')

    src_data_shards_fps = list(src_data_dir.glob('shard-*'))
    print(f'Found {len(src_data_shards_fps)} source shards.')

    n_jobs = len(src_data_shards_fps)
    src_data_shards_fps_jobs = np.array_split(src_data_shards_fps, n_jobs)
    jobs = [(src_data_shards_fps_job, dest_dir, target_2min_shards_tbl)
            for src_data_shards_fps_job in src_data_shards_fps_jobs]

    if parallel_processing:
        pool = multiprocessing.Pool(processes=n_procs)
        async_results = [pool.apply_async(create_shards_for_hpo_dataset, job) for job in jobs]
        pool.close()
        pool.join()
    else:
        for job in jobs:
            create_shards_for_hpo_dataset(*jobs)

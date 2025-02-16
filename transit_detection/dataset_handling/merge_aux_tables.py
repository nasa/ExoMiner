"""
Merge auxillary tables corresponding to TFRecord shards, for chunks of the transit detection dataset.

Each entry in a shard and aux_tbl should be per TCE in order to merge with expected functionality.
"""

# 3rd party
import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd

def merge_chunks(dest_tfrec_path, dest_aux_tbl_path, chunk_paths, update_col=True):
    """
    Merge aux_tbl chunks for a shard into a single aux_tbl

    Input:
        dest_tfrec_path: Path, corresponding to destination for merged tfrecord dataset shard
        dest_aux_tbl_path: Path, corresponding to destination for merged auxillary tbl
        chunk_paths: list(tuple(str, str)), corresponding to chunk shards and aux tbls to process, respectively
        update_col: bool, if True update 'tfrec_fn' column in csv to new shard name

    Returns:
        None
    """
    aux_tbls = []

    with tf.io.TFRecordWriter(str(dest_tfrec_path)) as writer:
        try:                
            for chunk_shard_path, chunk_aux_tbl_path in chunk_paths:
                # write to new TFRecordDataset
                dataset_chunk = tf.data.TFRecordDataset(str(chunk_shard_path))

                for example in dataset_chunk: 
                    # copy each example to dest_tfrec_path
                    writer.write(example.numpy())

                # load aux_tbls
                aux_tbl_chunk = pd.read_csv(chunk_aux_tbl_path)
                # copy aux_tbl chunk to dest_aux_tbl_path
                aux_tbls.append(aux_tbl_chunk)

        except Exception as e:
            print(f"ERROR: For {chunk_shard_path.name} and {chunk_aux_tbl_path.name}: {e}")

    # concatenate aux_tbls
    merged_aux_tbl = pd.concat(aux_tbls, ignore_index=True)

    # update 'tfrec_fn' column
    if update_col:
        merged_aux_tbl['tfrec_fn'] = dest_aux_tbl_path.name

    merged_aux_tbl.to_csv(dest_aux_tbl_path, index=False)


def get_chunk_paths(chunked_dataset_dir, chunk_num_range, shard_prefix, aux_tbl_prefix) -> list[tuple]:
    """
    Gets corresponding shard and aux_tbl chunk paths for a range of inclusive chunk nums.

    Input: 
        chunked_dataset_dir: Path, directory containing shard and aux tbl chunks
        chunk_num_range: tuple(int, int), the inclusive range of chunk numbers to process, ie: (1,100)
        shard_prefix: str, a prefix that chunk shards adhere to
        aux_tbl_prefix: str, a prefix that chunk aux tbls adhere to

    Returns: 
        A list of tuples for corresponding chunk shards and chunk aux tbls
    """

    chunk_shards = []
    chunk_aux_tbls = []

    chunk_num_start, chunk_num_end = chunk_num_range
    chunks_to_process = np.arange(start=chunk_num_start, stop=chunk_num_end + 1) # range of chunks to process

    for chunk_i in chunks_to_process:
        try:
            chunk_shard_path = list(chunked_dataset_dir.rglob(f"*{shard_prefix}{str(chunk_i).zfill(4)}"))[0]
            chunk_aux_tbl_path = list(chunked_dataset_dir.rglob(f"*{aux_tbl_prefix}{str(chunk_i).zfill(4)}.csv"))[0]

            if chunk_shard_path.exists() and chunk_aux_tbl_path.exists():
                chunk_shards.append(chunk_shard_path)
                chunk_aux_tbls.append(chunk_aux_tbl_path)
                
        except Exception as e:
            print(f"ERROR: Chunk {chunk_i} not found: {e}")

    return list(zip(chunk_shards, chunk_aux_tbls))

def test_chunk_paths(chunked_dataset_dir, shard_prefix, aux_tbl_prefix):
    chunks = np.zeros(8611)
    for chunk_i, chunk in enumerate(chunks, start=1):
        try:
            chunk_shard_path = list(chunked_dataset_dir.rglob(f"*{shard_prefix}{str(chunk_i).zfill(4)}"))[0]
            chunk_aux_tbl_path = list(chunked_dataset_dir.rglob(f"*{aux_tbl_prefix}{str(chunk_i).zfill(4)}.csv"))[0]

            if chunk_shard_path.exists() and chunk_aux_tbl_path.exists():
                print(f'Chunk exists: {chunk_i}')

        except Exception as e:
            print(f"ERROR: Chunk {chunk_i} not found: {e}")

if __name__ == "__main__":

    # directory containing unmerged tfrecord and auxillary chunks
    chunked_dataset_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024/tfrecords")

    # directory for merged tf record and auxillary table
    dest_dataset_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024/merged_tfrecords")

    shard_prefix = 'test_shard_0001-'
    aux_tbl_prefix = 'data_tbl_chunk-'

    # test_chunk_paths(chunked_dataset_dir, shard_prefix, aux_tbl_prefix)
    chunk_paths = get_chunk_paths(chunked_dataset_dir, (1,8611), shard_prefix, aux_tbl_prefix)

    # stop here for testing
    dest_tfrec_path = dest_dataset_dir / 'merged_shard_0001-0001'
    dest_aux_tbl_path = dest_dataset_dir / 'merged_data_tbl_chunks_0001-8611.csv'

    # merge all chunks
    merge_chunks(dest_tfrec_path, dest_aux_tbl_path, chunk_paths, update_col=False)



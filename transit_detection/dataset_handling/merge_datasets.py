"""
Split tfrec dataset into training, validation, and test sets
"""
# 3rd party
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # tensorflow Info/Warning messages not printed
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import multiprocessing
from functools import partial
from copy import deepcopy
import time
from pathlib import Path


def process_shard(shard_num, pri_tfrec_dir, sec_tfrec_dir, output_dir, num_shards):

    try:
        np.random.seed(42)
        shard_num = str(shard_num).zfill(4)
        for split_name in ["train", "test", "val"]:
            pri_tfrec_fp = (
                pri_tfrec_dir
                / split_name
                / f"norm_{split_name}_shard_{shard_num}-{num_shards}.tfrecord"
            )
            sec_tfrec_fp = (
                sec_tfrec_dir
                / split_name
                / f"norm_{split_name}_shard_{shard_num}-{num_shards}.tfrecord"
            )
            if not pri_tfrec_fp.exists():
                print(f"pri_tfrec_fp: {str(pri_tfrec_fp)} does not exist")
                continue
            print(f"Processing: {str(pri_tfrec_fp.name)}")
            if not pri_tfrec_fp.exists():
                print(f"pri_tfrec_dir: {str(pri_tfrec_fp)} does not exist")
                continue
            print(f"Processing: {str(sec_tfrec_fp.name)}")

            output_tfrec_dir = output_dir / split_name
            output_tfrec_dir.mkdir(parents=True, exist_ok=True)
            output_tfrec_fp = (
                output_tfrec_dir
                / f"norm_{split_name}_shard_{shard_num}-{num_shards}.tfrecord"
            )

            pri_tfrec_dataset = tf.data.TFRecordDataset(str(pri_tfrec_fp))
            sec_tfrec_dataset = tf.data.TFRecordDataset(str(sec_tfrec_fp))
            writer = tf.io.TFRecordWriter(str(output_tfrec_fp))

            pri_tfrec_dataset = tf.data.TFRecordDataset(str(pri_tfrec_fp))
            sec_tfrec_dataset = tf.data.TFRecordDataset(str(sec_tfrec_fp))

            # Write all examples from both datasets to the output file
            with tf.io.TFRecordWriter(str(output_tfrec_fp)) as writer:
                for record in pri_tfrec_dataset:
                    writer.write(record.numpy())
                for record in sec_tfrec_dataset:
                    writer.write(record.numpy())

    except Exception as e:
        print(f"Error processing shard {shard_num}: {e}")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    num_shards = 8593
    max_num_processes = 8

    pri_dataset_name = "TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_filt_3sig_it_EB_no_ntp"
    sec_dataset_name = "TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_filt_3sig_it_EB_ONLY_BAL_NTP"

    dest_dataset_name = "TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_filt_3sig_it_EB_bal_ntp"


    # src directory for raw tfrecord shards
    pri_tfrec_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/datasets/{pri_dataset_name}/tfrecords"
    )
    assert (
        pri_tfrec_dir.exists()
    ), f"ERROR: src_tfrec_dir, {str(pri_tfrec_dir)}, does not exist"

    sec_tfrec_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/datasets/{sec_dataset_name}/tfrecords"
    )
    assert (
        sec_tfrec_dir.exists()
    ), f"ERROR: src_tfrec_dir, {str(sec_tfrec_dir)}, does not exist"

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/datasets/{dest_dataset_name}/tfrecords"
    )
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # destination directory for logging
    log_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/logging/merge_{dest_dataset_name}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # log dataset split
    logger = logging.getLogger(f"split_dataset_logger")
    logger.setLevel(logging.INFO)

    log_path = Path(log_dir) / f"split_dataset.log"
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter("%(asctime)s - %(levelname)s- %(message)s")
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    # set random seed
    np.random.seed(42)

    # define partial func to predefine default values for directories and target sets
    partial_func = partial(
        process_shard,
        pri_tfrec_dir=pri_tfrec_dir,
        sec_tfrec_dir=sec_tfrec_dir,
        output_dir=dest_tfrec_dir,
        num_shards=num_shards,
    )
    start = time.time()
    
    num_processes = min(max_num_processes, os.cpu_count())
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)

    jobs = [
        shard_num for shard_num in range(1, num_shards + 1)
    ]  # chunk 1 to chunk XXXX

    logger.info(
        f"Beginning processing {len(jobs)} chunks from range {jobs[0]} to {jobs[-1]}"
    )

    results = []
    for shard_num in jobs:
        result = pool.apply_async(partial_func, args=[shard_num])
        results.append(result)

    for result in results:
        result.get()
    pool.close()
    pool.join()

    end = time.time()
    print(f"Finished processing train val test split using in {end - start} seconds")

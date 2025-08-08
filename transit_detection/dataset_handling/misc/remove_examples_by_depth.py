"""
Split tfrec dataset into training, validation, and test sets
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Info/Warning messages not printed

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import logging

import multiprocessing
from functools import partial
from copy import deepcopy
import time


# local
from src_preprocessing.tf_util import example_util


def process_shard(shard_num, input_dir, output_dir, num_shards):
    remove_dispositions = ["NEB", "NPC"]

    try:
        shard_num = str(shard_num).zfill(4)
        for split_name in ["train", "test", "val"]:
            input_tfrec_fp = (
                input_dir
                / split_name
                / f"norm_{split_name}_shard_{shard_num}-{num_shards}.tfrecord"
            )
            if not input_tfrec_fp.exists():
                print(f"input_tfrec_fp: {str(input_tfrec_fp)} does not exist")
                continue
            print(f"Processing: {str(input_tfrec_fp.name)}")

            output_tfrec_dir = output_dir / split_name
            output_tfrec_dir.mkdir(parents=True, exist_ok=True)
            output_tfrec_fp = (
                output_tfrec_dir
                / f"norm_{split_name}_shard_{shard_num}-{num_shards}.tfrecord"
            )

            tfrec_dataset = tf.data.TFRecordDataset(str(input_tfrec_fp))
            writer = tf.io.TFRecordWriter(str(output_tfrec_fp))

            total_count = 0
            invalid_count = 0

            for str_record in tfrec_dataset:
                total_count += 1
                example = tf.train.Example()
                example.ParseFromString(str_record.numpy())

                disposition = (
                    example.features.feature["disposition"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )

                label = (
                    example.features.feature["label"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )

                depth = example.features.feature["tce_depth"].float_list.value[0]

                if disposition in remove_dispositions:
                    invalid_count += 1
                elif (label == "1") and (depth > 100000):
                    invalid_count += 1
                else:
                    writer.write(str_record.numpy())

            writer.close()

            print(
                f"[{split_name} | shard {shard_num}] removed {invalid_count} / {total_count} examples due to disposition."
            )

    except Exception as e:
        print(f"Error processing shard {shard_num}: {e}")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    num_shards = 1815

    # src directory for raw tfrecord shards
    src_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_07-11-2025_no_ntp_no_detrend_split_norm/tfrecords"
    )
    assert (
        src_tfrec_dir.exists()
    ), f"ERROR: src_tfrec_dir, {str(src_tfrec_dir)}, does not exist"

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_07-11-2025_no_ntp_no_detrend_split_norm_filtered_v2/tfrecords"
    )
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # destination directory for logging
    log_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/logging/filter_TESS_exoplanet_dataset_07-11-2025_no_ntp_no_detrend_split_norm_v2"
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
        input_dir=src_tfrec_dir,
        output_dir=dest_tfrec_dir,
        num_shards=num_shards,
    )
    start = time.time()
    # using N-2 = 126 as tested working value
    num_processes = min(8, os.cpu_count())
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)

    jobs = [
        shard_num for shard_num in range(1, num_shards + 1)
    ]  # chunk 1 to chunk 8611

    logger.info(
        f"Beginning processing {len(jobs)} chunks from range {jobs[0]} to {jobs[-1]}"
    )

    results = []
    for shard_num in jobs:
        result = pool.apply_async(partial_func, args=[shard_num])
        results.append(result)

    for result in results:
        logger.info(result.get())
    pool.close()
    pool.join()

    end = time.time()
    print(f"Finished processing train val test split using in {end - start} seconds")

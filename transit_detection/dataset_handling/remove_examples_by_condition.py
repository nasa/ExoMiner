"""
Split tfrec dataset into training, validation, and test sets
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Info/Warning messages not printed

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
from pathlib import Path

# local
from src_preprocessing.tf_util import example_util


def process_shard(shard_num, input_dir, output_dir, num_shards):
    remove_dispositions = []

    try:
        np.random.seed(42)
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

                tw_flag = (
                    example.features.feature["transit_example"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )

                depth = example.features.feature["tce_depth"].float_list.value[0]

                # Handling tic offset

                tce_dikco_msky = example.features.feature[
                    "tce_dikco_msky"
                ].float_list.value[0]
                tce_dikco_msky_err = example.features.feature[
                    "tce_dikco_msky_err"
                ].float_list.value[0]

                TESS_PX_SCALE = 21  # arcsec
                KEPLER_PX_SCALE = 3.98  # arcsec
                tess_lower_bound_err = 2.5
                kepler_lower_bound_err = 0.0667
                TESS_TO_KEPLER_PX_SCALE_RATIO = TESS_PX_SCALE / KEPLER_PX_SCALE
                tce_dikco_msky_original = tce_dikco_msky * TESS_TO_KEPLER_PX_SCALE_RATIO
                tce_dikco_msky_err_original = (
                    tce_dikco_msky_err * TESS_TO_KEPLER_PX_SCALE_RATIO
                    + tess_lower_bound_err
                    - TESS_TO_KEPLER_PX_SCALE_RATIO * kepler_lower_bound_err
                )

                
                # if disposition != "NTP" or tw_flag != "1":
                #     invalid_count += 1
                # elif np.random.randint(1,150) != 1:
                #     invalid_count += 1
                if disposition in remove_dispositions:
                    invalid_count += 1
                # elif disposition not in ['NTP']:
                #     invalid_count += 1
                # elif tw_flag != 
                # elif disposition == "EB" and label == "1":
                #     should_keep = np.random.randint(1, 12) == 1
                #     if should_keep:
                #         writer.write(str_record.numpy())
                #     else:
                #         invalid_count += 1
                # elif disposition == "NTP" and label == "0":
                #     invalid_count += 1
                # elif disposition == "NTP" and label == "1":
                #     should_keep = np.random.randint(1, 150) == 1
                #     if should_keep:
                #         writer.write(str_record.numpy())
                #     else:
                #         invalid_count += 1
                # elif (label == "1") and (depth > 100000):
                #     invalid_count += 1
                elif ((disposition == "EB") and (label == "1")) and (
                    ((tce_dikco_msky_err < 0) or (tce_dikco_msky_err_original < 0)
                     or
                     ((tce_dikco_msky_original / tce_dikco_msky_err_original) > 3.0))
                ):
                    invalid_count += 1
                # elif ((disposition == "EB") and (label == "1")) and (
                #     (tce_dikco_msky_original / tce_dikco_msky_err_original) > 3.0
                # ):
                #     invalid_count += 1
                else:
                    writer.write(str_record.numpy())

            writer.close()

            print(
                f"[{split_name} | shard {shard_num}] removed {invalid_count} / {total_count} examples due to disposition or condition."
            )

    except Exception as e:
        print(f"Error processing shard {shard_num}: {e}")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    num_shards = 8593
    max_num_processes = 8

    src_dataset_name = "TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_filt_3sig_it_EB"
    filt_suffix = "ONLY_BAL_NTP"

    # src directory for raw tfrecord shards
    src_tfrec_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/datasets/{src_dataset_name}/tfrecords"
    )
    assert (
        src_tfrec_dir.exists()
    ), f"ERROR: src_tfrec_dir, {str(src_tfrec_dir)}, does not exist"

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/datasets/{src_dataset_name}_{filt_suffix}/tfrecords"
    )
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # destination directory for logging
    log_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/logging/filter_{src_dataset_name}_{filt_suffix}"
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

    num_processes = min(max_num_processes, os.cpu_count())
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
        result.get()
    pool.close()
    pool.join()

    end = time.time()
    print(f"Finished processing train val test split using in {end - start} seconds")

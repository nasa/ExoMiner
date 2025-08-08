"""
Used to prune corrupted examples from a dataset that prevent training
- Should primarily be used in the case of a handful of corrupted examples (often due to interrupted processing during dataset construction)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Info/Warning messages not printed

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

import multiprocessing
from functools import partial
from copy import deepcopy
import time
import glob
from collections import defaultdict
import csv

# local
from src_preprocessing.tf_util import example_util
from src_preprocessing.lc_preprocessing.utils_manipulate_tfrecords import (
    create_table_with_tfrecord_examples,
)


def has_bad_values(tensor):
    """Return True if tensor contains any NaNs or Infs."""
    return tf.reduce_any(
        tf.math.logical_or(tf.math.is_nan(tensor), tf.math.is_inf(tensor))
    ).numpy()

def process_shard(shard_num, input_dir, output_dir, num_shards):
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

            invalid_count = 0
            cleaned_count = 0
            total_count = 0
            corrupted_uid_reasons = defaultdict(list)

            for str_record in tfrec_dataset:
                total_count += 1
                example = tf.train.Example()
                example.ParseFromString(str_record.numpy())

                uid = example.features.feature.get("uid", None)
                uid = (
                    uid.bytes_list.value[0].decode("utf-8")
                    if uid and uid.bytes_list.value
                    else "UNKNOWN_UID"
                )
                example_is_valid = True

                def check_and_log(field_name):
                    if field_name in example.features.feature:
                        tensor = tf.convert_to_tensor(
                            example.features.feature[field_name].float_list.value,
                            dtype=tf.float32,
                        )
                        if has_bad_values(tensor):
                            corrupted_uid_reasons[uid].append(
                                f"{field_name}: NaN or Inf"
                            )
                            return False
                    return True

                fields_to_check = [
                    "flux",
                    "flux_norm",
                    "flux_quality",
                    "diff_img",
                    "oot_img",
                    "snr_img",
                    "target_img",
                    "diff_img_stdnorm",
                    "oot_img_stdnorm",
                    "snr_img_stdnorm",
                ]
                for field in fields_to_check:
                    if not check_and_log(field):
                        example_is_valid = False

                if example_is_valid:
                    writer.write(str_record.numpy())
                    cleaned_count += 1
                else:
                    invalid_count += 1

            writer.close()

            # print(
            #     f"[{split_name} | shard {shard_num}] Cleaned {cleaned_count} / {total_count} examples."
            # )

            if invalid_count > 0:
                print(
                    f"[{split_name} | shard {shard_num}] removed {invalid_count} / {total_count} examples due to corruption."
                )

            # Save UID corruption report
            if corrupted_uid_reasons:
                shard_csv_path = (
                    output_tfrec_dir
                    / f"corrupted_uids_{split_name}_{shard_num}-{num_shards}.csv"
                )
                with open(shard_csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["uid", "corruption_reasons"])
                    for uid, reasons in corrupted_uid_reasons.items():
                        writer.writerow([uid, "; ".join(reasons)])

    except Exception as e:
        print(f"Error processing shard {shard_num}: {e}")


def get_merged_aux_tbl_df(aux_tbl_dir: Path, aux_tbl_pattern: str) -> pd.DataFrame:

    aux_tbls = []

    aux_tbl_fps = glob.glob(str(aux_tbl_dir / aux_tbl_pattern))

    print(f"Found {len(aux_tbl_fps)} aux_tbl fps")

    for i, aux_tbl_fp in enumerate(aux_tbl_fps):
        print(f"Processing {i}")
        try:
            # load aux_tbls
            aux_tbl = pd.read_csv(str(aux_tbl_fp))

            # copy aux_tbl chunk to dest_aux_tbl_path
            aux_tbls.append(aux_tbl)

        except Exception as e:
            print(
                f"ERROR: For {Path(aux_tbl_fp).name} and {Path(aux_tbl_fp).name}: {e}"
            )
            continue

    # concatenate aux_tbls
    merged_aux_tbl = pd.concat(aux_tbls, ignore_index=True)

    return merged_aux_tbl


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    num_shards = 1815
    max_num_processes = 8

    # src directory for raw tfrecord shards
    src_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_07-11-2025_no_ntp_detrend_split_norm/tfrecords"
    )
    assert (
        src_tfrec_dir.exists()
    ), f"ERROR: src_tfrec_dir, {str(src_tfrec_dir)}, does not exist"

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_07-11-2025_no_ntp_detrend_split_norm_clean/tfrecords"
    )
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # destination directory for logging
    log_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/logging/clean_TESS_exoplanet_dataset_07-11-2025_no_ntp_detrend_split_norm"
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
    pool = multiprocessing.Pool(processes=8, maxtasksperchild=1)

    jobs = [
        shard_num for shard_num in range(1, num_shards + 1)
    ] 

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

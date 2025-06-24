"""
Split tfrec dataset into training, validation, and test sets
"""

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

# local
from src_preprocessing.tf_util import example_util
from src_preprocessing.lc_preprocessing.utils_manipulate_tfrecords import (
    create_table_with_tfrecord_examples,
)


def process_shard(
    input_shard_num, input_dir, output_dir, train_targets, val_targets, test_targets
):
    try:
        input_shard_num = str(input_shard_num).zfill(4)

        input_shard_fp = input_dir / f"raw_shard_{input_shard_num}-8611.tfrecord"

        train_tfrec_dir = output_dir / "train"
        train_tfrec_dir.mkdir(parents=True, exist_ok=True)
        val_tfrec_dir = output_dir / "val"
        val_tfrec_dir.mkdir(parents=True, exist_ok=True)
        test_tfrec_dir = output_dir / "test"
        test_tfrec_dir.mkdir(parents=True, exist_ok=True)

        train_tfrec_fp = (
            train_tfrec_dir / f"train_shard_{input_shard_num}-8611.tfrecord"
        )
        val_tfrec_fp = val_tfrec_dir / f"val_shard_{input_shard_num}-8611.tfrecord"
        test_tfrec_fp = test_tfrec_dir / f"test_shard_{input_shard_num}-8611.tfrecord"

        train_count = 0
        val_count = 0
        test_count = 0

        writers = {
            "train": tf.io.TFRecordWriter(str(train_tfrec_fp)),
            "val": tf.io.TFRecordWriter(str(val_tfrec_fp)),
            "test": tf.io.TFRecordWriter(str(test_tfrec_fp)),
        }

        shard_dataset = tf.data.TFRecordDataset(str(input_shard_fp))

        for str_record in shard_dataset:
            example = tf.train.Example()
            example.ParseFromString(str_record.numpy())

            target_id = example.features.feature["target_id"].int64_list.value[0]

            split = ""

            if target_id in train_targets:
                split = "train"
                train_count += 1
            elif target_id in val_targets:
                split = "val"
                val_count += 1
            elif target_id in test_targets:
                split = "test"
                test_count += 1
            else:
                raise ValueError(f"Target ID {target_id} not found in any set")

            writers[split].write(str_record.numpy())

        for writer in writers.values():
            writer.close()

        tfrec_tbls = []

        data_fields = {
            "uid": "str",
            "label": "str",
            "target_id": "int",
            "tce_plnt_num": "int",
            "numberOfPlanets": "int",
            "tce_num_transits": "int",
            "tce_num_transits_obs": "int",
            "mag_cat": "int",
            "mission": "int",
            "sector": "str",
            "tce_maxmes": "float",
            "tce_maxmesd": "float",
            "tce_depth": "float",
            "tce_period": "float",
            "tce_time0bk": "float",
            "tce_duration": "float",
            "disposition": "str",
            "disposition_source": "str",
        }

        if train_count > 0:
            tfrec_tbls.append(
                create_table_with_tfrecord_examples(train_tfrec_fp, data_fields)
            )
        if val_count > 0:
            tfrec_tbls.append(
                create_table_with_tfrecord_examples(val_tfrec_fp, data_fields)
            )
        if test_count > 0:
            tfrec_tbls.append(
                create_table_with_tfrecord_examples(test_tfrec_fp, data_fields)
            )

        tfrec_tbl = pd.concat(tfrec_tbls, axis=0)

        aux_tbl_path = output_dir / "aux_tbls"
        aux_tbl_path.mkdir(parents=True, exist_ok=True)

        tfrec_tbl.to_csv(
            aux_tbl_path / f"shards_tbl_{input_shard_num}-8611.csv", index=False
        )

        print(f"COMPLETED processing tfrecord {Path(input_shard_fp).name} successfully")

        return f"COMPLETED processing tfrecord {Path(input_shard_fp).name} successfully with {train_count} train ex., {val_count} val ex., and {test_count} test ex."

    except Exception as e:
        print(f"ERROR while processing tfrecord {Path(input_shard_fp).name}: {e}")
        return f"ERROR while processing tfrecord {Path(input_shard_fp).name}: {e}"


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
    # src directory for raw tfrecord shards
    src_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_06-20-2025/tfrecords"
    )

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_06-20-2025_split/tfrecords"
    )
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # destination directory for logging
    log_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/logging/split_TESS_exoplanet_dataset_06-20-2025"
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

    # load merged aux tbl
    print(f"Processing Merged Aux TBL DataFrame")
    aux_tbl = get_merged_aux_tbl_df(src_tfrec_dir, "data_tbl_????-????.csv")
    print(f"Finished DataFrame")

    # get unique targets from aux tbl
    unique_targets = aux_tbl["target_id"].unique()

    # shuffle targets
    shuffled_targets = np.random.permutation(unique_targets)

    # fractions for set sizes
    train_set_frac = 0.8
    val_set_frac = 0.1

    # number of unique targets
    N = len(shuffled_targets)

    print(f"Processing: {N} unique targets")
    logger.info(f"Processing: {N} unique targets")

    # compute exact sizes for each split
    num_train = int(N * 0.8)
    num_val = int(N * 0.1)
    num_test = N - num_train - num_val  # remaining fraction

    # load targets per set from randomly shuffled targets
    train_targets = set(shuffled_targets[:num_train])
    val_targets = set(shuffled_targets[num_train : num_train + num_val])
    test_targets = set(shuffled_targets[num_train + num_val :])

    logger.info(f"Using {len(train_targets)} targets in training set")
    logger.info(f"Using {len(val_targets)} targets in validation set")
    logger.info(f"Using {len(test_targets)} targets in test set")

    # Save the dataset targets to a npy file
    dataset_targets = {}
    dataset_targets["train_set"] = train_targets
    dataset_targets["val_set"] = val_targets
    dataset_targets["test_set"] = test_targets

    # save targets to numpy for later validation
    target_npy_path = Path("/nobackupp27/jochoa4/work_dir/data/stats/")
    target_npy_path.mkdir(parents=True, exist_ok=True)
    target_npy_path = target_npy_path / "split_dataset_targets.npy"

    np.save(target_npy_path, dataset_targets)

    logger.info(f"Saving targets to: {target_npy_path}")

    # define partial func to predefine default values for directories and target sets
    partial_func = partial(
        process_shard,
        # def input_shard_num per job
        input_dir=src_tfrec_dir,
        output_dir=dest_tfrec_dir,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
    start = time.time()
    # using N-2 = 126 as tested working value
    pool = multiprocessing.Pool(processes=16, maxtasksperchild=1)

    jobs = [shard_num for shard_num in range(1, 8611 + 1)]  # chunk 1 to chunk 8611

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

"""
Normalization pipeline for flux windows and img features (diff_img, oot_img, snr_img) from TESS transit_detection dataset

Normalize flux window data in TFRecords using per-window statistics (i.e., no statistics computed based
on a training set.)

Normalize img feature data in TFRecords using per dataset statistics (i.e statistics computed based on a
training set)

Input: TFRecord data set with flux window and difference image data.
Output: new TFRecord data set with per-image normalized flux window and per dataset difference image data.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import multiprocessing
from functools import partial
import logging

# local
from src_preprocessing.tf_util import example_util


def normalize_flux(flux_window):
    """
    Normalize a flux window based on per-window statistics

    Input: flux_window : List[float]

    Output: flux_window_norm : List[float]
    """
    # vectorized
    # load as np array
    flux_window = np.array(flux_window)
    median = np.median(flux_window)

    med_centered_flux = flux_window - median

    # Functions with a flux window with 100 indices, assuming 5tds window size
    center = len(flux_window) // 2  # 100 // 2 = 50

    half_td = (len(flux_window) // 5) // 2  # 100 // 5 = 20 // 2 = 10

    one_td_minimum = np.min(
        med_centered_flux[center - half_td : center + half_td]
    )  # Minmum value 1 transit duration around center

    # zero division eps term added to denominator to avoid division by zero
    norm_flux_window = med_centered_flux / (one_td_minimum + 1e-12)

    return norm_flux_window.tolist()


def normalize_img_feature(img_feature, img_set_med, img_set_std):
    """
    Normalize an img feature based on per-set statistics (diff_img, oot_img, snr_img)

    Input: img_feature : np.array of shape (X, X)
           img_set_med : median computed for img feature based on set (i.e. training set)
           img_set_std : std computed for img feature based on set (i.e. training set)

    Output: norm_img_feature: normalized np.array of shape (X, X)
    """
    # X_n = [x - median(X_train) ] / [std(X_train) + zero_division_eps]
    norm_img_feature = (img_feature - img_set_med) / (img_set_std + 1e-12)

    return norm_img_feature


def normalize_and_write_shard(src_tfrec_fp, dest_tfrec_fp, src_norm_stats_dir):
    if not src_tfrec_fp.exists():
        print("FILEPATH DOES NOT EXIST")
        print(src_tfrec_fp.name)
        return False

    try:
        with tf.io.TFRecordWriter(path=str(dest_tfrec_fp)) as writer:

            # Load source shard as tfrec dataset
            src_tfrecord_dataset = tf.data.TFRecordDataset(src_tfrec_fp)

            for string_record in src_tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()

                example.ParseFromString(string_record)

                # normalize flux window
                example_flux = np.array(
                    example.features.feature["flux"].float_list.value
                )

                example_norm_flux = normalize_flux(example_flux)

                example_util.set_float_feature(
                    ex=example,
                    name="flux_norm",
                    value=example_norm_flux,
                    allow_overwrite=False,
                )

                # normalize diff img, oot_img, snr_img
                img_dims = (33, 33)

                for img_feature in ["diff_img", "oot_img", "snr_img"]:
                    # get median/std etc from somewhere for training / val set
                    norm_stats = np.load(
                        src_norm_stats_dir / f"train_set_{img_feature}_stats.npy",
                        allow_pickle=True,
                    ).item()

                    img_feature_set_med, img_feature_set_std = (
                        norm_stats["median"],
                        norm_stats["std_mad"],
                    )

                    example_img_feature = tf.reshape(
                        tf.io.parse_tensor(
                            example.features.feature[img_feature].bytes_list.value[0],
                            tf.float32,
                        ),
                        img_dims,
                    ).numpy()

                    example_norm_img_feature = normalize_img_feature(
                        example_img_feature, img_feature_set_med, img_feature_set_std
                    )

                    example_util.set_tensor_feature(
                        ex=example,
                        name=f"{img_feature}_stdnorm",
                        value=example_norm_img_feature,
                        allow_overwrite=False,
                    )

                # write to new TFRecord
                writer.write(example.SerializeToString())
        return True
    except Exception as e:
        print(f"ERROR while processing shard: {e}")
        return False


if __name__ == "__main__":

    # Define split name when processing multiple split directories
    SPLIT_NAME = "val"  # TODO: normalize train, test

    # Define logging directory
    log_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/logging/norm_sets_v2/{SPLIT_NAME}_set"
    )  # TODO: train, test
    log_dir.mkdir(parents=True, exist_ok=True)

    # # Define source dataset directory & dest dataset directory with same structure + norm suffix
    src_dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split"
    )
    dest_dataset_dir = src_dataset_dir.parent / (str(src_dataset_dir.name) + "_norm_v2")

    # Define source/destination tfrec directory based on src dataset + split_name specification
    src_tfrec_dir = src_dataset_dir / "tfrecords" / SPLIT_NAME

    dest_tfrec_dir = dest_dataset_dir / "tfrecords" / SPLIT_NAME
    dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

    # Define destination norm stats directory for the split
    src_norm_stats_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/stats/TESS_exoplanet_dataset_11-25-2024_split_v4"
    )
    src_norm_stats_dir.mkdir(parents=True, exist_ok=True)

    # Define pool for multiprocessing
    pool = multiprocessing.Pool(processes=126)  # TODO: move up processes

    # Setup logger for overseeing parallelized processes
    logger = logging.getLogger(f"norm_logger_{SPLIT_NAME}")
    logger.setLevel(logging.INFO)

    log_path = Path(log_dir) / f"norm_{SPLIT_NAME}.log"
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    # Create list of shard_nums to process from 1 to NUM_SHARDS bound
    shards_to_process = [shard_num for shard_num in range(1, 8611 + 1)]

    results = []

    # Process shards in parallel using pool
    for shard_num in shards_to_process:
        logger.info(f"Processing chunk {shard_num}.")
        print((f"Processing chunk {shard_num}."))

        # define title in {SHARD_PREFIX}_{SHARD_NUM}-{NUM_SHARDS} format ex) test_shard_XXXX-8611 format
        SHARD_TITLE = (
            SPLIT_NAME + "_shard_" + str(shard_num).zfill(4) + "-" + str(8611).zfill(4)
        )

        # define src tfrec to process and dest tfrec to write to
        src_tfrec_fp = src_tfrec_dir / SHARD_TITLE
        dest_tfrec_fp = dest_tfrec_dir / ("norm_" + SHARD_TITLE)

        result = pool.apply_async(
            normalize_and_write_shard,
            args=(src_tfrec_fp, dest_tfrec_fp, src_norm_stats_dir),
        )
        results.append(result.get())

    pool.close()
    pool.join()

    print(f"TFRecord shards successfully processed: {sum(results)}")
    logger.info(f"TFRecord shards successfully processed: {sum(results)}")

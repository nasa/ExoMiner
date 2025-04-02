"""
Given a directory containing tfrecord shards, merge them until reaching an optimal number of examples.
"""

import pandas as pd
from pathlib import Path
import tensorflow as tf
from transit_detection.dataset_handling.resize.utils_optimally_merge import (
    create_split_example_per_shard_df,
    create_tfrecord_merge_map,
)


# def reformat_orig_shard_name(shard_name: str, prefix) -> str:
#     return prefix + shard_name


def convert_shard_to_fp(shard_name: str, shard_dir: Path) -> str:
    return str(shard_dir / shard_name)


def merge_shards(original_shard_fps: list[str], merged_shard_fp: str) -> None:
    with tf.io.TFRecordWriter(merged_shard_fp) as writer:
        for original_shard_fp in original_shard_fps:
            raw_dataset = tf.data.TFRecordDataset(original_shard_fp)
            for raw_record in raw_dataset:
                writer.write(raw_record.numpy())


if __name__ == "__main__":

    original_dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split_norm_v2/"
    )
    original_tfrec_dir = original_dataset_dir / "tfrecords"

    opt_dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split_norm_v2_opt/"
    )
    opt_dataset_dir.mkdir(parents=True, exist_ok=True)

    stats_output_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/stats/TESS_exoplanet_dataset_11-12-2024_split_norm_v2_opt/"
    )
    stats_output_dir.mkdir(parents=True, exist_ok=True)

    tfrec_dir = opt_dataset_dir / "tfrecords"
    tfrec_dir.mkdir(parent=True, exist_ok=True)

    resize_df = create_split_example_per_shard_df(
        tfrec_split_dir=original_tfrec_dir,
        output_dir=stats_output_dir,
        save_csv=True,
    )

    merge_map_df = create_tfrecord_merge_map(
        resize_df=resize_df, output_merge_map_csv_dir=stats_output_dir
    )

    for split_name in ["train", "val", "test"]:
        split_dir = tfrec_dir / split_name
        split_dir.mkdir(parent=True, exist_ok=True)

        split_df = merge_map_df[merge_map_df["split"] == split_name]

        for merged_shard_name, group_df in split_df.groupby("merged_shard"):
            print(f"Processing merged shard: {merged_shard_name}")

            original_shards = group_df["original_shard"].tolist()

            # # val_shard_4283-8611 -> norm_val_shard_4283-8611
            # original_shards = [
            #     reformat_orig_shard_name(shard_name, "norm_")
            #     for shard_name in original_shards
            # ]

            # norm_val_shard_4283-8611 -> prefix_dir/norm_val_shard_4283-8611
            original_shards = [
                convert_shard_to_fp(shard_name, original_tfrec_dir)
                for shard_name in original_shards
            ]

            print(f"Num original shards : {len(original_shards)}")

            merge_shards(original_shard_fps=original_shards, merged_shard_fp=Path(""))

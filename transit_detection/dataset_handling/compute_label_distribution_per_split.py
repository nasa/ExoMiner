import pandas as pd
import tensorflow as tf
import glob
from collections import defaultdict
from pathlib import Path


def get_tfrec_example_info(tfrec_fp: str) -> pd.DataFrame:
    raw_dataset = tf.data.TFRecordDataset([tfrec_fp])

    tfrec_fn = str(Path(tfrec_fp).name)
    print(f"\nProcessing tfrec_fn: {tfrec_fn}")

    row_dict = {
        "tfrec_fn": tfrec_fn,
        "label_1_count": 0,
        "label_0_count": 0,
        "total_examples": 0,
        "NTP": 0,
        "EB": 0,
        "CP": 0,
        "KP": 0,
        "NPC": 0,
        "NEB": 0,
    }

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        disposition = (
            example.features.feature["disposition"].bytes_list.value[0].decode("utf-8")
        )
        label = example.features.feature["label"].float_list.value[0]

        if disposition in row_dict:
            row_dict[disposition] += 1
        else:
            print(f"ERROR: disposition: {disposition} not in row_dict")

        if label == 1.0:
            row_dict["label_1_count"] += 1
        elif label == 0.0:
            row_dict["label_0_count"] += 1
        else:
            print(f"ERROR: label: {label} not in row_dict")

        row_dict["total_examples"] += 1

    return pd.DataFrame([row_dict])


def process_split(split_fp_pattern: str, csv_save_fp: str = "") -> pd.DataFrame:
    """
    input a pattern like: dataset/tfrecords/train/train_shard_????-????

    """
    tfrec_fps = glob.glob(split_fp_pattern)
    df_list = [get_tfrec_example_info(tfrec_fp=tfrec_fp) for tfrec_fp in tfrec_fps]
    final_df = pd.concat(df_list, ignore_index=True)
    if csv_save_fp:
        final_df.to_csv(csv_save_fp)
    return final_df


if __name__ == "__main__":
    # Process train and val:

    for split in ["train", "val", "test"]:
        print(f"PROCESSING SPLIT: {split}")
        split_fp_pattern = f"/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split_norm_v3/tfrecords/{split}/norm_{split}_shard_????-????"

        csv_save_dir = Path(
            "/nobackupp27/jochoa4/work_dir/job_runs/label_distribution_per_split_v2/"
        )
        csv_save_dir.mkdir(parents=True, exist_ok=True)
        csv_save_fp = csv_save_dir / f"label_count_per_shard_{split}.csv"

        process_split(split_fp_pattern=split_fp_pattern, csv_save_fp=str(csv_save_fp))
        print(f"Finished processing split: {split}")

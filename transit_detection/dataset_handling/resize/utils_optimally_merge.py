import tensorflow as tf
from pathlib import Path
import glob
import pandas as pd
import os


def create_example_per_shard_df(
    tfrec_pattern: str, output_dir="", split=""
) -> pd.DataFrame:
    tfrec_files = sorted(Path().glob(tfrec_pattern))
    print(f"Found {len(tfrec_files)} files matching pattern: {tfrec_pattern}")

    # Count examples per shard
    records = []
    for shard in tfrec_files:
        count = sum(1 for _ in tf.data.TFRecordDataset(shard))
        records.append({"tfrec_fn": shard.name, "total_examples": count})

    return pd.DataFrame(records)


def create_split_example_per_shard_df(
    tfrec_split_dir: str, output_dir: str, save_csv: bool = True
) -> pd.DataFrame:

    train_df = create_example_per_shard_df(
        Path(tfrec_split_dir) / "train" / "norm_train_shard_????-8611"
    )
    val_df = create_example_per_shard_df(
        Path(tfrec_split_dir) / "val" / "norm_val_shard_????-8611"
    )
    test_df = create_example_per_shard_df(
        Path(tfrec_split_dir) / "test" / "norm_test_shard_????-8611"
    )

    # train_df["split"] = "train"
    # val_df["split"] = "val"
    # test_df["split"] = "test"

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df = all_df.sort_values(by="total_examples", ascending=False)

    if save_csv:
        all_df.to_csv((Path(output_dir) / "all_shard_example_counts.csv"), index=False)
        print("Saved shard_example_counts.csv")

    return all_df


import tensorflow as tf
import pandas as pd

from utils_merge_by_size import create_map_from_df, sort_df_by_examples, get_split_type


def create_tfrecord_merge_map(
    resize_df,
    output_merge_map_csv_dir,
    min_examples: int = 5000,
    max_extra: int = 1000,
):
    # group original tfrecord shards into new merged shards based on minimum example threshold
    df = resize_df

    filename_col, total_examples_col = "tfrec_fn", "total_examples"

    df["split"] = df[filename_col].apply(get_split_type)

    splits = ["train", "val", "test"]

    split_map_dfs = []

    for split in splits:
        merge_map_df = pd.DataFrame(columns=["merged_shard", "original_shard"])

        split_df = df[df["split"] == split]

        split_df = split_df.sort_values(by=total_examples_col, ascending=True)

        # Merge shards using greedy approach
        current_group = []
        current_count = 0
        shard_index = 1

        """
        for file, examples (row in df):
            

           if adding example puts us over hard limit ->
                finalize group and start new group with current example
            otherwise
                add to group
            
            what if shard > limits:
                make its own shard

        
        """

        for _, row in split_df.iterrows():
            tfrec_fn, examples = row[filename_col], row[total_examples_col]

            # If the current shard is larger than 'min_examples', store it as its own group
            if examples >= min_examples:
                merged_filename = f"{split}_resized_shard_{str(shard_index).zfill(4)}"
                temp_df = pd.DataFrame(
                    {
                        "resized_shard": [merged_filename],
                        "original_shard": [tfrec_fn],
                        "resized_examples": [examples],
                    }
                )
                merge_map_df = pd.concat([merge_map_df, temp_df], ignore_index=True)
                shard_index += 1
                continue

            # If adding this shard would exceed the strict max limit, finalize the current group

            # Otherwise

            if current_count + examples < min_examples:
                current_group.append((tfrec_fn, examples))
                current_count += examples

            else:
                # If adding this shard would exceed the strict max limit, finalize the current group
                if current_count + examples > min_examples + max_extra:
                    merged_filename = (
                        f"{split}_resized_shard_{str(shard_index).zfill(4)}"
                    )
                    for orig_shard, count in current_group:
                        temp_df = pd.DataFrame(
                            {
                                "resized_shard": [merged_filename],
                                "original_shard": [orig_shard],
                                "resized_examples": [current_count],
                            }
                        )
                        merge_map_df = pd.concat(
                            [merge_map_df, temp_df], ignore_index=True
                        )
                    # start new group
                    shard_index += 1
                    current_group = [(tfrec_fn, examples)]
                    current_count = examples
                else:
                    # merge it
                    current_group.append((tfrec_fn, examples))
                    current_count += examples

        # # Add the last group if it exists
        # if current_group:
        #     merged_filename = f"{split}_merged_shard_{str(shard_index).zfill(4)}"
        #     for orig_shard in current_group:
        #         temp_df = pd.DataFrame({"merged_shard" : [merged_filename], "original_shard": [orig_shard], "merged_examples" : [current_count]})
        #         merge_map_df = pd.concat([merge_map_df, temp_df], ignore_index=True)

        merge_map_df["split"] = split
        split_map_dfs.append(merge_map_df)

    output_merge_map_df = pd.DataFrame(
        columns=["merged_shard", "original_shard", "split", "merged_examples"]
    )

    for split_df in split_map_dfs:
        output_merge_map_df = pd.concat(
            [output_merge_map_df, split_df], ignore_index=True
        )

    output_merge_map_df.to_csv(
        output_merge_map_csv_dir + f"split_output_merge_map.csv", index=False
    )

    return merge_map_df

from pathlib import Path
import glob
import tensorflow as tf
import pandas as pd


"Flip a dataset to use bytes_list instead of float_list while maintaining value"

if __name__ == "__main__":

    original_dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split_norm_v2_opt/tfrecords"
    )

    target_dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/job_runs/log_dataset_examples/TESS_exoplanet_dataset_11-12-2024_split_norm_20percent_subset"
    )

    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    original_tfrec_dir = original_dataset_dir / "tfrecords"
    target_tfrec_dir = target_dataset_dir / "tfrecords"
    target_tfrec_dir.mkdir(parents=True, exist_ok=True)

    for split in ["test", "train", "val"]:
        original_split_dir = original_tfrec_dir / split
        target_split_dir = target_tfrec_dir / split

        target_split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing split: {split}")

        print(f"Using original_split_dir: {original_split_dir}")
        print(f"Using target_split_dir: {target_split_dir}")

        shard_pattern = f"norm_{split}_shard_????-????"
        print(f"Using shard_pattern: {shard_pattern}")

        tfrec_fp_pattern = str(original_split_dir / shard_pattern)
        tfrec_shard_fps = glob.glob(tfrec_fp_pattern)

        print(f"Found {len(tfrec_shard_fps)} using pattern {tfrec_fp_pattern}")

        for tfrec_shard_fp in tfrec_shard_fps:

            target_tfrec_shard_fp = target_split_dir / Path(tfrec_shard_fp).name
            print(f"Writing to {target_tfrec_shard_fp}")

            with tf.io.TFRecordWriter(target_tfrec_shard_fp) as writer:
                # iterate through the source shard
                tfrec_dataset = tf.data.TFRecordDataset(str(tfrec_shard_fp))

                for str_record in tfrec_dataset.as_numpy_iterator():
                    example = tf.train.Example()
                    example.ParseFromString(str_record)

                    original_label = example.features.feature["label"].float_list.value[
                        0
                    ]

                    if original_label == 0.0:
                        new_label = b"0"
                    elif original_label == 1.0:
                        new_label = b"1"
                    else:
                        print(
                            f"Error with type of original_label: {original_label} of type: {type(original_label)}"
                        )

                    # Remove current label type
                    example.features.feature["label"].ClearField("float_list")

                    # Add new label
                    example.features.feature["label"].bytes_list.value.append(new_label)

                    # Write example
                    writer.write(example.SerializeToString())

        print(f"Found {len(split_examples)} examples in split {split}")
        split_df = pd.DataFrame(split_examples)
        csv_output_dir = output_dir / f"{split}_dataset_examples.csv"
        split_df.to_csv(str(csv_output_dir), index=False)
        print(f"Saved log to: {str(csv_output_dir)}")
        print(f"Finished processing split: {split}")

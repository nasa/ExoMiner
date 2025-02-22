"""

"""

import tensorflow as tf
import pandas as pd
from pathlib import Path

from utils_merge_by_size import sort_df_by_examples, get_split_type

def create_tfrecord_merge_map(input_csv_path, output_merge_map_csv_dir, N: int = 5500):
    # group original tfrecord shards into new merged shards based on minimum example threshold
    df = pd.read_csv(input_csv_path)

    filename_col, total_examples_col = "tfrec_fn", "total_examples"

    df["split"] = df[filename_col].apply(get_split_type)

    splits = ["train", "val", "test"]

    merge_map = []

    for split in splits:
        merge_map_df = pd.DataFrame(columns=["merged_shard", "original_shard"])

        split_df = df[df["split"] == split]

        split_df = split_df.sort_values(by=total_examples_col, ascending=False).reset_index(drop=True)

        """
        for file, examples (row in df):
            

           if adding example puts us over hard limit ->
                finalize group and start new group with current example
            otherwise
                add to group
            
            what if shard > limits:
                make its own shard

        
        """

        bins = [] # List of bins, each containing TFRecord file names
        bin_totals = [] # Total examples per bin
        

        for _, row in split_df.iterrows():
            tfrec_fn, examples = row[filename_col], row[total_examples_col]

            placed = False

            for i in range((len(bins))):
                if bin_totals[i] + examples <= N:
                    bins[i].append(tfrec_fn)
                    bin_totals[i] += examples
                    placed = True
                    break

            if not placed:
                # Step 3: Create new bin if no fit
                bins.append([tfrec_fn])
                bin_totals.append(examples)

        
        for bin_idx, (bin_files, total_examples) in enumerate(zip(bins, bin_totals), start=1):
            merged_filename = f"{split}_resized_shard_{str(bin_idx).zfill(4)}-{str(len(bins)).zfill(4)}"
            merge_map.append(
                {
                    "merged_fn" : merged_filename,
                    "merged_examples" : total_examples,
                    "original_fn" : bin_files,
                    "split" : split
                }
            )
            
    merge_df = pd.DataFrame(merge_map)

    merge_df.to_csv(output_merge_map_csv_dir + f"split_resize_merge_map.csv", index=False)

    return merge_map_df
    

if __name__ == "__main__":
    examples_per_file_csv = "/Users/jochoa4/Desktop/test_transfers/examples_per_split_filename_0001-8611.csv"
    output_csv_map_dir = "/Users/jochoa4/Desktop/test_runs/create_resize_df_by_split/"
    Path(output_csv_map_dir).mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(examples_per_file_csv)
    input_df = sort_df_by_examples(input_df, ascending=True)

    merged_map = create_tfrecord_merge_map(examples_per_file_csv, output_csv_map_dir)
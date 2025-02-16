"""

"""

import tensorflow as tf
import pandas as pd

from utils_merge_by_size import create_map_from_df, sort_df_by_examples, get_split_type

def create_tfrecord_merge_map(input_csv_path, output_merge_map_csv_dir, min_examples: int = 5000, max_extra: int = 1000):
    # group original tfrecord shards into new merged shards based on minimum example threshold
    df = pd.read_csv(input_csv_path)

    filename, total_examples = "tfrec_fn", "total_examples"

    df["split"] = df[filename].apply(get_split_type)

    splits = ["train", "val", "test"]

    split_map_dfs = []

    for split in splits:
        merge_map_df = pd.DataFrame(columns=["merged_shard", "original_shard"])

        split_df = df[df["split"] == split]

        split_df = split_df.sort_values(by=total_examples, ascending=True)
        
        # Merge shards using greedy approach
        current_group = []
        current_count = 0
        shard_index =  1

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
            tfrec_fn, examples = row[filename], row[total_examples]

            # If the current shard is larger than 'min_examples', store it as its own group
            if examples >= min_examples:
                merged_filename = f"{split}_merged_shard_{str(shard_index).zfill(4)}"
                temp_df = pd.DataFrame({"merged_shard" : [merged_filename], "original_shard": [tfrec_fn], "merged_examples" : [examples]})
                merge_map_df = pd.concat([merge_map_df, temp_df], ignore_index=True)
                shard_index += 1
                continue

            # If adding this shard would exceed the strict max limit, finalize the current group
            if current_count + examples < min_examples:
                current_group.append((tfrec_fn, examples))
                current_count += examples

            else:
                if current_count + examples > min_examples + max_extra:
                    merged_filename = f"{split}_merged_shard_{str(shard_index).zfill(4)}"
                    for orig_shard, count in current_group:
                        temp_df = pd.DataFrame({"merged_shard" : [merged_filename], "original_shard": [orig_shard], "merged_examples" : [current_count]})
                        merge_map_df = pd.concat([merge_map_df, temp_df], ignore_index=True)
                    # start new group    
                    shard_index += 1
                    current_group = [(tfrec_fn, examples)]
                    current_count = examples
                else:
                    # merge it
                    current_group.append((tfrec_fn, examples))
                    current_count += examples


        # Add the last group if it exists
        if current_group:
            merged_filename = f"{split}_merged_shard_{str(shard_index).zfill(4)}"
            for orig_shard in current_group:
                temp_df = pd.DataFrame({"merged_shard" : [merged_filename], "original_shard": [orig_shard], "merged_examples" : [current_count]})
                merge_map_df = pd.concat([merge_map_df, temp_df], ignore_index=True)

        merge_map_df["split"] = split
        split_map_dfs.append(merge_map_df)

    output_merge_map_df = pd.DataFrame(columns=["merged_shard", "original_shard", "split", "merged_examples"])

    for split_df in split_map_dfs:
        output_merge_map_df = pd.concat([output_merge_map_df, split_df], ignore_index=True)

    output_merge_map_df.to_csv(output_merge_map_csv_dir + f"split_output_merge_map.csv", index=False)

    return merge_map_df
    

if __name__ == "__main__":
    examples_per_file_csv = "/Users/jochoa4/Desktop/test_transfers/examples_per_split_filename_0001-8611.csv"
    output_csv_map_dir = "/Users/jochoa4/Desktop/test_transfers/"

    input_df = pd.read_csv(examples_per_file_csv)
    input_df = sort_df_by_examples(input_df, ascending=True)
    
    merged_map = create_tfrecord_merge_map(examples_per_file_csv, output_csv_map_dir)

    


    




"""
Given a merged auxillary table containing info about tfrecords and list of targets, rename target file to a proper upper bound

"""
import numpy as np
import pandas as pd


def get_reformatted_title(chunk_num, upper_bound):
    old_title_format = f"test_shard_0001-{str(chunk_num).zfill(4)}"
    new_title_format = f"shard_{str(chunk_num).zfill(4)}-{str(upper_bound).zfill(4)}"

    return (old_title_format, new_title_format)


if __name__ == "__main__":

    input_aux_table = "/Users/jochoa4/Desktop/test_transfers/merged_data_tbl_chunks_0001-8611.csv"
    output_aux_table ="/Users/jochoa4/Desktop/test_transfers/raw_data_tbl_chunks_0001-8611.csv"

    title_map = {}

    for chunk_num in range(1, 8611 + 1):
        old_title, new_title = get_reformatted_title(chunk_num, 8611)
        title_map[old_title] = new_title

    df = pd.read_csv(input_aux_table)

    def replace_title(old_title):
        if old_title not in title_map:
            raise KeyError(f"Title '{old_title}' not found in hash map")
        return title_map[old_title]

    df["tfrec_fn"] = df["tfrec_fn"].apply(replace_title)

    df.to_csv(output_aux_table, index=False)

    print(f"Processed CSV saved as {output_aux_table.split('/')[-1]}")
    



    
        
    








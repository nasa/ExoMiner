"""
Given a merged auxillary table containing info about tfrecords and list of targets, rename target file containing tce based on its predefined split.

Additionally, reformat title if neccesary
"""
import numpy as np
import pandas as pd

def update_aux_fp_based_on_target(input_csv_path, output_csv_path, target_column, filename_column, split_sets):

    df = pd.read_csv(input_csv_path)

    def prepend_prefix(row):
        target_value = row[target_column]

        if target_value in split_sets['train_set']:
            prefix = "train_"
        elif target_value in split_sets['val_set']:
            prefix = "val_"
        elif target_value in split_sets['test_set']:
            prefix = "test_"
        else:
            raise ValueError(f"Target '{target_value}' not found in any of the provided sets")
        
        return prefix + row[filename_column]
    
    df[filename_column] = df.apply(prepend_prefix, axis=1)
    df.to_csv(output_csv_path, index=False)

    print(f"Processed CSV saved as {output_csv_path}")


if __name__ == "__main__":

    input_aux_table_path = "/Users/jochoa4/Desktop/test_transfers/raw_data_tbl_chunks_0001-8611.csv"
    output_aux_table_path = "/Users/jochoa4/Desktop/test_transfers/split_data_tbl_chunks_0001-8611.csv"

    split_sets = np.load("/Users/jochoa4/Desktop/test_transfers/split_dataset_targets.npy", allow_pickle=True).item()

    update_aux_fp_based_on_target(input_aux_table_path, output_aux_table_path, "target_id", "tfrec_fn", split_sets)




import pandas as pd

def sum_examples_by_file_and_write_to_txt(input_csv_path, output_csv_path, output_txt_path, filename_column, positive_column, negative_column):
    """
    Groups by filename, sums positive and negative counts, sorts by total count,
    and writes results to both a text file and a CSV.

    Input:
        input_csv_path: (str) path to input CSV file.
        output_csv_path: (str) path to output CSV file.
        output_txt_path: (str) path to output txt file.
        filename_column: (str) title of column holding filename
        postiive_column: (str) title of column with postiive examples
        negative_column: (str) title of column with negative examples
    """

    df = pd.read_csv(input_csv_path)

    grouped_df = df.groupby(filename_column)[[positive_column, negative_column]].sum().reset_index()

    grouped_df["total_examples"] = grouped_df[positive_column] + grouped_df[negative_column]
    sorted_df = grouped_df.sort_values(by="total_examples", ascending=False)

    with open(output_txt_path, "w") as f:
        for _, row in sorted_df.iterrows():
            f.write(f"{row[filename_column]}: + {row[positive_column]}, - {row[negative_column]}, Total: {row['total_examples']}\n")

    sorted_df[[filename_column, "total_examples"]].to_csv(output_csv_path, index=False)
    
    print(f"Processed and sorted data saved to {output_txt_path}")


if __name__ == "__main__":

    input_csv_path = "/Users/jochoa4/Desktop/test_transfers/split_data_tbl_chunks_0001-8611.csv"
    output_csv_path = "/Users/jochoa4/Desktop/test_transfers/examples_per_split_filename_0001-8611.csv"
    output_txt_path = "/Users/jochoa4/Desktop/test_transfers/sorted_chunks_by_examples_after_split_0001-8611.txt"

    sum_examples_by_file_and_write_to_txt(input_csv_path, output_csv_path, output_txt_path, "tfrec_fn", "n_transit_examples", "n_not_transit_examples")
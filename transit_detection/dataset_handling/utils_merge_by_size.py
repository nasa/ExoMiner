import pandas as pd

def create_map_from_df(df, key_column, value_column) -> dict:
    return dict(zip(df[key_column], df[value_column]))


def sort_df_by_examples(df, column_key="total_examples", ascending=True):
    sorted_df = df.sort_values(by=column_key, ascending=ascending)
    return sorted_df

def get_split_type(filename):
    if filename.startswith("train"):
        return "train"
    elif filename.startswith("val"):
        return "val"
    elif filename.startswith("test"):
        return "test"
    else:
        return "unknown"
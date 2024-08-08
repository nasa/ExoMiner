"""
Count the number of unique brown dwarfs in each shard and each set(test, train, val).
"""

# 3rd party
import pandas as pd

df = pd.read_csv("/Users/agiri1/Desktop/ExoBD_Datasets/train_test_split/trainset.csv")
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "Kepler_Dataset")]["target_id"])))
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "TESS_Dataset")]["target_id"])))
df = pd.read_csv("/Users/agiri1/Desktop/ExoBD_Datasets/train_test_split/valset.csv")
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "Kepler_Dataset")]["target_id"])))
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "TESS_Dataset")]["target_id"])))
df = pd.read_csv("/Users/agiri1/Desktop/ExoBD_Datasets/train_test_split/testset.csv")
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "Kepler_Dataset")]["target_id"])))
print(len(set(df[(df["label"] == "BD") & (df["shard"] == "TESS_Dataset")]["target_id"])))
import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
import sklearn
import matplotlib.pyplot as plt
import numpy as np
"""
Concatenate the CSV output for each of the n folds in n-Fold cross-validation to create test_predictions for the entire dataset.
"""
total_val_df = []
total_test_df = []
total_train_df = []
n=3
for i in range(n):
    total_val_df.append(pd.read_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/model{i}/ranked_predictions_valset.csv"))
    total_test_df.append(pd.read_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/model{i}/ranked_predictions_testset.csv"))
    total_train_df.append(pd.read_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/model{i}/ranked_predictions_trainset.csv"))
total_val_df = pd.concat(total_val_df)
total_test_df = pd.concat(total_test_df)
total_train_df = pd.concat(total_train_df)
total_val_df.to_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/ensemble_ranked_predictions_valset.csv")
total_test_df.to_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/ensemble_ranked_predictions_testset.csv")
total_train_df.to_csv(f"/Users/agiri1/Desktop/ExoPlanet/output_dir/ensemble_ranked_predictions_trainset.csv")
"""
Concatenate the CSV output for each of the n folds in n-Fold cross-validation to create test_predictions for the entire
dataset.
"""

# 3rd party
import pandas as pd

n = 3  # number of CV iterations

total_val_df = []
total_test_df = []
total_train_df = []
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

import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
import sklearn
import matplotlib.pyplot as plt
sets = ["train", "test", "val"]
model = "model_without_centroid+diff_img_5Fold_CV"
"""
Plot the correlation between the score and planet radius to see if the model is classifying bigger objects as Brown Dwarfs at a higher rate.
Graph plotted for each set in total dataset and for each set in each fold when doing cross-validation
"""
def plot_total_graph(model):
    for i in sets:
        df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/ensemble_ranked_predictions_{i}set.csv")
        plt.title(f"Score vs Planet Radius ({i})")
        plt.scatter(df["score"], df["tce_period"], label="Planets")
        plt.scatter(df[(df['label'] == "BD")]['score'], df[(df['label'] == "BD")]['tce_period'], c="red", label="Brown Dwarfs")
        plt.xlabel("Score")
        plt.ylabel("Planet Radius")
        plt.legend()
        plt.savefig(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/plnt_radius_vs_score_{i}_set.png")
        plt.show()
def plot_sub_graph(model, num_folds = 5):
    for j in range(num_folds):
        for i in sets:
            df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{j}/ranked_predictions_{i}set.csv")
            plt.title(f"Score vs Planet Radius ({i})")
            plt.scatter(df["score"], df["tce_period"], label="Planets")
            plt.scatter(df[(df['label'] == "BD")]['score'], df[(df['label'] == "BD")]['tce_period'], c="red", label="Brown Dwarfs")
            plt.xlabel("Score")
            plt.ylabel("Planet Radius")
            plt.legend()
            plt.savefig(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}//model{j}/plnt_radius_vs_score_{i}_set.png")
            plt.show()
plot_total_graph(model)
plot_sub_graph(model)
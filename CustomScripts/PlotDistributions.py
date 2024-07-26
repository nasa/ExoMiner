import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
import sklearn
import matplotlib.pyplot as plt
import numpy as np
sets = ["train","test","val"]
model = "model_without_centroid+diff_img_5Fold_CV"
"""
Plot the Distribution of the score of Brown Dwarfs compared to the score of Exoplanets to visualize the separation in the distribution of exoplanet scores and brown dwarf scores.
Graph plotted for each set in total dataset and for each set in each fold when doing cross-validation
"""
def plot_total_graph(model):
    for i in sets:
        df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/ensemble_ranked_predictions_{i}set.csv")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt1 = ax1.hist(df[((df['label'] == "PC") | (df['label'] == "KP") | (df['label'] == "CP"))]['score'], label="Planet Score",alpha = 0.3,  color = "blue")
        plt2 = ax2.hist(df[((df['label'] == "BD"))]['score'],
            label="Brown Dwarfs Score", alpha=0.3, color= "Orange")

        ax1.set_label("Planet")
        ax2.set_label("Brown Dwarf")
        ax1.set_xlabel("Score")
        ax1.set_ylabel("# of instances(Planet)")
        ax2.set_ylabel("# of instances(Brown Dwarf)")
        ax1.set_title(f"Brown Dwarfs Distribution vs Planet Distribution({i})")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.savefig(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/{i}_set_score_dist.png")
        plt.show()


def plot_sub_graph(model, num_folds=5):
    for j in range(num_folds):
        for i in sets:
            df = pd.read_csv(
                f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{j}/ranked_predictions_{i}set.csv")
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            plt1 = ax1.hist(df[((df['label'] == "PC") | (df['label'] == "KP") | (df['label'] == "CP"))]['score'],
                            label="Planet Score", alpha=0.3, color="blue")
            plt2 = ax2.hist(df[((df['label'] == "BD"))]['score'],
                            label="Brown Dwarfs Score", alpha=0.3, color="Orange")

            ax1.set_label("Planet")
            ax2.set_label("Brown Dwarf")
            ax1.set_xlabel("Score")
            ax1.set_ylabel("# of instances(Planet)")
            ax2.set_ylabel("# of instances(Brown Dwarf)")
            ax1.set_title(f"Brown Dwarfs Distribution vs Planet Distribution({i})")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            plt.savefig(
                f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{j}/{i}_set_score_dist.png")
            plt.show()
plot_total_graph(model)
plot_sub_graph(model)
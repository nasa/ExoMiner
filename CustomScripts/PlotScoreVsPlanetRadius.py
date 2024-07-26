import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
import sklearn
import matplotlib.pyplot as plt
sets = ["train", "test", "val"]
"""
Plot the misclassified and correctly classified Brown Dwarfs with respect to planet radius and period to see if the misclassifications are concentrated in a certain area.
Graph plotted for each set in total dataset and for each set in each fold when doing cross-validation
"""
def plot_total_graph(model):
    for i in sets:
        # df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/total_{i}_predictions.csv")
        df = pd.read_csv(
            f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/ensemble_ranked_predictions_{i}set.csv")
        plt.title(f"Planet Radius vs TCE Period ({i})")
        plt.scatter(df[df['score'] > 0.5]["tce_prad"], df[df['score'] > 0.5]["tce_period"],
                    label="Correctly Classified Planets", alpha=0.8)
        plt.scatter(df[(df['label'] == "BD") & (df['score'] > 0.5)]['tce_prad'],
                    df[(df['label'] == "BD") & (df['score'] > 0.5)]['tce_period'], c="red",
                    label=" Misclassified Brown Dwarfs", alpha=0.8)
        plt.scatter(df[df['score'] < 0.5]["tce_prad"], df[df['score'] < 0.5]["tce_period"], c="orange",
                    label="Misclassified Planets", alpha=0.8)
        plt.scatter(df[(df['label'] == "BD") & (df['score'] < 0.5)]['tce_prad'],
                    df[(df['label'] == "BD") & (df['score'] < 0.5)]['tce_period'], c="purple",
                    label=" Correctly Classified Brown Dwarfs", alpha=0.8)
        plt.xlabel("TCE Period (Log)")
        plt.xlabel("TCE Radius (Log)")
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.savefig(
            f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/{i}_set_misclassification_plot.png")
        plt.show()
def plot_sub_graph(model, num_folds = 5):
    for j in range(num_folds):
        for i in sets:
            # df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/total_{i}_predictions.csv")
            df = pd.read_csv(
                f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{j}/ranked_predictions_{i}set.csv")
            plt.title(f"Planet Radius vs TCE Period ({i})")
            plt.scatter(df[df['score'] > 0.5]["tce_prad"], df[df['score'] > 0.5]["tce_period"],
                        label="Correctly Classified Planets", alpha=0.8)
            plt.scatter(df[(df['label'] == "BD") & (df['score'] > 0.5)]['tce_prad'],
                        df[(df['label'] == "BD") & (df['score'] > 0.5)]['tce_period'], c="red",
                        label=" Misclassified Brown Dwarfs", alpha=0.8)
            plt.scatter(df[df['score'] < 0.5]["tce_prad"], df[df['score'] < 0.5]["tce_period"], c="orange",
                        label="Misclassified Planets", alpha=0.8)
            plt.scatter(df[(df['label'] == "BD") & (df['score'] < 0.5)]['tce_prad'],
                        df[(df['label'] == "BD") & (df['score'] < 0.5)]['tce_period'], c="purple",
                        label=" Correctly Classified Brown Dwarfs", alpha=0.8)
            plt.xlabel("TCE Period (Log)")
            plt.xlabel("TCE Radius (Log)")
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.savefig(
                f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{j}/{i}_set_misclassification_plot.png")
            plt.show()
model = "model_without_centroid+diff_img_5Fold_CV"
plot_total_graph(model)
plot_sub_graph(model)

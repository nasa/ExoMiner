"""
Plot the Distribution of the score of Brown Dwarfs compared to the score of Exoplanets to visualize the separation in
the distribution of exoplanet scores and brown dwarf scores. Graph plotted for each set in total dataset and for each
set in each fold when doing cross-validation.
"""

# 3rd party
import pandas as pd
import matplotlib.pyplot as plt


def plot_total_graph(df, savefp, fold_i):

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    _ = ax1.hist(df[((df['label'] == "PC") | (df['label'] == "KP") | (df['label'] == "CP"))]['score'],
                 label="Planet Score", alpha=0.3, color="blue")
    _ = ax2.hist(df[((df['label'] == "BD"))]['score'], label="Brown Dwarfs Score", alpha=0.3, color="Orange")

    ax1.set_label("Planet")
    ax2.set_label("Brown Dwarf")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("# of instances(Planet)")
    ax2.set_ylabel("# of instances(Brown Dwarf)")
    ax1.set_title(f"Brown Dwarfs Distribution vs Planet Distribution({fold_i})")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.savefig(savefp)
    plt.show()


def plot_sub_graph(savefp, fold_i):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    _ = ax1.hist(df[((df['label'] == "PC") | (df['label'] == "KP") | (df['label'] == "CP"))]['score'],
                    label="Planet Score", alpha=0.3, color="blue")
    _ = ax2.hist(df[((df['label'] == "BD"))]['score'],
                    label="Brown Dwarfs Score", alpha=0.3, color="Orange")

    ax1.set_label("Planet")
    ax2.set_label("Brown Dwarf")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("# of instances(Planet)")
    ax2.set_ylabel("# of instances(Brown Dwarf)")
    ax1.set_title(f"Brown Dwarfs Distribution vs Planet Distribution({fold_i})")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.savefig(savefp)
    plt.show()


datasets = ["train", "test", "val"]
model = "model_without_centroid+diff_img_5Fold_CV"
num_folds = 5

for dataset in datasets:

    # for ensemble
    df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/ensemble_ranked_predictions_{dataset}set.csv")
    savefp = f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/{dataset}_set_score_dist.png"

    # for each cv iteration
    for fold_i in range(num_folds):

        plot_total_graph(df, savefp, fold_i)

        df = pd.read_csv(f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{fold_i}/ranked_predictions_{dataset}set.csv")
        savefp = f"/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/{model}/model{fold_i}/{dataset}_set_score_dist.png"
        plot_sub_graph(savefp, fold_i)

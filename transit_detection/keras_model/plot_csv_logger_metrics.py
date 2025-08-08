import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


"""
dict_keys(['loss', 'binary_accuracy', 'precision', 'recall', 'prec_thr', 'rec_thr', 'auc_pr', 'auc_roc', 'tp', 'fp', 'tn', 'fn', 'val_loss', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_prec_thr', 'val_rec_thr', 'val_auc_pr', 'val_auc_roc', 'val_tp', 'val_fp', 'val_tn', 'val_fn'])
"""


def plot_history_metrics_old(csv_log_fp, save_fp) -> None:
    """Expects path to csv_log from CSVLogger keras callback or equivalent form"""

    df = pd.read_csv(csv_log_fp)
    # df = df[:-10]
    epochs = df["epoch"] + 1  # offset zero index for all epoch

    train_loss = df["loss"]  # .clip(None, 1)
    val_loss = df["val_loss"]
    train_auc_pr = df["auc_pr"]
    val_auc_pr = df["val_auc_pr"]

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.title("Validation Loss")
    plt.yscale("log")
    plt.legend()
    # plt.xticks(epochs[::5])

    # Plot PR Auc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc_pr, label="Training PR AUC")
    plt.plot(epochs, val_auc_pr, label="Validation PR AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR AUC")
    plt.title("Training & Validation PR AUC")
    plt.legend()
    # plt.xticks(epochs[::5])

    plt.savefig(save_fp, format="png")


def plot_history_metrics(csv_log_fp: str, save_fp: str) -> None:
    """Reads a Keras-style CSV log and writes out a multipanel metric plot."""

    df = pd.read_csv(csv_log_fp)
    # df = df[:-20]
    df["epoch"] = df["epoch"] + 1

    # Derive F1 if we have precision & recall
    if {"precision", "recall", "val_precision", "val_recall"}.issubset(df.columns):
        df["f1"] = (
            2
            * df["precision"]
            * df["recall"]
            / (df["precision"] + df["recall"] + 1e-12)
        )
        df["val_f1"] = (
            2
            * df["val_precision"]
            * df["val_recall"]
            / (df["val_precision"] + df["val_recall"] + 1e-12)
        )

    # Pick which metrics to plot
    metrics = [
        ("loss", "Loss", True),  # (column, y-label, log-scale?)
        ("binary_accuracy", "Accuracy", False),
        ("precision", "Precision", False),
        ("recall", "Recall", False),
        ("f1", "F1-score", False),
        ("auc_pr", "AUC-PR", False),
        ("auc_roc", "AUC-ROC", False),
    ]
    # keep only those present in our df
    metrics = [
        m for m in metrics if m[0] in df.columns and ("val_" + m[0]) in df.columns
    ]

    # Plot
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, ylabel, use_log) in zip(axes, metrics):
        ax.plot(df["epoch"], df[col], label=f"train {ylabel}")
        ax.plot(df["epoch"], df["val_" + col], label=f"val {ylabel}")
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend(loc="best", fontsize="small")

    # Mark best epoch on AUC-PR
    if "val_auc_pr" in df.columns:
        best = df["val_auc_pr"].idxmax()
        best_epoch = df.loc[best, "epoch"]
        axes[metrics.index(("auc_pr", "AUC-PR", False))].axvline(
            best_epoch, color="gray", linestyle="--", linewidth=1
        )
        axes[metrics.index(("auc_pr", "AUC-PR", False))].text(
            best_epoch,
            df["val_auc_pr"].max(),
            f"best @ epoch {int(best_epoch)}",
            va="bottom",
            ha="right",
            fontsize="small",
        )

    # save
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(save_fp, dpi=300)
    plt.close(fig)


experiment_dirs = Path(
    "/Users/jochoa4/Desktop/pfe_transfers/experiments_08-04-2025"
).glob("train_model*")
for experiment_dir in experiment_dirs:

    csv_log_fp = experiment_dir / "training_metrics_log.csv"
    assert csv_log_fp.exists(), f"ERROR: csv_log_fp - {csv_log_fp} does not exist!"
    save_dir = Path(
        f"/Users/jochoa4/Desktop/plots/experiments_08-04-2025/{experiment_dir.name}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    save_fp = save_dir / "metrics.png"

    plot_history_metrics(csv_log_fp=csv_log_fp, save_fp=save_fp)

    save_fp = save_dir / "loss_auc_pr_plot.png"

    plot_history_metrics_old(csv_log_fp=csv_log_fp, save_fp=save_fp)

# train_dir_name = "experiments_07-29-2025/train_model_TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_low_lr"

# csv_log_fp = (
#     f"/Users/jochoa4/Desktop/pfe_transfers/{train_dir_name}/training_metrics_log.csv"
# )

# save_dir = Path(f"/Users/jochoa4/Desktop/plots/{train_dir_name}")
# save_dir.mkdir(parents=True, exist_ok=True)

# save_fp = save_dir / "metrics.png"

# plot_history_metrics(csv_log_fp=csv_log_fp, save_fp=save_fp)

# save_fp = save_dir / "loss_auc_pr_plot.png"

# plot_history_metrics_old(csv_log_fp=csv_log_fp, save_fp=save_fp)

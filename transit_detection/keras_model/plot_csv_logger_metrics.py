import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

"""
dict_keys(['loss', 'binary_accuracy', 'precision', 'recall', 'prec_thr', 'rec_thr', 'auc_pr', 'auc_roc', 'tp', 'fp', 'tn', 'fn', 'val_loss', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_prec_thr', 'val_rec_thr', 'val_auc_pr', 'val_auc_roc', 'val_tp', 'val_fp', 'val_tn', 'val_fn'])
"""


def plot_history_metrics(csv_log_fp, save_fp) -> None:
    """Expects path to csv_log from CSVLogger keras callback or equivalent form"""

    df = pd.read_csv(csv_log_fp)
    epochs = df["epoch"] + 1  # offset zero index for all epoch

    train_loss = df["loss"]
    # val_loss = df["val_loss"]
    train_auc_pr = df["auc_pr"]
    val_auc_pr = df["val_auc_pr"]

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    # plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.title("Training & Validation Loss")
    # plt.title("Validation Loss")
    plt.yscale("log")
    plt.legend()

    # Plot PR Auc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc_pr, label="Training PR AUC")
    plt.plot(epochs, val_auc_pr, label="Validation PR AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR AUC")
    plt.title("Training & Validation PR AUC")
    plt.legend()

    plt.savefig(save_fp, format="png")


save_dir = Path("/Users/jochoa4/Desktop/plots/train_keras_model_min_snr_20")
save_dir.mkdir(parents=True, exist_ok=True)

save_fp = save_dir / "loss_auc_pr_plot.png"

csv_log_fp = "/Users/jochoa4/Desktop/pfe_transfers/train_keras_model_min_snr_20/training_metrics_log.csv"
plot_history_metrics(csv_log_fp=csv_log_fp, save_fp=save_fp)

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
dict_keys(['loss', 'binary_accuracy', 'precision', 'recall', 'prec_thr', 'rec_thr', 'auc_pr', 'auc_roc', 'tp', 'fp', 'tn', 'fn', 'val_loss', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_prec_thr', 'val_rec_thr', 'val_auc_pr', 'val_auc_roc', 'val_tp', 'val_fp', 'val_tn', 'val_fn'])
"""


def plot_history_metrics(history, save_fp) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)

    train_loss = history["loss"]
    val_loss = history["val_loss"]
    train_auc_pr = history["auc_pr"]
    val_auc_pr = history["val_auc_pr"]

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.title("Training & Validation Loss")
    plt.title("Validation Loss")
    plt.yscale("log")
    plt.legend()

    # Plot PR Auc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc_pr, label="Training PR AUC")
    # plt.plot(epochs, val_auc_pr, label="Validation PR AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR AUC")
    plt.title("Training & Validation PR AUC")
    plt.legend()

    plt.savefig(save_fp, format="png")


history_np = np.load(
    "/Users/jochoa4/Desktop/pfe_transfers/train_keras_model_v2/res_train.npy",
    allow_pickle=True,
)

history = history_np.item()

save_dir = Path("/Users/jochoa4/Desktop/plots/train_keras_model_v2")
save_dir.mkdir(parents=True, exist_ok=True)

save_fp = save_dir / "loss_auc_pr_plot.png"

plot_history_metrics(history=history, save_fp=save_fp)

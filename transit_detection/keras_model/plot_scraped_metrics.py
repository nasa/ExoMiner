import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from transit_detection.keras_model.utils_scrape_metrics import (
    create_epoch_metrics_dict_from_log,
)

"""
epoch_dict: {
                1: {val_loss: XX.XX,
                    loss: XX.XX,
                    ...
                    }
                2: {val_loss: XX.XX,
                    loss: XX.XX,
                    ...
                    }
                .
                .
                .
                EPOCH_X: {val_loss: XX.XX,
                    loss: XX.XX,
                    ...
                    }
            }
"""


def plot_history_metrics(epoch_metrics_dict, save_fp) -> None:
    epochs = epoch_metrics_dict.keys()

    train_loss = []
    val_loss = []
    train_auc_pr = []
    val_auc_pr = []
    for epoch in epochs:
        train_loss.append(epoch_metrics_dict[epoch]["loss"])
        val_loss.append(epoch_metrics_dict[epoch]["val_loss"])
        train_auc_pr.append(epoch_metrics_dict[epoch]["auc_pr"])
        val_auc_pr.append(epoch_metrics_dict[epoch]["val_auc_pr"])

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
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


save_dir = Path("/Users/jochoa4/Desktop/plots/train_keras_model_v3")
save_dir.mkdir(parents=True, exist_ok=True)

save_fp = save_dir / "loss_auc_pr_plot.png"


epoch_log_fp = "/Users/jochoa4/Desktop/pfe_transfers/train_keras_model_v3.txt"
epoch_metrics_dict = create_epoch_metrics_dict_from_log(epoch_log_fp=epoch_log_fp)
print(epoch_metrics_dict)
plot_history_metrics(epoch_metrics_dict=epoch_metrics_dict, save_fp=save_fp)

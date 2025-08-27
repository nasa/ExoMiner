"""Scrape metrics from epoch log"""

import re


def create_epoch_metrics_dict_from_log(epoch_log_fp: str) -> dict:

    epoch_line_delim = (
        "us/step"  # Some unique delim to seperate epoch information by without overlap
    )

    with open(epoch_log_fp, "r") as file:
        content = file.read()

        epoch_info = content.split(epoch_line_delim)

        # Clean epoch 1 dump:
        epoch_info[0] = epoch_info[0].split("...")[-1]

        metrics = ["loss", "val_loss", "auc_pr", "val_auc_pr"]

        metrics_dict = {}

        for i, info in enumerate(epoch_info):
            try:
                # print(i)
                pattern = re.compile("Epoch (.*?):")
                match = re.search(pattern, info)
                epoch = int(match.group(1))  # Get epoch as int

                metrics_dict[epoch] = {}
                for metric in metrics:
                    pattern = re.compile(f"- {metric}: (.*?) - ")
                    match = re.search(pattern, info)
                    metric_val = float(match.group(1))
                    metrics_dict[epoch][metric] = metric_val
            except Exception as e:
                print(f"Error: {e} at iteration {i}")

        return metrics_dict

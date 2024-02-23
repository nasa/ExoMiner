"""
Plot loss and other metric curves that were computed during model fit using src_cv/cv.py script. Results file must have
training and validation set results.

Results file is a dictionary in which each key maps to a metric/loss evaluated in a given data set. Training and
validation results are available for each epoch.
Metric results for the training set do not have any prefix (e.g., 'loss').
Results for the validation set have prefix 'val_'
"""

# 3rd party
import numpy as np
from pathlib import Path

# local
from src.utils_train import plot_metric_from_res_file

# file pat to results numpy file
res_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_q1q17dr25_simdata/experiments/cv_exominer_obsplanets_inj1_2-22-2024_1312/cv_iter_0/models/model9/res_eval.npy')
# file path to save image to
save_fp = Path('/Users/msaragoc/Downloads/plot_prauc_curve.png')

# load results file
res = np.load(res_fp, allow_pickle=True).item()
print(f'Metrics/loss available: {res.keys()}')

metric_name_chosen = 'auc_pr'

# set epochs array
epochs_arr = np.arange(len(res[metric_name_chosen]))

# choose metrics to be plotted
chosen_metrics = [metric_name for metric_name in res if metric_name_chosen in metric_name]
chosen_res = {metric_name: {'epochs': epochs_arr, 'values': res[metric_name]} for metric_name in chosen_metrics}

plot_metric_from_res_file(chosen_res, save_fp)

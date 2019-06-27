"""
Utility functions for the TensorFlow estimator.
"""

# 3rd party
import os
import numpy as np


def early_stopping(best_value, curr_value, last_best, patience, minimize, min_delta=-np.inf):
    """ Early stopping.

    :param best_value: float, best value so far
    :param curr_value: float, current value
    :param last_best: int, best iteration with respect to the current iteration
    :param patience: int, number of epochs to wait before early stopping
    :param minimize: bool, if True, metric tracked is being minimized, False vice-versa
    :param min_delta: float, minimum change in the monitored quantity to qualify as an improvement
    :return:
        stop: bool, True if early stopping
        last_best: int, best iteration with respect to the current iteration
        best_value: float, best value so far
    """

    delta = curr_value - best_value

    # if current value is worse than best value so far
    if (minimize and delta >= 0) or (not minimize and delta <= 0):
        last_best += 1
    # if current value is better than best value so far
    # reset when best value was seen
    elif (minimize and delta < 0) or (not minimize and delta > 0) and np.abs(delta) > min_delta:
        last_best = 0
        best_value = curr_value

    # reached patience, stop training
    if last_best == patience:
        stop = True
    else:
        stop = False

    return stop, last_best, best_value


def delete_checkpoints(model_dir, keep_model_i):
    """ Delete checkpoints.

    :param model_dir: str, filepath to the directory in which the model checkpoints are saved
    :param keep_model_i: int, model instance to be kept based on epoch index
    :return:
    """

    ckpt_files = [ckpt for ckpt in os.listdir(model_dir) if 'model.ckpt' in ckpt]
    ckpt_files = sorted(ckpt_files, key=lambda file: file.split('.')[1].split('-')[1])

    ckpt_steps = np.unique([int(ckpt_file.split('.')[1].split('-')[1]) for ckpt_file in ckpt_files])

    for i in range(len(ckpt_files)):
        if str(ckpt_steps[keep_model_i - 1]) not in ckpt_files[i]:
            fp = os.path.join(model_dir, ckpt_files[i])
            if os.path.isfile(fp):  # delete file
                os.unlink(fp)

"""
Utility functions used during training for the TensorFlow custom estimator.
- Early stopping, data augmentation techniques, ...
"""

# 3rd party
import os
import numpy as np
import tensorflow as tf


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

    # get checkpoint files
    ckpt_files = [ckpt for ckpt in os.listdir(model_dir) if 'model.ckpt' in ckpt]
    # sort files based on steps
    ckpt_files = sorted(ckpt_files, key=lambda file: file.split('.')[1].split('-')[1])
    # get the steps id for each checkpoint
    ckpt_steps = np.unique([int(ckpt_file.split('.')[1].split('-')[1]) for ckpt_file in ckpt_files])
    # print('keeping checkpoint ({}) {}'.format(keep_model_i, ckpt_steps[keep_model_i - 1]))

    for i in range(len(ckpt_files)):
        if 'model.ckpt-' + str(ckpt_steps[keep_model_i - 1]) + '.' not in ckpt_files[i]:
            # print('deleting checkpoint file {}'.format(ckpt_files[i]))
            fp = os.path.join(model_dir, ckpt_files[i])
            if os.path.isfile(fp):  # delete file
                os.unlink(fp)

    # updating checkpoint txt file
    with open(model_dir + "/checkpoint", "w") as file:
        file.write('model_checkpoint_path: "model.ckpt-{}"\n'.format(int(ckpt_steps[keep_model_i - 1])))
        file.write('all_model_checkpoint_paths: "model.ckpt-{}"'.format(int(ckpt_steps[keep_model_i - 1])))


def phase_inversion(timeseries_tensor, should_reverse):
    """ Inverts phase of the time-series.

    :param timeseries_tensor: time-series tensor; shape (N,) in which N is the length of the time-series; dtype float32
    :param should_reverse:
    :return:
        original time-series with inverted phase
    """

    return tf.cond(should_reverse,
                   lambda: tf.reverse(timeseries_tensor, axis=[0]),
                   lambda: tf.identity(timeseries_tensor), name='inversion')


def add_whitegaussiannoise(timeseries_tensor, mean, rms_oot):
    """ Adds Gaussian noise with mean "mean" and standard deviation sample uniformly from [0, rms_oot].

    :param timeseries_tensor: time-series tensor; shape (N,) in which N is the length of the time-series; dtype float32
    :param mean: float, mean value for the Gaussian
    :param rms_oot: float, out-of-transit RMS of the time-series
    :return:
        original time-series with added Gaussian noise
    """

    return timeseries_tensor + tf.random.normal(timeseries_tensor.shape,
                                                mean,
                                                tf.random.uniform(shape=(),
                                                                  minval=0,
                                                                  maxval=rms_oot,
                                                                  dtype=tf.dtypes.float32),
                                                name='gaussiannoise')


def phase_shift(timeseries_tensor, bin_shift):
    """ Shifts the time-series by n bins with n being drawn uniformly from bin_shift. The time-series slides and the
    shifted end parts move from one end to the other.

    :param timeseries_tensor: time-series tensor; shape (N,) in which N is the length of the time-series; dtype float32
    :param bin_shift: list, minimum and maximum shift interval
    :return:
        original time-series phase-shifted
    """

    shift = tf.random.uniform(shape=(),
                      minval=bin_shift[0],
                      maxval=bin_shift[1],
                      dtype=tf.dtypes.int32, name='randuniform')

    if shift == 0:
        return timeseries_tensor
    elif shift > 0:
        return tf.concat([tf.slice(timeseries_tensor, (shift,), (timeseries_tensor.get_shape()[0] - shift,)),
                          tf.slice(timeseries_tensor, (0,), (shift,))],
                         axis=0, name='pos_shift')
    else:
        return tf.concat([tf.slice(timeseries_tensor, timeseries_tensor.get_shape() - tf.constant(shift), (shift,)),
                          tf.slice(timeseries_tensor, (0,), (timeseries_tensor.get_shape()[0] - shift,))],
                         axis=0, name='neg_shift')

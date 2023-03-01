"""
Implementation of real-time data augmentation methods.
"""

# 3rd party
import tensorflow as tf


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
    :param bin_shift: shift, int number of bins to shift the time-series
    :return:
        original time-series phase-shifted
    """

    if bin_shift == 0:
        return timeseries_tensor
    elif bin_shift > 0:
        return tf.concat([tf.slice(timeseries_tensor, (bin_shift, 0),
                                   (timeseries_tensor.get_shape()[0] - bin_shift, 1)),
                          tf.slice(timeseries_tensor, (0, 0), (bin_shift, 1))],
                         axis=0, name='pos_shift')
    else:
        bin_shift = tf.math.abs(bin_shift)
        return tf.concat([tf.slice(timeseries_tensor,
                                   (timeseries_tensor.get_shape()[0] - bin_shift, 0), (bin_shift, 1)),
                          tf.slice(timeseries_tensor, (0, 0), (timeseries_tensor.get_shape()[0] - bin_shift, 1))],
                         axis=0, name='neg_shift')
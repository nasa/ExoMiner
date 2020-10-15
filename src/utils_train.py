"""
Utility functions used during training, evaluating and predicting.
- Custom callbacks, data augmentation techniques, ...
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


class LayerOutputCallback(tf.keras.callbacks.Callback):

    def __init__(self, input_fn, batch_size, layer_name, summary_writer, buckets, description='custom'):
        """ Callback that writes to a histogram summary the output of a given layer.

        :param input_fn: input function
        :param batch_size: int, batch size
        :param layer_name: str, name of the layer
        :param summary_writer: summary writer
        :param buckets: int, bucket size
        :param description: str, optional description
        """

        super(LayerOutputCallback, self).__init__()
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.layer_name = layer_name
        self.summary_writer = summary_writer
        self.description = description
        self.buckets = buckets

    # def on_batch_end(self, batch, logs=None):
    #
    #     layer = [l for l in self.model.layers if l.name == 'convbranch_wscalar_concat'][0]
    #     get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #     print('Batch {} | Input to FC block: '.format(batch), get_layer_output.outputs)

    # def on_train_batch_end(self, batch, logs=None):
    #
    #     num_batches = int(24000 / self.batch_size)
    #     if batch in np.linspace(0, num_batches, 10, endpoint=False, dtype='uint32'):
    #         # pass
    #         layer = [l for l in self.model.layers if l.name == self.layer_name][0]
    #         get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #         for batch_i, (batch_input, batch_label) in enumerate(self.inputFn):
    #             if batch_i == batch:
    #                 batch_output_layer = get_layer_output(batch_input)
    #                 # print('Batch {} | Input to FC block: '.format(batch), batch_output_layer)
    #                 with self.summaryWriter.as_default():
    #                     tf.summary.histogram(data=tf.convert_to_tensor(batch_output_layer, dtype=tf.float32), name='{}_output'.format(self.layer_name), step=batch, buckets=30, description='aaa')

    def on_epoch_end(self, epoch, logs=None):
        """ Write to a summary the output of a given layer at the end of each epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        layer = [l for l in self.model.layers if l.name == self.layer_name][0]
        get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
        data = []
        for batch_i, (batch_input, batch_label) in enumerate(self.input_fn):
            if batch_i == 0:
                batch_output_layer = get_layer_output(batch_input)
                data.append(batch_output_layer)

        with self.summary_writer.as_default():

            tf.summary.histogram(data=tf.convert_to_tensor([data], dtype=tf.float32),
                                 name='{}_output'.format(self.layer_name), step=epoch, buckets=self.buckets,
                                 description=self.description)

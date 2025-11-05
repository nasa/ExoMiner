""" Custom TensorFlow Keras layers. """

# 3rd party
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np

@tf.keras.utils.register_keras_serializable()
class MeanAttentionNormalization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MeanAttentionNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute mean across axis 1
        mean_attn = tf.reduce_mean(inputs, axis=1)
        # Normalize across the last axis
        sum_attn = tf.reduce_sum(mean_attn, axis=-1, keepdims=True)
        return mean_attn / (sum_attn + tf.keras.backend.epsilon())

@tf.keras.utils.register_keras_serializable() 
class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1, **kwargs):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.k
        })
        return config

@tf.keras.utils.register_keras_serializable()
class StdLayer(keras.layers.Layer):
    """
    Creates TF Keras std layer. Can be called on a list of inputs with the same shape to compute standard deviation.
    """

    def __init__(self, axis=-1, **kwargs):
        super(StdLayer, self).__init__(**kwargs)
        self.axis = axis

    # @tf.function(jit_compile=True)
    def _std_fn(self, inputs):
        """ Computes standard deviation from inputs.

            Args:
                inputs: list of TF Keras tensors of same shape

            Returns: TF Keras tensor with same shape of `inputs` with std values
        """

        # mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        # variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)
        # std = tf.sqrt(variance)

        stacked = tf.stack(inputs, axis=self.axis)  # shape: [20, batch, 1, 25, 8]
        mean = tf.reduce_mean(stacked, axis=self.axis, keepdims=False)
        variance = tf.reduce_mean(tf.square(stacked - tf.expand_dims(mean, axis=self.axis)), axis=self.axis)
        std = tf.sqrt(variance)

        return std

    def call(self, inputs, training=None, mask=None):
        """ Call the custom standard deviation layer.

            Args:
                inputs: list of TF Keras tensors of same shape
                training: if True, behaves differently for training (not needed for this type of layer - added for
                    compatibility)
                mask: if not None, then specifies which inputs should be ignored (not needed for this type of layer -
                    added for compatibility)

            Returns: TF Keras tensor with same shape of `inputs` with std values
        """

        std = self._std_fn(inputs)

        return std

@tf.keras.utils.register_keras_serializable()
class SplitLayer(keras.layers.Layer):
    """ Creates a custom split layer. """

    def __init__(self, num_or_size_splits, axis=0, **kwargs):
        """ Constructor for the split layer.

        Args:
            num_or_size_splits: either an int indicating the number of splits along axis or a 1-D integer Tensor or
                Python list containing the sizes of each output tensor along axis. If an int, then it must evenly divide
                value.shape[axis]; otherwise the sum of sizes along the split axis must match that of the value.
            axis: An int or scalar int32 Tensor. The dimension along which to split. Must be in the range
                [-rank(value), rank(value)). Defaults to 0.
            **kwargs:
        """

        super(SplitLayer, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    # @tf.function(jit_compile=True)
    def _split_fn(self, inputs):
        """ Splits the inputs.

        Args:
            inputs: TF Keras layer, input to be split

        Returns: if num_or_size_splits is an int returns a list of num_or_size_splits Tensor objects; if
            num_or_size_splits is a 1-D list for 1-D Tensor returns num_or_size_splits.get_shape[0] Tensor objects
            resulting from splitting value.

        """
        return tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis)

    def call(self, inputs, training=None, mask=None):
        """ Calls the custom split layer.

        Args:
            inputs: TF Keras layer, input to be split
            training: if True, behaves differently for training (not needed for this type of layer - added for
                compatibility)
            mask: if not None, then specifies which inputs should be ignored (not needed for this type of layer -
                added for compatibility)

        Returns: if num_or_size_splits is an int returns a list of num_or_size_splits Tensor objects; if
            num_or_size_splits is a 1-D list for 1-D Tensor returns num_or_size_splits.get_shape[0] Tensor objects
            resulting from splitting value.
        """

        return self._split_fn(inputs)

@tf.keras.utils.register_keras_serializable()
class ReduceSumLayer(tf.keras.layers.Layer):
    """Creates TF Keras reduce sum layer."""

    def __init__(self, axis=-1, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, training=None, mask=None):
        return tf.reduce_sum(inputs, axis=1)


@tf.keras.utils.register_keras_serializable()
class IdentityConv1DInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        kernel_size, in_channels, out_channels = shape
        weights = tf.zeros(shape, dtype=dtype)

        # Place identity weights in the center of the kernel
        center = kernel_size // 2  # works for both even and odd sizes
        diag_size = tf.minimum(in_channels, out_channels)

        indices = tf.stack([
            tf.fill([diag_size], center),         # kernel position
            tf.range(diag_size, dtype=tf.int32),  # input channels
            tf.range(diag_size, dtype=tf.int32)   # output channels
        ], axis=1)

        updates = tf.ones([diag_size], dtype=dtype)
        weights = tf.tensor_scatter_nd_update(weights, indices, updates)

        return weights

    def get_config(self):
        return {}

@tf.keras.utils.register_keras_serializable()
class IdentityConv2DInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        kernel_height, kernel_width, in_channels, out_channels = shape
        weights = tf.zeros(shape, dtype=dtype)

        # Center of the kernel
        center_h = kernel_height // 2
        center_w = kernel_width // 2
        diag_size = tf.minimum(in_channels, out_channels)

        # Create indices for the identity mapping
        indices = tf.stack([
            tf.fill([diag_size], center_h),       # height position
            tf.fill([diag_size], center_w),       # width position
            tf.range(diag_size, dtype=tf.int32),  # input channels
            tf.range(diag_size, dtype=tf.int32)   # output channels
        ], axis=1)

        updates = tf.ones([diag_size], dtype=dtype)
        weights = tf.tensor_scatter_nd_update(weights, indices, updates)

        return weights

    def get_config(self):
        return {}



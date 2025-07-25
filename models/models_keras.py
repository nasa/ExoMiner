""" Implementation of models using Keras functional API. """

# 3rd party
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# local
from models.utils_models import create_inputs


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


class StdLayer(Layer):
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


class SplitLayer(Layer):
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



class ExoMinerMLP(object):

    def __init__(self, config, features):
        """ Initializes the ExoMiner architecture that processes only scalar features.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        if self.config['multi_class'] or \
                (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(np.unique(list(config['label_map'].values())))
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def build_scalar_inputs(self):
        """ Builds the scalar input features.

        :return:
            scalar_input, TF keras tensor with the different scalar input features
        """


        scalar_inputs = [self.inputs[feature_name] for feature_name in
                         self.config['scalar_branches']['scalar_branch']]
        if len(scalar_inputs) > 1:
            scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'scalar_inputs_concat')(scalar_inputs)
        else:
            scalar_inputs = scalar_inputs[0]

        return scalar_inputs


    def build_fc_block(self, scalar_inputs):
        """ Builds the FC block after the convolutional branches.

        :param scalar_inputs: tf keras tensor with scalar input features
        :return:
            net: tf keras model with added FC block
        """

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']) if self.config[
                                                                                  'decay_rate'] is not None else None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net if fc_layer_i > 0 else scalar_inputs)
            else:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net if fc_layer_i > 0 else scalar_inputs)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu{}'.format(fc_layer_i))(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate'], name=f'dropout_fc{fc_layer_i}')(net)

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {}

        scalar_inputs = self.build_scalar_inputs()

        net = self.build_fc_block(scalar_inputs)

        # create output layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output


class ExoMinerSmall(object):

    def __init__(self, config, features):
        """ Initializes the ExoMiner architecture that processes only scalar features.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        if self.config['multi_class'] or \
                (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(np.unique(list(config['label_map'].values())))
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def build_conv_branches(self):
        """ Builds convolutional branches.

        :return:
            conv_branches, dict with the different convolutional branches
        """

        conv_branch_selected = [
            # global branches
            'global_flux',
            # 'flux_trend',

            # local branches
            'local_flux',
            # 'local_centroid',
            # 'momentum_dump',

            # periodogram branch
            # 'flux_periodogram',
        ]

        config_mapper = {'blocks': {
            'global_flux': 'num_glob_conv_blocks',
            # 'flux_trend': 'num_glob_conv_blocks',
            'local_flux': 'num_loc_conv_blocks',
            # 'local_centroid': 'num_loc_conv_blocks',
            # 'momentum_dump': 'num_loc_conv_blocks',
            # 'flux_periodogram': 'num_pgram_conv_blocks',
        },
            'pool_size': {
                'global_flux': 'pool_size_glob',
                'local_flux': 'pool_size_loc',
                # 'flux_trend': 'pool_size_glob',
                # 'local_centroid': 'pool_size_loc',
                # 'momentum_dump': 'pool_size_loc',
                # 'flux_periodogram': 'pool_size_pgram',
            },
            'kernel_size': {
                'global_flux': 'kernel_size_glob',
                'local_flux': 'kernel_size_loc',
                # 'flux_trend': 'kernel_size_glob',
                # 'local_centroid': 'kernel_size_loc',
                # 'momentum_dump': 'kernel_size_loc',
                # 'flux_periodogram': 'kernel_size_pgram',
            },
            'conv_ls_per_block': {
                'global_flux': 'glob_conv_ls_per_block',
                'local_flux': 'loc_conv_ls_per_block',
                # 'flux_trend': 'glob_conv_ls_per_block',
                # 'local_centroid': 'loc_conv_ls_per_block',
                # 'momentum_dump': 'loc_conv_ls_per_block',
                # 'flux_periodogram': 'pgram_conv_ls_per_block',
            },
            'init_conv_filters': {
                'global_flux': 'init_glob_conv_filters',
                'local_flux': 'init_loc_conv_filters',
                # 'flux_trend': 'init_glob_conv_filters',
                # 'local_centroid': 'init_loc_conv_filters',
                # 'momentum_dump': 'init_loc_conv_filters',
                # 'flux_periodogram': 'init_pgram_conv_filters',
            }
        }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in conv_branch_selected
                         if branch_name in self.config['conv_branches']}
        if len(conv_branches) == 0:
            return {}

        for branch_i, branch in enumerate(conv_branches):  # create a convolutional branch

            branch_view_inputs = [self.inputs[view_name] for view_name in self.config['conv_branches'][branch]['views']]

            # add var time series
            if len(branch_view_inputs) > 1:
                branch_view_inputs = tf.keras.layers.Concatenate(axis=2, name=f'{branch}_input')(branch_view_inputs)

            # get init parameters for the given view
            n_blocks = self.config[config_mapper['blocks'][branch]]
            kernel_size = self.config[config_mapper['kernel_size'][branch]]
            pool_size = self.config[config_mapper['pool_size'][branch]]
            kernel_stride = self.config['kernel_stride']
            pool_stride = self.config['pool_stride']
            conv_ls_per_block = self.config[config_mapper['conv_ls_per_block'][branch]]
            init_conv_filters = self.config[config_mapper['init_conv_filters'][branch]]

            # create convolutional branches
            for conv_block_i in range(n_blocks):  # create convolutional blocks

                num_filters = 2 ** (init_conv_filters + conv_block_i)

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': kernel_size,
                               'strides': kernel_stride,
                               'padding': 'same'
                               }
                pool_kwargs = {
                    'pool_size': pool_size,
                    'strides': pool_stride
                }

                for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional block
                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None,
                                                 name='{}_conv_{}_{}'.format(branch, conv_block_i,
                                                                            seq_conv_block_i),
                                                 **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                      seq_conv_block_i == 0
                                                                else net)

                    # if seq_conv_block_i == conv_ls_per_block - 1:
                    #     net = tf.keras.layers.BatchNormalization(
                    #         axis=-1,
                    #         momentum=0.99,
                    #         epsilon=0.001,
                    #         center=True,
                    #         scale=True,
                    #         beta_initializer='zeros',
                    #         gamma_initializer='ones',
                    #         moving_mean_initializer='zeros',
                    #         moving_variance_initializer='ones',
                    #         beta_regularizer=None,
                    #         gamma_regularizer=None,
                    #         beta_constraint=None,
                    #         gamma_constraint=None,
                    #         synchronized=False,
                    #         name=f'{branch}_conv_{conv_block_i}_{seq_conv_block_i}_batch_norm'
                    #     )(net)

                    if self.config['non_lin_fn'] == 'lrelu':
                        net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                        name='{}_lrelu_{}_{}'.format(branch, conv_block_i,
                                                                                    seq_conv_block_i))(net)
                    elif self.config['non_lin_fn'] == 'relu':
                        net = tf.keras.layers.ReLU(name='{}_relu_{}_{}'.format(branch, conv_block_i,
                                                                              seq_conv_block_i))(net)
                    elif self.config['non_lin_fn'] == 'prelu':
                        net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                    alpha_regularizer=None,
                                                    alpha_constraint=None,
                                                    shared_axes=[1, 2],
                                                    name='{}_prelu_{}_{}'.format(branch, conv_block_i,
                                                                                seq_conv_block_i))(net)

                net = tf.keras.layers.MaxPooling1D(**pool_kwargs,
                                                   name='{}_maxpooling_{}'.format(branch, conv_block_i))(net)

            net = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch}_global_max_pooling')(net)
            # # flatten output of the convolutional branch
            # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            # net = tf.keras.layers.LayerNormalization(
            #     axis=-1,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     rms_scaling=False,
            #     beta_initializer='zeros',
            #     gamma_initializer='ones',
            #     beta_regularizer=None,
            #     gamma_regularizer=None,
            #     beta_constraint=None,
            #     gamma_constraint=None,
            #     name=f'{branch}_flux_layer_norm'
            # )(net)
            # net = tf.keras.layers.BatchNormalization(
            #     axis=-1,
            #     momentum=0.99,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     beta_initializer='zeros',
            #     gamma_initializer='ones',
            #     moving_mean_initializer='zeros',
            #     moving_variance_initializer='ones',
            #     beta_regularizer=None,
            #     gamma_regularizer=None,
            #     beta_constraint=None,
            #     gamma_constraint=None,
            #     synchronized=False,
            #     name=f'{branch}_flux_batch_norm'
            # )(net)

            # concatenate scalar features with features extracted in the convolutional branch for the time series views
            if self.config['conv_branches'][branch]['scalars'] is not None:
                scalar_inputs = [
                    self.inputs[feature_name] if feature_name != 'mag_cat' else tf.cast(self.inputs['mag_cat'],
                                                                                        tf.float32)
                    for feature_name in self.config['conv_branches'][branch]['scalars']]
                if len(scalar_inputs) > 1:
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'{branch}_scalar_input')(scalar_inputs)
                else:
                    scalar_inputs = scalar_inputs[0]

                net = tf.keras.layers.Concatenate(axis=1, name='{}_flatten_wscalar'.format(branch))([
                    net,
                    scalar_inputs
                ])

            # add FC layer that extracts features from the combined feature vector of features from the convolutional
            # branch (flattened) and corresponding scalar features
            if self.config['num_fc_conv_units'] > 0:
                net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='{}_fc'.format(branch))(net)

                # net = tf.keras.layers.BatchNormalization(
                #     axis=-1,
                #     momentum=0.99,
                #     epsilon=0.001,
                #     center=True,
                #     scale=True,
                #     beta_initializer='zeros',
                #     gamma_initializer='ones',
                #     moving_mean_initializer='zeros',
                #     moving_variance_initializer='ones',
                #     beta_regularizer=None,
                #     gamma_regularizer=None,
                #     beta_constraint=None,
                #     gamma_constraint=None,
                #     synchronized=False,
                #     name=f'{branch}_fc_batch_norm'
                # )(net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01, name='{}_fc_lrelu'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='{}_fc_relu'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1],
                                                name='{}_fc_prelu'.format(branch))(net)

                net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'],
                                              name=f'{branch}_dropout_fc_conv')(net)

            conv_branches[branch] = net

        return conv_branches

    def build_scalar_branches(self):
        """ Builds the scalar branches.

        :return:
            scalar_branches_net, dict with the different scalar branches
        """

        scalar_branches_net = {scalar_branch_name: None for scalar_branch_name in self.config['scalar_branches']}
        for scalar_branch_name in self.config['scalar_branches']:

            scalar_inputs = [self.inputs[feature_name] for feature_name in
                             self.config['scalar_branches'][scalar_branch_name]]
            if len(scalar_inputs) > 1:
                scalar_input = tf.keras.layers.Concatenate(axis=1, name=f'{scalar_branch_name}_scalar_input')(
                    scalar_inputs)
            else:
                scalar_input = scalar_inputs[0]

            scalar_fc_output = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                                     kernel_regularizer=regularizers.l2(self.config['decay_rate']) if
                                                     self.config['decay_rate'] is not None else None,
                                                     activation=None,
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     bias_initializer='zeros',
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name=f'fc_{scalar_branch_name}_scalar')(scalar_input)

            # scalar_fc_output = tf.keras.layers.BatchNormalization(
            #     axis=-1,
            #     momentum=0.99,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     beta_initializer='zeros',
            #     gamma_initializer='ones',
            #     moving_mean_initializer='zeros',
            #     moving_variance_initializer='ones',
            #     beta_regularizer=None,
            #     gamma_regularizer=None,
            #     beta_constraint=None,
            #     gamma_constraint=None,
            #     synchronized=False,
            #     name=f'fc_{scalar_branch_name}_scalar_batch_norm'
            # )(scalar_fc_output)

            if self.config['non_lin_fn'] == 'lrelu':
                scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name=f'fc_lrelu_{scalar_branch_name}_scalar')(
                    scalar_fc_output)
            elif self.config['non_lin_fn'] == 'relu':
                scalar_fc_output = tf.keras.layers.ReLU(name=f'fc_relu_{scalar_branch_name}_scalar')(scalar_fc_output)
            elif self.config['non_lin_fn'] == 'prelu':
                scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                         alpha_regularizer=None,
                                                         alpha_constraint=None,
                                                         shared_axes=[1],
                                                         name=f'fc_prelu_{scalar_branch_name}_scalar')(scalar_fc_output)
            scalar_branches_net[scalar_branch_name] = scalar_fc_output  # scalar_input  # scalar_fc_output

        return scalar_branches_net

    def connect_segments(self, branches):
        """ Connect the different branches.

        :param branches: dict, branches to be concatenated
        :return:
            model output before FC layers
        """

        branches_to_concatenate = []
        for branch_name, branch in branches.items():
            branches_to_concatenate.append(branch)

        if len(branches_to_concatenate) > 1:
            net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(branches_to_concatenate)
        else:
            net = branches_to_concatenate[0]

        return net

    def build_fc_block(self, scalar_inputs):
        """ Builds the FC block after the convolutional branches.

        :param scalar_inputs: tf keras tensor with scalar input features
        :return:
            net: tf keras model with added FC block
        """

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']) if self.config[
                                                                                  'decay_rate'] is not None else None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net if fc_layer_i > 0 else scalar_inputs)
            else:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net if fc_layer_i > 0 else scalar_inputs)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu{}'.format(fc_layer_i))(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate'], name=f'dropout_fc{fc_layer_i}')(net)

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {}

        branches_net.update(self.build_scalar_branches())

        branches_net.update(self.build_conv_branches())

        # merge branches
        net = self.connect_segments(branches_net)

        # create FC layers
        net = self.build_fc_block(net)

        # create output layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output


class ExoMinerDiffImg(object):

    def __init__(self, config, features):
        """ Initializes the ExoMiner architecture that processes local flux through the same convolutional branch
        before extracting features specifically to each local flux-related diagnostic. The core architecture consists of
        one convolutional branch per test diagnostic, except for the ones related to the local flux views (i.e., local
        flux, secondary, and odd-even). Those are processed through the same convolutional branch. The extracted
        features from each convolutional branch are flattened and merged, and then fed into a final FC block for
        classification. The 'unfolded local flux' branch consists of a set of phases that are processed together in
        this specific branch.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.task = config['task']
        self.features = features

        if self.config['multi_class'] or \
                (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(np.unique(list(config['label_map'].values())))
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)


    def build_diff_img_branch(self):
        """ Builds the difference image branch.

        :return: dict with the difference image branch
        """

        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        branch_view_inputs = [self.inputs[view_name] for view_name in self.config['diff_img_branch']['imgs']]
        branch_view_inputs = [tf.keras.layers.Reshape(l.shape[1:] + (1,), name=f'diff_imgs_expanding_{l.name}_dims')(l)
                              for l in branch_view_inputs]

        branch_view_inputs = tf.keras.layers.Concatenate(axis=4, name='input_diff_img_concat')(branch_view_inputs)

        # get number of conv blocks, layers per block, and kernel and pool sizes for the branch
        n_blocks = self.config['num_diffimg_conv_blocks']
        kernel_size = (1, self.config['kernel_size_diffimg'], self.config['kernel_size_diffimg'])
        pool_size = (1, self.config['pool_size_diffimg'], self.config['pool_size_diffimg'])
        kernel_stride = (1, self.config['kernel_stride'], self.config['kernel_stride'])
        pool_stride = (1, self.config['pool_stride'], self.config['pool_stride'])
        n_layers_per_block = self.config['diffimg_conv_ls_per_block']

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (self.config['init_diffimg_conv_filters'] + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': kernel_size,
                           'strides': kernel_stride,
                           'padding': 'same'
                           }
            pool_kwargs = {
                'pool_size': pool_size,
                'strides': pool_stride
            }

            for seq_conv_block_i in range(n_layers_per_block):  # create convolutional block

                net = tf.keras.layers.Conv3D(dilation_rate=1,
                                             activation=None,
                                             use_bias=True,
                                             bias_initializer='zeros',
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None,
                                             name='diff_imgs_conv{}_{}'.format(conv_block_i, seq_conv_block_i),
                                             **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                  seq_conv_block_i == 0
                                                            else net)

                # if seq_conv_block_i == n_layers_per_block - 1:
                #     net = tf.keras.layers.BatchNormalization(
                #         axis=-1,
                #         momentum=0.99,
                #         epsilon=0.001,
                #         center=True,
                #         scale=True,
                #         beta_initializer='zeros',
                #         gamma_initializer='ones',
                #         moving_mean_initializer='zeros',
                #         moving_variance_initializer='ones',
                #         beta_regularizer=None,
                #         gamma_regularizer=None,
                #         beta_constraint=None,
                #         gamma_constraint=None,
                #         synchronized=False,
                #         name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_batch_norm'
                #     )(net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='diff_imgs_lrelu_{}_{}'.format(conv_block_i,
                                                                                        seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='diff_imgs_relu_{}_{}'.format(conv_block_i,
                                                                                  seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1, 2],
                                                name='diff_imgs_prelu_{}_{}'.format(conv_block_i,
                                                                                    seq_conv_block_i))(net)

            if conv_block_i != n_blocks - 1:  # do not add maxpooling layer before global maxpooling layer
                net = tf.keras.layers.MaxPooling3D(**pool_kwargs,
                                                   name='diff_imgs_maxpooling_{}_{}'.format(conv_block_i,
                                                                                            seq_conv_block_i))(net)

        # split extracted features for each sector/quarter
        diff_imgs_split = SplitLayer(net.shape[1], 1, name='diff_imgs_split_extracted_features')(net)
        diff_imgs_global_max_res = []
        for img_i, extracted_img in enumerate(diff_imgs_split):
            # remove sector/quarter dimension
            extracted_img = tf.keras.layers.Reshape(extracted_img.shape[2:])(extracted_img)
            # compute pooling
            global_max_pooling_img = tf.keras.layers.GlobalAveragePooling2D(name=f'diff_imgs_global_max_pooling_{img_i}')(extracted_img)
            # add channel dimension need for concatenation after
            global_max_pooling_img = tf.keras.layers.Reshape(
                (1,) + global_max_pooling_img.shape[1:],
                name=f'diff_imgs_global_max_pooling_expand_dim_{img_i}')(global_max_pooling_img)
            # add result to list of pooling for current sector/quarter
            diff_imgs_global_max_res.append(
                global_max_pooling_img
            )

        # concatenate global max pooling features for all sectors/quarters
        net = tf.keras.layers.Concatenate(axis=1, name=f'diff_imgs_global_max_pooling_concat')(diff_imgs_global_max_res)

        # net = tf.keras.layers.LayerNormalization(
        #     axis=-1,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     rms_scaling=False,
        #     beta_initializer='zeros',
        #     gamma_initializer='ones',
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     name=f'diff_imgs_layer_norm'
        # )(net)
        # net = tf.keras.layers.BatchNormalization(
        #     axis=-1,
        #     momentum=0.99,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer='zeros',
        #     gamma_initializer='ones',
        #     moving_mean_initializer='zeros',
        #     moving_variance_initializer='ones',
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     synchronized=False,
        #     name=f'diff_imgs_batch_norm'
        # )(net)

        # # flatten output of the convolutional branch
        # net = tf.keras.layers.Reshape((net.shape[1], np.prod(net.shape[2:])), name='diff_imgs_flatten')(net)

        # add per-image scalar features
        if self.config['diff_img_branch']['imgs_scalars'] is not None:

            scalar_inputs = [self.inputs[feature_name]
                             if 'pixel' not in feature_name else self.inputs[feature_name]
                             for feature_name in self.config['diff_img_branch']['imgs_scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=0, name=f'diff_imgs_imgs_scalars_inputs_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            net = tf.keras.layers.Concatenate(axis=2, name='diff_imgs_imgsscalars_concat')([net, scalar_inputs])

        net = tf.keras.layers.Reshape(net.shape[1:] + (1, ), name=f'diff_imgs_expanding_w_imgs_scalars')(net)

        # compress features from image and scalar sector data into a set of features
        net = tf.keras.layers.Conv2D(filters=self.config['num_fc_diff_units'],
                                     kernel_size=(1, net.shape[2]),
                                     strides=(1, 1),
                                     padding='valid',
                                     kernel_initializer=weight_initializer,
                                     dilation_rate=1,
                                     activation=None,
                                     use_bias=True,
                                     bias_initializer='zeros',
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     kernel_constraint=None,
                                     bias_constraint=None,
                                     name='diff_imgs_convfc'.format('diff_img'),
                                     )(net)

        # net = tf.keras.layers.BatchNormalization(
        #     axis=-1,
        #     momentum=0.99,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer='zeros',
        #     gamma_initializer='ones',
        #     moving_mean_initializer='zeros',
        #     moving_variance_initializer='ones',
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     synchronized=False,
        #     name=f'diff_imgs_convfc_batch_norm'
        # )(net)

        if self.config['non_lin_fn'] == 'lrelu':
            net = tf.keras.layers.LeakyReLU(alpha=0.01, name='diff_imgs_convfc_lrelu')(net)
        elif self.config['non_lin_fn'] == 'relu':
            net = tf.keras.layers.ReLU(name='diff_imgs_convfc_relu')(net)
        elif self.config['non_lin_fn'] == 'prelu':
            net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                        alpha_regularizer=None,
                                        alpha_constraint=None,
                                        shared_axes=[1],
                                        name='diff_imgs_convfc_prelu')(net)

        net = tf.keras.layers.Flatten(data_format='channels_last', name='diff_imgs_flatten_convfc')(net)

        # add scalar features
        if self.config['diff_img_branch']['scalars'] is not None:

            scalar_inputs = [self.inputs[feature_name] for feature_name in self.config['diff_img_branch']['scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'diff_imgs_scalars_inputs_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            # concatenate scalar features with remaining features
            net = tf.keras.layers.Concatenate(axis=1, name='diff_imgs_flatten_w_scalar_inputs_concat')([net, scalar_inputs])

        # add FC layer that extracts features from the combined feature vector of features from the convolutional
        # branch (flattened) and corresponding scalar features
        if self.config['num_fc_conv_units'] > 0:
            net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                        kernel_regularizer=None,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        name='diff_imgs_fc')(net)

            # net = tf.keras.layers.BatchNormalization(
            #     axis=-1,
            #     momentum=0.99,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     beta_initializer='zeros',
            #     gamma_initializer='ones',
            #     moving_mean_initializer='zeros',
            #     moving_variance_initializer='ones',
            #     beta_regularizer=None,
            #     gamma_regularizer=None,
            #     beta_constraint=None,
            #     gamma_constraint=None,
            #     synchronized=False,
            #     name=f'diff_imgs_fc_batch_norm'
            # )(net)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='diff_imgs_fc_lrelu')(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='diff_imgs_fc_relu')(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='diff_imgs_fc_prelu')(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'], name=f'diff_imgs_dropout_fc')(net)

        return net


    def build_fc_block(self, net):
        """ Builds the FC block after the convolutional branches.

        :param net: model upstream the FC block
        :return:
            net: model with added FC block
        """

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']) if self.config[
                                                                                  'decay_rate'] is not None else None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net)
            else:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net)

            # if fc_layer_i != self.config['num_fc_layers'] - 1:
            #     net = tf.keras.layers.BatchNormalization(
            #         axis=-1,
            #         momentum=0.99,
            #         epsilon=0.001,
            #         center=True,
            #         scale=True,
            #         beta_initializer='zeros',
            #         gamma_initializer='ones',
            #         moving_mean_initializer='zeros',
            #         moving_variance_initializer='ones',
            #         beta_regularizer=None,
            #         gamma_regularizer=None,
            #         beta_constraint=None,
            #         gamma_constraint=None,
            #         synchronized=False,
            #         name=f'fc{fc_layer_i}_batch_norm'
            #     )(net)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu{}'.format(fc_layer_i))(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate'], name=f'dropout_fc{fc_layer_i}')(net)

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        diff_img_convnet = self.build_diff_img_branch()

        # create FC layers
        net = self.build_fc_block(diff_img_convnet)

        # create output layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.task == 'classification':
            if self.output_size == 1:
                output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
            else:
                output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)
        elif self.task == 'regression':
            output = tf.keras.layers.Activation('linear', name='linear')(logits)
        else:
            raise ValueError(f'Task type not implemented: {self.task} | only `classification` or '
                             f'`regression` are valid at this point.')

        return output


def process_extracted_conv_features_odd_even_flux(odd_even_flux_extracted_features):
    """ Perform odd-even-specific processing of extracted features from convolutional branch.

        Args:
            :param odd_even_flux_extracted_features: TF Keras Tensor, extracted features for odd and even flux from
                joint convolutional branch

        Returns: TF Keras tensor with processed odd-even features
     """

    odd_even_flux_extracted_features_split = SplitLayer(2,
                                                        axis=1,
                                                        name='local_fluxes_split_oe')(odd_even_flux_extracted_features)
    odd_even_flux_extracted_features_subtract = tf.keras.layers.Subtract(name='local_fluxes_subtract_oe')(
        odd_even_flux_extracted_features_split)

    return odd_even_flux_extracted_features_subtract


def process_extracted_conv_features_unfolded_flux(unfolded_flux_extracted_features, num_filters_conv_stats,
                                                  kernel_size_conv_stats, weight_initializer):
    """ Prepares local flux inputs to be processed by the same convolutional branch.

        Args:
            :param unfolded_flux_extracted_features: TF Keras tensor, extracted features for local unfolded flux
                from joint convolutional branch
            :param num_filters_conv_stats: int, number of convolutional features used to process features
            :param kernel_size_conv_stats: int, kernel size for convolutional layer
            :param weight_initializer: str, weight initializer for convolutional layer

        Returns: TF Keras tensor with concatenated local flux input features
     """

    # split layer in preparation to avg/min/max
    unfolded_flux_extracted_features_split = SplitLayer(unfolded_flux_extracted_features.shape[1],
                     axis=1,
                     name='unfolded_flux_split')(unfolded_flux_extracted_features)

    # get stats for all extracted features of each flux phase
    merge_layer_avg = tf.keras.layers.Average(name='unfolded_flux_avg')(unfolded_flux_extracted_features_split)
    merge_layer_min = tf.keras.layers.Minimum(name='unfolded_flux_min')(unfolded_flux_extracted_features_split)
    merge_layer_max = tf.keras.layers.Maximum(name='unfolded_flux_max')(unfolded_flux_extracted_features_split)
    # merge_layer_std = StdLayer(axis=1, name='unfolded_flux_std')(net)

    # concat 3 different layers
    unfolded_flux_extracted_features_merge_stats = (
        tf.keras.layers.Concatenate(axis=1, name='unfolded_flux_concat_stats')([
        merge_layer_min,
        merge_layer_max,
        merge_layer_avg,
        # merge_layer_std
    ]))

    # set the 3 layers to be the channels
    input_conv = tf.keras.layers.Permute((2, 3, 1),
                                  name='unfolded_flux_permute_stats')(unfolded_flux_extracted_features_merge_stats)

    # convolve features of the different channels
    conv_kwargs = {'filters': num_filters_conv_stats,
                   'kernel_initializer': weight_initializer,
                   'kernel_size': kernel_size_conv_stats,
                   }
    output_conv = tf.keras.layers.Conv2D(dilation_rate=1,
                                 activation=None,
                                 use_bias=True,
                                 bias_initializer='zeros',
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 kernel_constraint=None,
                                 bias_constraint=None,
                                 name='unfolded_flux_2conv',
                                 **conv_kwargs)(input_conv)

    return output_conv


class ExoMinerPlusPlus(object):
    """ ExoMiner architecture used in the TESS paper."""

    def __init__(self, config, features):
        """ Initializes the ExoMiner architecture that processes local flux through the same convolutional branch
        before extracting features specifically to each local flux-related diagnostic. The core architecture consists of
        one convolutional branch per test diagnostic, except for the ones related to the local flux views (i.e., local
        flux, secondary, and odd-even). Those are processed through the same convolutional branch. The extracted
        features from each convolutional branch are flattened and merged, and then fed into a final FC block for
        classification. The 'unfolded local flux' branch consists of a set of phases that are processed together in
        this specific branch.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        if self.config['multi_class'] or \
                (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(np.unique(list(config['label_map'].values())))
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)


    def build_conv_unfolded_flux(self):
        """ Creates a separate branch to process the unfolded flux feature. Max, min, and average features are extracted
        from the set of phases provided as input to the branch. These feature maps are then convolved with a final 1D
        convolutional layer to create the final features from this branch that are then concatenated with other scalar
        features.

        Returns: dict, unfolded convolutional branch

        """

        # hard-coded hyperparameter for convolution of extracted statistics
        kernel_size_conv_stats = 1
        num_filters_conv_stats = 4

        config_mapper = {'blocks': 'num_unfolded_conv_blocks',
                         'pool_size': 'pool_size_unfolded',
                         'kernel_size': 'kernel_size_unfolded',
                         'init_conv_filters': 'init_unfolded_conv_filters',
                         'conv_ls_per_block': 'unfolded_conv_ls_per_block'
                         }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        # initialize branch
        view_inputs = self.inputs["unfolded_local_flux_view_fluxnorm"]

        # expand dims to prevent last dimension being "lost"
        view_inputs = tf.keras.layers.Reshape(view_inputs.shape[1:] + (1,),
                                              name='expanding_unfolded_flux_dim')(view_inputs)

        branch = "local_unfolded_flux"
        unfolded_conv_branches = {branch: None}

        # get init parameters for given view
        n_blocks = self.config[config_mapper['blocks']]
        kernel_size = self.config[config_mapper['kernel_size']]
        pool_size = self.config[config_mapper['pool_size']]
        init_conv_filters = self.config[config_mapper['init_conv_filters']]
        conv_ls_per_block = self.config[config_mapper['conv_ls_per_block']]

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (init_conv_filters + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': (kernel_size, 1),
                           'strides': self.config['kernel_stride'],
                           'padding': 'same'
                           }

            for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional layers for the block

                net = tf.keras.layers.Conv2D(dilation_rate=1,
                                             activation=None,
                                             use_bias=True,
                                             bias_initializer='zeros',
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None,
                                             name='unfolded_flux_conv{}_{}_{}'.format(branch, conv_block_i,
                                                                                      seq_conv_block_i),
                                             **conv_kwargs)(view_inputs if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0 else net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='unfolded_flux_lrelu{}_{}_{}'.format(branch,
                                                                                              conv_block_i,
                                                                                seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='unfolded_flux_relu{}_{}_{}'.format(branch, conv_block_i,
                                                                          seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1, 2],
                                                name='unfolded_flux_prelu{}_{}_{}'.format(branch, conv_block_i,
                                                                            seq_conv_block_i))(net)

            net = tf.keras.layers.MaxPooling2D(pool_size=(1, pool_size),
                                               strides=(1, self.config['pool_stride']),
                                               name='unfolded_flux_maxpooling_{}_{}'.format(branch,
                                                                                            conv_block_i))(net)

        # split layer in preparation to avg/min/max
        net = SplitLayer(net.shape[1], axis=1, name='unfolded_flux_split_input')(net)

        # get avg, min, max of all layers
        merge_layer_avg = tf.keras.layers.Average(name='unfolded_flux_avg')(net)
        merge_layer_min = tf.keras.layers.Minimum(name='unfolded_flux_min')(net)
        merge_layer_max = tf.keras.layers.Maximum(name='unfolded_flux_max')(net)

        # concat 3 different layers
        net = tf.keras.layers.Concatenate(axis=1, name='unfolded_flux_merge')([
            merge_layer_min, merge_layer_max, merge_layer_avg])

        # set the 3 layers to be the channels
        net = tf.keras.layers.Permute((2, 3, 1), name='unfolded_flux_permute_merge')(net)

        # convolve output with conv1d to produce final output
        conv_kwargs = {'filters': num_filters_conv_stats,
                       'kernel_initializer': weight_initializer,
                       'kernel_size': (kernel_size_conv_stats, 1),
                       }

        net = tf.keras.layers.Conv2D(dilation_rate=1,
                                     activation=None,
                                     use_bias=True,
                                     bias_initializer='zeros',
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     kernel_constraint=None,
                                     bias_constraint=None,
                                     name='unfolded_flux_2conv{}_{}'.format(branch, 1),
                                     **conv_kwargs)(net)

        # flatten output of the convolutional branch
        net = tf.keras.layers.Flatten(data_format='channels_last', name='unfolded_flux_flatten_{}'.format(branch))(net)

        # concatenate scalar features with features extracted in the convolutional branch for the time series views
        if self.config['conv_branches'][branch]['scalars'] is not None:
            scalar_inputs = [self.inputs[feature_name]
                             for feature_name in self.config['conv_branches'][branch]['scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'{branch}_scalar_input')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                net,
                scalar_inputs
            ])

        if self.config['num_fc_conv_units'] > 0:
            net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                        kernel_regularizer=None,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        name='fc_{}'.format(branch))(net)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu_{}'.format(branch))(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'], name=f'dropout_fc_conv_{branch}')(net)

        unfolded_conv_branches[branch] = net

        return unfolded_conv_branches

    def build_conv_branches(self):
        """ Builds convolutional branches.

        :return:
            conv_branches, dict with the different convolutional branches
        """

        conv_branch_selected = [
            'global_flux',
            'local_centroid',
            'momentum_dump',
            'flux_trend',
            'flux_periodogram',
        ]
        odd_even_branch_name = 'local_odd_even'  # specific to odd and even branch
        config_mapper = {'blocks': {
            'global_view': 'num_glob_conv_blocks',
            'local_view': 'num_loc_conv_blocks',
            'centr_view': 'num_centr_conv_blocks'
        },
            'pool_size': {
                'global_view': 'pool_size_glob',
                'local_view': 'pool_size_loc',
                'centr_view': 'pool_size_centr'
            },
            'kernel_size': {
                'global_view': 'kernel_size_glob',
                'local_view': 'kernel_size_loc',
                'centr_view': 'kernel_size_centr'
            },
            'conv_ls_per_block': {
                'global_view': 'glob_conv_ls_per_block',
                'local_view': 'loc_conv_ls_per_block',
                'centr_view': 'centr_conv_ls_per_block'
            },
            'init_conv_filters': {
                'global_view': 'init_glob_conv_filters',
                'local_view': 'init_loc_conv_filters',
                'centr_view': 'init_centr_conv_filters'
            }
        }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in conv_branch_selected
                         if branch_name in self.config['conv_branches']}
        if len(conv_branches) == 0:
            return {}

        for branch_i, branch in enumerate(conv_branches):  # create a convolutional branch

            branch_view_inputs = [self.inputs[view_name] for view_name in self.config['conv_branches'][branch]['views']]

            # add var time series
            if len(branch_view_inputs) > 1:
                branch_view_inputs = tf.keras.layers.Concatenate(axis=2, name=f'input_{branch}')(branch_view_inputs)
            else:
                branch_view_inputs = branch_view_inputs[0]

            # get init parameters for the given view
            n_blocks = self.config[config_mapper['blocks'][('centr_view', 'global_view')['global' in branch]]]
            kernel_size = self.config[config_mapper['kernel_size'][('centr_view', 'global_view')['global' in branch]]]
            pool_size = self.config[config_mapper['pool_size'][('centr_view', 'global_view')['global' in branch]]]
            conv_ls_per_block = self.config[
                config_mapper['conv_ls_per_block'][('centr_view', 'global_view')['global' in branch]]]
            init_conv_filters = self.config[
                config_mapper['init_conv_filters'][('centr_view', 'global_view')['global' in branch]]]

            # create convolutional branches
            for conv_block_i in range(n_blocks):  # create convolutional blocks

                num_filters = 2 ** (init_conv_filters + conv_block_i)

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': (1, kernel_size)
                               if branch == odd_even_branch_name else kernel_size,
                               'strides': (1, self.config['kernel_stride'])
                               if branch == odd_even_branch_name else self.config['kernel_stride'],
                               'padding': 'same'
                               }

                for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional block

                    if branch == odd_even_branch_name:
                        net = tf.keras.layers.Conv2D(dilation_rate=1,
                                                     activation=None,
                                                     use_bias=True,
                                                     bias_initializer='zeros',
                                                     kernel_regularizer=None,
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i,
                                                                                seq_conv_block_i),
                                                     **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                          seq_conv_block_i == 0
                                                                    else net)
                    else:
                        net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                     activation=None,
                                                     use_bias=True,
                                                     bias_initializer='zeros',
                                                     kernel_regularizer=None,
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i,
                                                                                seq_conv_block_i),
                                                     **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                          seq_conv_block_i == 0
                                                                    else net)

                    if self.config['non_lin_fn'] == 'lrelu':
                        net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                        name='lrelu{}_{}_{}'.format(branch, conv_block_i,
                                                                                    seq_conv_block_i))(net)
                    elif self.config['non_lin_fn'] == 'relu':
                        net = tf.keras.layers.ReLU(name='relu{}_{}_{}'.format(branch, conv_block_i,
                                                                              seq_conv_block_i))(net)
                    elif self.config['non_lin_fn'] == 'prelu':
                        net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                    alpha_regularizer=None,
                                                    alpha_constraint=None,
                                                    shared_axes=[1, 2],
                                                    name='prelu{}_{}_{}'.format(branch, conv_block_i,
                                                                                seq_conv_block_i))(net)

                if branch == odd_even_branch_name:
                    net = tf.keras.layers.MaxPooling2D(pool_size=(1, pool_size),
                                                       strides=(1, self.config['pool_stride']),
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)
                else:
                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size,
                                                       strides=self.config['pool_stride'],
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)

            if branch == odd_even_branch_name:  # subtract extracted features for odd and even views branch
                net = SplitLayer(2, axis=1, name='split_oe')(net)
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)
                # net = tf.keras.layers.Permute((2, 3, 1), name='permute2_oe')(net)  # needed for Conv2D

            # flatten output of the convolutional branch
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            # concatenate scalar features with features extracted in the convolutional branch for the time series views
            if self.config['conv_branches'][branch]['scalars'] is not None:
                scalar_inputs = [
                    self.inputs[feature_name] if feature_name != 'mag_cat' else tf.cast(self.inputs['mag_cat'],
                                                                                        tf.float32)
                    for feature_name in self.config['conv_branches'][branch]['scalars']]
                if len(scalar_inputs) > 1:
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'{branch}_scalar_input')(scalar_inputs)
                else:
                    scalar_inputs = scalar_inputs[0]

                net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_inputs
                ])

            # add FC layer that extracts features from the combined feature vector of features from the convolutional
            # branch (flattened) and corresponding scalar features
            if self.config['num_fc_conv_units'] > 0:
                net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc_{}'.format(branch))(net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1],
                                                name='fc_prelu_{}'.format(branch))(net)
                # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten2_{}'.format(branch))(net)

                net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'],
                                              name=f'dropout_fc_conv_{branch}')(net)

            conv_branches[branch] = net

        return conv_branches

    def build_joint_local_conv_branches(self):
        """ Builds convolutional branch that processes local views (i.e., flux, secondary, and odd and even) jointly.
        It includes the option of adding the variability views as an extra channel (dim = [view, bins, avg/var]). The
        extracted feature maps are then split, flattened and concatenated with the respective scalar features before
        feeding into a small FC layer, that makes the output size for all branches the same.

        :return:
            conv_branches, dict with the different convolutional branches for the local views
        """

        # list of branches to be combined + their sizes
        local_transit_branches = ['local_odd_even', 'local_flux', 'local_weak_secondary']
        local_transit_sz = {'local_odd_even': 2, 'local_flux': 1, 'local_weak_secondary': 1}

        odd_even_branch_name = 'local_odd_even'  # specific to odd and even branch
        config_mapper = {'blocks': {
            'global_view': 'num_glob_conv_blocks',
            'local_view': 'num_loc_conv_blocks',
            'centr_view': 'num_centr_conv_blocks'
        },
            'pool_size': {
                'global_view': 'pool_size_glob',
                'local_view': 'pool_size_loc',
                'centr_view': 'pool_size_centr'
            },
            'kernel_size': {
                'global_view': 'kernel_size_glob',
                'local_view': 'kernel_size_loc',
                'centr_view': 'kernel_size_centr'
            },
            'conv_ls_per_block': {
                'global_view': 'glob_conv_ls_per_block',
                'local_view': 'loc_conv_ls_per_block',
                'centr_view': 'centr_conv_ls_per_block'
            },
            'init_conv_filters': {
                'global_view': 'init_glob_conv_filters',
                'local_view': 'init_loc_conv_filters',
                'centr_view': 'init_centr_conv_filters'
            }
        }

        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in local_transit_branches
                         if branch_name in self.config['conv_branches']}
        if len(conv_branches) == 0:
            return {}

        # aggregate local views input features
        local_transit_features = []
        for transit_branch in conv_branches:
            if transit_branch == 'local_odd_even':
                odd_transit_features = [view for view in self.config['conv_branches'][transit_branch]['views'] if
                                        'odd' in view]
                local_transit_features.append(odd_transit_features)
                even_transit_features = [view for view in self.config['conv_branches'][transit_branch]['views'] if
                                         'even' in view]
                local_transit_features.append(even_transit_features)
            else:
                local_transit_features.append(self.config['conv_branches'][transit_branch]['views'])

        local_transit_views_inputs = []
        for local_transit_feature in local_transit_features:
            view_inputs = [
                tf.keras.layers.Reshape((1,) + self.inputs[view_name].shape[1:],
                                        name=f'expanding_{view_name}_dim')(self.inputs[view_name])
                for view_name in local_transit_feature]

            # view_inputs = [self.inputs[view_name] for view_name in local_transit_feature]
            view_inputs = tf.keras.layers.Concatenate(
                axis=-1,
                name=f'local_flux_concat_{local_transit_feature[0]}_with_var')(view_inputs)

            local_transit_views_inputs.append(view_inputs)

        # combine the local transits to put through conv block (dim = [view, bins, avg/var view])
        if len(local_transit_views_inputs) > 1:
            branch_view_inputs = tf.keras.layers.Concatenate(
                axis=1,
                name='local_flux_concat_local_views')(local_transit_views_inputs)
        else:
            branch_view_inputs = local_transit_views_inputs[0]

        # convolve inputs with convolutional blocks
        # get init parameters for the given view
        n_blocks = self.config[config_mapper['blocks']['local_view']]
        kernel_size = self.config[config_mapper['kernel_size']['local_view']]
        pool_size = self.config[config_mapper['pool_size']['local_view']]
        conv_ls_per_block = self.config[config_mapper['conv_ls_per_block']['local_view']]
        init_conv_filters = self.config[config_mapper['init_conv_filters']['local_view']]

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (init_conv_filters + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': (1, kernel_size),
                           'strides': self.config['kernel_stride'],
                           'padding': 'same'
                           }

            for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional block

                net = tf.keras.layers.Conv2D(dilation_rate=1,
                                             activation=None,
                                             use_bias=True,
                                             bias_initializer='zeros',
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None,
                                             name='local_flux_conv{}_{}'.format(conv_block_i, seq_conv_block_i),
                                             **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                  seq_conv_block_i == 0
                                                            else net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='local_flux_lrelu_{}_{}'.format(conv_block_i,
                                                                              seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='local_flux_relu_{}_{}'.format(conv_block_i,
                                                                        seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1, 2],
                                                name='local_flux_prelu_{}_{}'.format(conv_block_i,
                                                                          seq_conv_block_i))(net)

            net = tf.keras.layers.MaxPooling2D(pool_size=(1, pool_size),
                                               strides=(1, self.config['pool_stride']),
                                               name='local_flux_maxpooling_{}'.format(conv_block_i))(net)

        # split up extracted features for each local transit view (flux, secondary, and odd-even)
        net = SplitLayer(net.shape[1], axis=1, name='local_flux_split_merge')(net)

        # combine them again based on which branch they are in
        cur = 0
        for branch in conv_branches:
            sz = local_transit_sz[branch]
            if sz != 1:  # multiple views in branch
                conv_branches[branch] = (
                    tf.keras.layers.Concatenate(axis=1,
                                                name='local_flux_transit_merge_{}'.format(branch))(net[cur:cur + sz]))
                cur += sz
            else:  # only one view in branch
                conv_branches[branch] = net[cur]
                cur += 1

        for branch_i, branch in enumerate(conv_branches):  # create a convolutional branch

            net = conv_branches[branch]

            if branch == odd_even_branch_name:  # subtract extracted features for odd and even views branch
                net = SplitLayer(2, axis=1, name='local_flux_split_oe')(conv_branches[branch])
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)

            # flatten output of the convolutional branch
            net = tf.keras.layers.Flatten(data_format='channels_last', name='local_flux_flatten_{}'.format(branch))(net)

            # concatenate scalar features with features extracted in the convolutional branch for the time series views
            if self.config['conv_branches'][branch]['scalars'] is not None:
                scalar_inputs = \
                    [self.inputs[feature_name] for feature_name in self.config['conv_branches'][branch]['scalars']]
                if len(scalar_inputs) > 1:
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1,
                                                                name=f'local_flux_{branch}_scalar_input')(scalar_inputs)
                else:
                    scalar_inputs = scalar_inputs[0]

                net = tf.keras.layers.Concatenate(axis=1, name='local_flux_flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_inputs
                ])

            # add FC layer that extracts features from the combined feature vector of features from the convolutional
            # branch (flattened) and corresponding scalar features
            if self.config['num_fc_conv_units'] > 0:
                net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='local_flux_fc_{}'.format(branch))(net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01, name='local_flux_fc_lrelu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='local_flux_fc_relu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1],
                                                name='local_flux_fc_prelu_{}'.format(branch))(net)

                net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'],
                                              name=f'local_flux_dropout_fc_conv_{branch}')(net)

            conv_branches[branch] = net

        return conv_branches

    def build_diff_img_branch(self):
        """ Builds the difference image branch.

        :return: dict with the difference image branch
        """

        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        branch_view_inputs = [self.inputs[view_name] for view_name in self.config['diff_img_branch']['imgs']]

        # expanding dimensionality
        branch_view_inputs = [tf.keras.layers.Reshape(l.shape[1:] + (1,), name=f'expanding_{l.name}_dims')(l)
                              for l in branch_view_inputs]

        branch_view_inputs = tf.keras.layers.Concatenate(axis=4, name='input_diff_img_concat')(branch_view_inputs)

        # get number of conv blocks for the given view
        n_blocks = 3
        kernel_size = (3, 3, 1)
        pool_size = (2, 2, 1)

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (2 + conv_block_i)
            # num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': kernel_size,
                           'strides': 1,
                           'padding': 'same'
                           }

            for seq_conv_block_i in range(self.config['conv_ls_per_block']):  # create convolutional block

                net = tf.keras.layers.Conv3D(dilation_rate=1,
                                             activation=None,
                                             use_bias=True,
                                             bias_initializer='zeros',
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None,
                                             name='convdiff_img_{}_{}'.format(conv_block_i, seq_conv_block_i),
                                             **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                                                                                  seq_conv_block_i == 0
                                                            else net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='lreludiff_img_{}_{}'.format(conv_block_i,
                                                                                      seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='reludiff_img_{}_{}'.format(conv_block_i,
                                                                                seq_conv_block_i))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1, 2],
                                                name='preludiff_img_{}_{}'.format(conv_block_i,
                                                                                  seq_conv_block_i))(net)

                net = tf.keras.layers.MaxPooling3D(pool_size=pool_size,
                                                   strides=self.config['pool_stride'],
                                                   name='maxpooling_diff_img_{}_{}'.format(conv_block_i,
                                                                                           seq_conv_block_i))(net)

        # flatten output of the convolutional branch
        net = tf.keras.layers.Permute((1, 2, 4, 3), name='permute_diff_imgs')(net)
        net = tf.keras.layers.Reshape((np.prod(list(net.shape[1:-1])), net.shape[-1]),
                                      name='flatten_diff_imgs')(net)

        # add per-image scalar features
        if self.config['diff_img_branch']['imgs_scalars'] is not None:

            scalar_inputs = [tf.keras.layers.Permute((2, 1),
                                                     name=f'permute_{feature_name}')(self.inputs[feature_name])
                             if 'pixel' not in feature_name else self.inputs[feature_name]
                             for feature_name in self.config['diff_img_branch']['imgs_scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'diff_img_imgsscalars_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            # concatenate per-image scalar features with extracted features from the difference images
            net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_diff_img_imgsscalars')([net, scalar_inputs])

        net = tf.keras.layers.Conv1D(filters=self.config['num_fc_conv_units'],
                                     kernel_size=net.shape[1:-1],
                                     strides=1,
                                     padding='valid',
                                     kernel_initializer=weight_initializer,
                                     dilation_rate=1,
                                     activation=None,
                                     use_bias=True,
                                     bias_initializer='zeros',
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     kernel_constraint=None,
                                     bias_constraint=None,
                                     name='convfc_{}'.format('diff_img'),
                                     )(net)

        if self.config['non_lin_fn'] == 'lrelu':
            net = tf.keras.layers.LeakyReLU(alpha=0.01, name='convfc_lrelu_diff_img')(net)
        elif self.config['non_lin_fn'] == 'relu':
            net = tf.keras.layers.ReLU(name='convfc_relu_diff_img')(net)
        elif self.config['non_lin_fn'] == 'prelu':
            net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                        alpha_regularizer=None,
                                        alpha_constraint=None,
                                        shared_axes=[1],
                                        name='convfc_prelu_diff_img')(net)

        net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_convfc_diff_img')(net)

        # add scalar features
        if self.config['diff_img_branch']['scalars'] is not None:

            scalar_inputs = [self.inputs[feature_name] for feature_name in self.config['diff_img_branch']['scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'diff_img_scalars_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            # concatenate scalar features with remaining features
            net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_diff_img_scalars')([net, scalar_inputs])

        # add FC layer that extracts features from the combined feature vector of features from the convolutional
        # branch (flattened) and corresponding scalar features
        if self.config['num_fc_conv_units'] > 0:
            net = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                        kernel_regularizer=None,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        name='fc_diff_img')(net)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_diff_img')(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu_diff_img')(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu_diff_img')(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'], name=f'dropout_fc_diff_img')(net)

        return {'diff_img': net}

    def build_scalar_branches(self):
        """ Builds the scalar branches.

        :return:
            scalar_branches_net, dict with the different scalar branches
        """

        scalar_branches_net = {scalar_branch_name: None for scalar_branch_name in self.config['scalar_branches']}
        for scalar_branch_name in self.config['scalar_branches']:

            scalar_inputs = [self.inputs[feature_name] for feature_name in
                             self.config['scalar_branches'][scalar_branch_name]]
            if len(scalar_inputs) > 1:
                scalar_input = tf.keras.layers.Concatenate(axis=1, name=f'{scalar_branch_name}_scalar_input')(
                    scalar_inputs)
            else:
                scalar_input = scalar_inputs[0]

            scalar_fc_output = tf.keras.layers.Dense(units=self.config['num_fc_conv_units'],
                                                     kernel_regularizer=regularizers.l2(self.config['decay_rate']) if
                                                     self.config['decay_rate'] is not None else None,
                                                     activation=None,
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     bias_initializer='zeros',
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name=f'fc_{scalar_branch_name}_scalar')(scalar_input)

            if self.config['non_lin_fn'] == 'lrelu':
                scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name=f'fc_lrelu_{scalar_branch_name}_scalar')(
                    scalar_fc_output)
            elif self.config['non_lin_fn'] == 'relu':
                scalar_fc_output = tf.keras.layers.ReLU(name=f'fc_relu_{scalar_branch_name}_scalar')(scalar_fc_output)
            elif self.config['non_lin_fn'] == 'prelu':
                scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                         alpha_regularizer=None,
                                                         alpha_constraint=None,
                                                         shared_axes=[1],
                                                         name=f'fc_prelu_{scalar_branch_name}_scalar')(scalar_fc_output)
            scalar_branches_net[scalar_branch_name] = scalar_fc_output  # scalar_input  # scalar_fc_output

        return scalar_branches_net

    def connect_segments(self, branches):
        """ Connect the different branches.

        :param branches: dict, branches to be concatenated
        :return:
            model output before FC layers
        """

        branches_to_concatenate = []
        for branch_name, branch in branches.items():
            branches_to_concatenate.append(branch)

        if len(branches_to_concatenate) > 1:
            net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(branches_to_concatenate)
        else:
            net = branches_to_concatenate[0]

        return net

    def build_fc_block(self, net):
        """ Builds the FC block after the convolutional branches.

        :param net: model upstream the FC block
        :return:
            net: model with added FC block
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']) if self.config[
                                                                                  'decay_rate'] is not None else None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net)
            else:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='fc{}'.format(fc_layer_i))(net)

            if self.config['non_lin_fn'] == 'lrelu':
                net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'relu':
                net = tf.keras.layers.ReLU(name='fc_relu{}'.format(fc_layer_i))(net)
            elif self.config['non_lin_fn'] == 'prelu':
                net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1],
                                            name='fc_prelu{}'.format(fc_layer_i))(net)

            net = tf.keras.layers.Dropout(self.config['dropout_rate'], name=f'dropout_fc{fc_layer_i}')(net)

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {}

        if self.config['scalar_branches'] is not None:
            branches_net.update(self.build_scalar_branches())

        if self.config['conv_branches'] is not None:
            branches_net.update(self.build_conv_branches())
            branches_net.update(self.build_joint_local_conv_branches())
            if 'local_unfolded_flux' in self.config['conv_branches']:
                branches_net.update(self.build_conv_unfolded_flux())

        if self.config['diff_img_branch'] is not None:
            branches_net.update(self.build_diff_img_branch())

        # merge branches
        net = self.connect_segments(branches_net)

        # create FC layers
        net = self.build_fc_block(net)

        # create output layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output



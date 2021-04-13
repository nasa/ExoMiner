""" Implementation of models. """

# 3rd party
import tensorflow as tf
import operator
from tensorflow.keras import regularizers
from tensorflow import keras


class CNN1dPlanetFinderv1(object):

    def __init__(self, config, features, scalar_params_idxs):
        """ Initializes the CNN 1D Planet Finder v1 model. The core structure consists of separate convolutional
        branches for non-related types of input (flux, centroid, odd and even time series, ...).

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        :param scalar_params_idxs: list, containing indices of the scalar parameters that are used
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features
        self.scalar_params_idxs = scalar_params_idxs

        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # ['global_flux_view', 'local_flux_view', 'global_centr_view_medcmaxn',
        #  'local_centr_view_medcmaxn', 'local_weak_secondary_view', 'local_flux_oddeven_views']
        if 'branches' not in self.config:
            self.branches = ['global_flux_view', 'local_flux_view']
        else:
            self.branches = self.config['branches']

        # self.is_training = None

        # # if doing multiclassification or using softmax as output layer, the output has to be equal to the number of
        # # classes
        # if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
        #     self.output_size = max(config['label_map'].values()) + 1
        # else:  # binary classification with sigmoid output layer
        #     self.output_size = 1
        #
        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = self.create_inputs()

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def create_inputs(self):

        inputs = {}

        for feature in self.features:

            if feature == 'scalar_params' and self.scalar_params_idxs is not None:
                input_feat_shape = len(self.scalar_params_idxs)
            else:
                input_feat_shape = self.features[feature]['dim']

            input = tf.keras.Input(shape=input_feat_shape,
                                   batch_size=None,
                                   name='{}'.format(feature),
                                   dtype=self.features[feature]['dtype'],
                                   sparse=False,
                                   tensor=None,
                                   ragged=False)

            inputs[feature] = input

        return inputs

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            if 'oddeven' not in branch:
                input_branch = self.inputs[branch]

            elif branch == 'local_flux_oddeven_views':

                # input_branch = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format('local_view_oddeven'))(
                #     [self.inputs['local_flux_odd_view'], self.inputs['local_flux_even_view']])
                input_branch = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format('local_view_oddeven'))([self.inputs['local_flux_odd_view_fluxnorm'], self.inputs['local_flux_even_view_fluxnorm']])

            elif branch == 'global_flux_oddeven_views':

                # input_branch = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format('global_view_oddeven'))(
                #     [self.inputs['global_flux_odd_view'], self.inputs['global_flux_even_view']])
                input_branch = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format('global_view_oddeven'))([self.inputs['global_flux_odd_view_fluxnorm'], self.inputs['global_flux_even_view_fluxnorm']])

            # with tf.variable_scope('ConvNet_%s' % view):

            # get number of conv blocks for the given view
            n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

            # get pool size for the given view
            pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

            for conv_block_i in range(n_blocks):

                # num_filters = self.config['init_conv_filters'] * (2 ** conv_block_i)
                num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters,
                          'kernel_initializer': weight_initializer,
                          'kernel_size': self.config['kernel_size'],
                          'strides': self.config['kernel_stride'],
                          'padding': 'same'}

                for seq_conv_block_i in range(self.config['conv_ls_per_block']):

                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None,
                                                 name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                 **kwargs)(input_branch if conv_block_i == 0 and
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

                net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=self.config['pool_stride'],
                                                   name='maxpooling{}{}'.format(branch, conv_block_i))(net)

                # if self.config['batch_norm'] and conv_block_i == n_blocks - 1:
                #     tf.keras.layers.BatchNormalization(axis=-1,
                #                                        momentum=0.99,
                #                                        epsilon=1e-3,
                #                                        center=True,
                #                                        scale=True,
                #                                        beta_initializer='zeros',
                #                                        gamma_initializer='ones',
                #                                        moving_mean_initializer='zeros',
                #                                        moving_variance_initializer='ones',
                #                                        beta_regularizer=None,
                #                                        gamma_regularizer=None,
                #                                        beta_constraint=None,
                #                                        gamma_constraint=None,
                #                                        renorm=False,
                #                                        renorm_clipping=None,
                #                                        renorm_momentum=0.99,
                #                                        fused=None,
                #                                        trainable=self.is_training,
                #                                        virtual_batch_size=None,
                #                                        adjustment=None,
                #                                        name='batch_norm{}_{}'.format(view, conv_block_i))(net)

            # Flatten
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv branches
        :return:
            model output before FC layers
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if 'scalar_params' in self.features:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)([
                pre_logits_concat, self.inputs['scalar_params']])

        if self.config['batch_norm']:
            pre_logits_concat = tf.keras.layers.BatchNormalization(axis=-1,
                                                                   momentum=0.99,
                                                                   epsilon=1e-3,
                                                                   center=True,
                                                                   scale=True,
                                                                   beta_initializer='zeros',
                                                                   gamma_initializer='ones',
                                                                   moving_mean_initializer='zeros',
                                                                   moving_variance_initializer='ones',
                                                                   beta_regularizer=None,
                                                                   gamma_regularizer=None,
                                                                   beta_constraint=None,
                                                                   gamma_constraint=None,
                                                                   renorm=False,
                                                                   renorm_clipping=None,
                                                                   renorm_momentum=0.99,
                                                                   fused=None,
                                                                   trainable=True,
                                                                   virtual_batch_size=None,
                                                                   adjustment=None,
                                                                   name='batch_norm_convflt_output')(pre_logits_concat)

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config['num_fc_layers']):

            # fc_neurons = self.config.init_fc_neurons / (2 ** fc_layer_i)
            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']),
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

            # TODO: investigate this, is it set automatically?
            # net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=keras.backend.learning_phase())
            net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)

        # create output FC layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output

    def build(self):
        """ Builds the model.

        :return:
        """

        # create convolutional branches
        cnn_layers = self.build_cnn_layers()

        # merge convolutional branches
        net = self.connect_segments(cnn_layers)

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


class CNN1dPlanetFinderv2(object):

    def __init__(self, config, features):
        """ Initializes the CNN 1D Planet Finder v1 model. The core structure consists of separate convolutional
        branches for non-related types of input (flux, centroid, odd and even time series, ...).

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features

        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        if 'branches' not in self.config:
            self.branches = ['global_flux_view', 'local_flux_view']
        else:
            self.branches = self.config['branches']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = self.create_inputs()

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def create_inputs(self):

        inputs = {}

        for feature in self.features:

            input = tf.keras.Input(shape=self.features[feature]['dim'],
                                   batch_size=None,
                                   name='{}'.format(feature),
                                   dtype=self.features[feature]['dtype'],
                                   sparse=False,
                                   tensor=None,
                                   ragged=False)

            inputs[feature] = input

        return inputs

    def build_cnn_layers(self):
        """ Builds the convolutional branches.

        :return:
            cnn_layers, dict with the different convolutional branches
        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            if branch == 'local_flux_oddeven_views':

                input_branch = \
                    tf.keras.layers.Concatenate(axis=2,
                                                name='input_{}'.format('local_view_oddeven'
                                                                       ))([self.inputs['local_flux_odd_view_fluxnorm'],
                                                                           self.inputs['local_flux_even_view_fluxnorm']]
                                                                          )
                input_branch = tf.keras.layers.Permute((2, 1), name='permute_oe')(input_branch)
                input_branch = tf.expand_dims(input_branch, axis=-1, name='expanding_oe')

                # get number of conv blocks for the given view
                n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

                # get pool size for the given view
                pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

                for conv_block_i in range(n_blocks):

                    num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                    # set convolution layer parameters from config
                    kwargs = {'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': (1, self.config['kernel_size']),
                              'strides': (1, self.config['kernel_stride']),
                              'padding': 'same'
                              }

                    for seq_conv_block_i in range(self.config['conv_ls_per_block']):

                        net = tf.keras.layers.Conv2D(dilation_rate=1,
                                                     activation=None,
                                                     use_bias=True,
                                                     bias_initializer='zeros',
                                                     kernel_regularizer=None,
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                     **kwargs)(input_branch if conv_block_i == 0 and
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

                    net = tf.keras.layers.MaxPooling2D(pool_size=(1, pool_size),
                                                       strides=(1, self.config['pool_stride']),
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)

                net = tf.split(net, 2, axis=1, name='split_oe')
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)
                # net = tf.keras.layers.Permute((2, 3, 1), name='permute2_oe')(net)  # needed for Conv2D

                # flatten output of the convolutional branch
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)
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

                # net = tf.keras.layers.Conv1D(filters=self.config['num_fc_conv_units'],
                #                              kernel_size=net.shape[1],
                #                              strides=1,
                #                              padding='valid',
                #                              kernel_initializer=weight_initializer,
                #                              dilation_rate=1,
                #                              activation=None,
                #                              use_bias=True,
                #                              bias_initializer='zeros',
                #                              kernel_regularizer=None,
                #                              bias_regularizer=None,
                #                              activity_regularizer=None,
                #                              kernel_constraint=None,
                #                              bias_constraint=None,
                #                              name='conv_{}'.format(branch),
                #                              )(net)

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

                # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)  # needed for Conv2D

                net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

                cnn_layers[branch] = net

            else:

                input_branch = self.inputs[branch]

                # with tf.variable_scope('ConvNet_%s' % view):

                # get number of conv blocks for the given view
                n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

                # get pool size for the given view
                pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

                for conv_block_i in range(n_blocks):

                    num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                    # set convolution layer parameters from config
                    kwargs = {'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
                              'padding': 'same'
                              }

                    for seq_conv_block_i in range(self.config['conv_ls_per_block']):

                        net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                     activation=None,
                                                     use_bias=True,
                                                     bias_initializer='zeros',
                                                     kernel_regularizer=None,
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                     **kwargs)(input_branch if conv_block_i == 0 and
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

                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=self.config['pool_stride'],
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)

                # flatten output of the convolutional branch
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

                # add scalar input
                if 'local_weak_secondary_view' in branch:
                    scalar_input = tf.keras.layers.Concatenate(axis=1, name='wks_scalar_input')(
                        [
                        # self.inputs['tce_maxmes_norm'],
                        # self.inputs['tce_albedo_stat_norm'],
                        # self.inputs['tce_ptemp_stat_norm'],
                        self.inputs['wst_depth_norm']
                        ])

                    net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                        net,
                        scalar_input
                    ])

                elif 'local_centr_view' in branch:
                    scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')(
                        [
                            self.inputs['tce_dikco_msky_norm'],
                            self.inputs['tce_dikco_msky_err_norm'],
                            self.inputs['tce_dicco_msky_norm'],
                            self.inputs['tce_dicco_msky_err_norm'],
                            self.inputs['tce_fwm_stat_norm']
                        ])

                    net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                        net,
                        scalar_input
                    ])

                elif 'local_flux_view' in branch:
                    scalar_input = self.inputs['transit_depth_norm']

                    net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                        net,
                        scalar_input
                    ])

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

                # net = tf.expand_dims(net, axis=-1)
                # net = tf.keras.layers.Conv1D(filters=self.config['num_fc_conv_units'],
                #                              kernel_size=net.shape[1],
                #                              strides=1,
                #                              padding='valid',
                #                              kernel_initializer=weight_initializer,
                #                              dilation_rate=1,
                #                              activation=None,
                #                              use_bias=True,
                #                              bias_initializer='zeros',
                #                              kernel_regularizer=None,
                #                              bias_regularizer=None,
                #                              activity_regularizer=None,
                #                              kernel_constraint=None,
                #                              bias_constraint=None,
                #                              name='conv_{}'.format(branch),
                #                              )(net)

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

                net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

                cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different convolutional branches; it also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different convolutional branches
        :return:
            model output before FC layers
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            net = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            net = tf.keras.layers.Concatenate(name='convbranch_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # scalar_input = tf.keras.layers.Concatenate(axis=1, name='stellar_dv_scalar_input')(
        #     [
        #         self.inputs['tce_steff_norm'],
        #         self.inputs['tce_slogg_norm'],
        #         self.inputs['tce_smet_norm'],
        #         self.inputs['tce_sradius_norm'],
        #         self.inputs['tce_smass_norm'],
        #         self.inputs['tce_sdens_norm'],
        #         self.inputs['tce_cap_stat_norm'],
        #         self.inputs['tce_hap_stat_norm'],
        #         self.inputs['tce_rb_tcount0_norm'],
        #         self.inputs['boot_fap_norm'],
        #         self.inputs['tce_period_norm'],
        #         self.inputs['tce_prad_norm']
        #     ])

        # net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')([
        #     net,
        #     scalar_input
        # ])

        stellar_scalar_input = tf.keras.layers.Concatenate(axis=1, name='stellar_scalar_input')(
            [
                self.inputs['tce_steff_norm'],
                self.inputs['tce_slogg_norm'],
                self.inputs['tce_smet_norm'],
                self.inputs['tce_sradius_norm'],
                self.inputs['tce_smass_norm'],
                self.inputs['tce_sdens_norm'],
            ])

        stellar_scalar_fc_output = tf.keras.layers.Dense(units=4,
                                    kernel_regularizer=regularizers.l2(
                                        self.config['decay_rate']),
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    name='fc_stellar_scalar')(stellar_scalar_input)

        if self.config['non_lin_fn'] == 'lrelu':
            stellar_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_stellar_scalar')(stellar_scalar_fc_output)
        elif self.config['non_lin_fn'] == 'relu':
            stellar_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_stellar_scalar')(stellar_scalar_fc_output)
        elif self.config['non_lin_fn'] == 'prelu':
            stellar_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                        alpha_regularizer=None,
                                        alpha_constraint=None,
                                        shared_axes=[1],
                                        name='fc_prelu_fc_relu_stellar_scalar')(stellar_scalar_fc_output)

        dv_scalar_input = tf.keras.layers.Concatenate(axis=1, name='dv_scalar_input')([
            self.inputs['tce_cap_stat_norm'],
            self.inputs['tce_hap_stat_norm'],
            self.inputs['tce_rb_tcount0_norm'],
            self.inputs['boot_fap_norm'],
            self.inputs['tce_period_norm'],
            self.inputs['tce_prad_norm']
        ])

        dv_scalar_fc_output = tf.keras.layers.Dense(units=4,
                                    kernel_regularizer=regularizers.l2(
                                        self.config['decay_rate']),
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    name='fc_dv_scalar')(dv_scalar_input)

        if self.config['non_lin_fn'] == 'lrelu':
            dv_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_dv_scalar')(dv_scalar_fc_output)
        elif self.config['non_lin_fn'] == 'relu':
            dv_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_dv_scalar')(dv_scalar_fc_output)
        elif self.config['non_lin_fn'] == 'prelu':
            dv_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                        alpha_regularizer=None,
                                        alpha_constraint=None,
                                        shared_axes=[1],
                                        name='fc_prelu_fc_relu_dv_scalar')(dv_scalar_fc_output)

        net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')([
            net,
            stellar_scalar_fc_output,
            dv_scalar_fc_output
        ])

        if self.config['batch_norm']:
            net = tf.keras.layers.BatchNormalization(axis=-1,
                                                     momentum=0.99,
                                                     epsilon=1e-3,
                                                     center=True,
                                                     scale=True,
                                                     beta_initializer='zeros',
                                                     gamma_initializer='ones',
                                                     moving_mean_initializer='zeros',
                                                     moving_variance_initializer='ones',
                                                     beta_regularizer=None,
                                                     gamma_regularizer=None,
                                                     beta_constraint=None,
                                                     gamma_constraint=None,
                                                     renorm=False,
                                                     renorm_clipping=None,
                                                     renorm_momentum=0.99,
                                                     fused=None,
                                                     trainable=True,
                                                     virtual_batch_size=None,
                                                     adjustment=None,
                                                     name='batch_norm_convbranch_wscalar_concat')(net)

        return net

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']),
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

            net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)

        # create output FC layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output

    def build(self):
        """ Builds the model.

        :return:
        """

        # create convolutional branches
        cnn_layers = self.build_cnn_layers()

        # merge convolutional branches
        net = self.connect_segments(cnn_layers)

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


class Astronet(object):

    def __init__(self, config, features, scalar_params_idxs):
        """ Initializes the Astronet model. The core structure consists of two convolutional
        branches - one for the global view flux time series and another one for the local flux view.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        :param scalar_params_idxs: list, containing indices of the scalar parameters that are used
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features
        self.scalar_params_idxs = scalar_params_idxs

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_flux_view_fluxnorm', 'local_flux_view_fluxnorm']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = self.create_inputs()

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def create_inputs(self):

        inputs = {}

        for feature in self.features:

            if feature == 'scalar_params' and self.scalar_params_idxs is not None:
                input_feat_shape = len(self.scalar_params_idxs)
            else:
                input_feat_shape = self.features[feature]['dim']

            input = tf.keras.Input(shape=input_feat_shape,
                                   batch_size=None,
                                   name='{}'.format(feature),
                                   dtype=self.features[feature]['dtype'],
                                   sparse=False,
                                   tensor=None,
                                   ragged=False)

            inputs[feature] = input

        return inputs

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            input_branch = self.inputs[branch]

            # with tf.variable_scope('ConvNet_%s' % view):

            # get number of conv blocks for the given view
            n_blocks = 2 if 'local' in branch else 5

            # get pool size for the given view
            pool_size = 7 if 'local' in branch else 5

            # number of filters in each convolutional block
            num_filters = [16, 32, 64, 128, 256] if 'global' in branch else [16, 32]

            for conv_block_i in range(n_blocks):

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters[conv_block_i],
                          'kernel_initializer': weight_initializer,
                          'kernel_size': 5,
                          'strides': 1,
                          'padding': "same"}

                for seq_conv_block_i in range(2):

                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None,
                                                 name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                 **kwargs)(input_branch if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0
                                                                                     else net)

                    net = tf.keras.layers.ReLU()(net)

                net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=2,
                                                   name='maxpooling{}{}'.format(branch, conv_block_i))(net)

            # Flatten
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv branches
        :return:
            model output before FC layers
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if 'scalar_params' in self.features:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)([
                pre_logits_concat, self.inputs['scalar_params']])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers.

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(4):

            fc_neurons = 512

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']),
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

            net = tf.keras.layers.LeakyReLU(alpha=0.01)(net) if self.config['non_lin_fn'] == 'prelu' \
                else tf.keras.layers.ReLU()(net)

            # TODO: investigate this, is it set automatically?
            # net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=keras.backend.learning_phase())
            net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)

        # create output FC layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output

    def build(self):
        """ Builds the model.

        :return:
        """

        # create convolutional branches
        cnn_layers = self.build_cnn_layers()

        # merge convolutional branches
        net = self.connect_segments(cnn_layers)

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


class Exonet(object):

    def __init__(self, config, features, scalar_params_idxs):
        """ Initializes the Exonet model. The core structure consists of two convolutional
        branches - one for the global view time series and another one for the local view ones.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        :param scalar_params_idxs: list, containing indices of the scalar parameters that are used
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features
        self.scalar_params_idxs = scalar_params_idxs

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_view', 'local_view']
        # self.branches = ['global_flux_view_fluxnorm', 'local_flux_view_fluxnorm', 'global_centr_fdl_view', 'local_centr_fdl_view']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = self.create_inputs()

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def create_inputs(self):

        inputs = {}

        for feature in self.features:

            if feature == 'scalar_params' and self.scalar_params_idxs is not None:
                input_feat_shape = len(self.scalar_params_idxs)
            else:
                input_feat_shape = self.features[feature]['dim']

            input = tf.keras.Input(shape=input_feat_shape,
                                   batch_size=None,
                                   name='{}'.format(feature),
                                   dtype=self.features[feature]['dtype'],
                                   sparse=False,
                                   tensor=None,
                                   ragged=False)

            inputs[feature] = input

        return inputs

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            if branch == 'global_view':
                input_branch = tf.keras.layers.Concatenate(name='global_input', axis=2)([self.inputs[input]
                                                                                         for input in self.inputs
                                                                                         if 'global' in input])
            else:
                input_branch = tf.keras.layers.Concatenate(name='local_input', axis=2)([self.inputs[input]
                                                                                        for input in self.inputs
                                                                                        if 'local' in input])

            # with tf.variable_scope('ConvNet_%s' % view):

            # get number of conv blocks for the given view
            n_blocks = 2 if 'local' in branch else 5

            # get pool size for the given view
            pool_size = 7 if 'local' in branch else 5

            # number of filters in each convolutional block
            num_filters = {'global_view': [16, 32, 64, 128, 256], 'local_view': [16, 32]}

            for conv_block_i in range(n_blocks):

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters[branch][conv_block_i],
                          'kernel_initializer': weight_initializer,
                          'kernel_size': 5,
                          'strides': 1,
                          'padding': "same"}

                for seq_conv_block_i in range(2):

                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None,
                                                 name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                 **kwargs)(input_branch if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0
                                                                                     else net)

                    net = tf.keras.layers.ReLU()(net)

                net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=2,
                                                   name='maxpooling{}{}'.format(branch, conv_block_i))(net)

            # Flatten
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv branches
        :return:
            model output before FC layers
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if 'scalar_params' in self.features:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)([
                pre_logits_concat, self.inputs['scalar_params']])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers.

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(4):

            fc_neurons = 512

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']),
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

            net = tf.keras.layers.LeakyReLU(alpha=0.01)(net) if self.config['non_lin_fn'] == 'prelu' \
                else tf.keras.layers.ReLU()(net)

            # TODO: investigate this, is it set automatically?
            # net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=keras.backend.learning_phase())
            net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)

        # create output FC layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output

    def build(self):
        """ Builds the model.

        :return:
        """

        # create convolutional branches
        cnn_layers = self.build_cnn_layers()

        # merge convolutional branches
        net = self.connect_segments(cnn_layers)

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


class Exonet_XS(object):

    def __init__(self, config, features, scalar_params_idxs):
        """ Initializes the Exonet-XS model which is a smaller version of Exonet. The core structure consists of two
        convolutional branches - one for the global view time series and another one for the local view ones.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        :param scalar_params_idxs: list, containing indices of the scalar parameters that are used
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features
        self.scalar_params_idxs = scalar_params_idxs

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_view', 'local_view']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = self.create_inputs()

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def create_inputs(self):

        inputs = {}

        for feature in self.features:

            if feature == 'scalar_params' and self.scalar_params_idxs is not None:
                input_feat_shape = len(self.scalar_params_idxs)
            else:
                input_feat_shape = self.features[feature]['dim']

            input = tf.keras.Input(shape=input_feat_shape,
                                   batch_size=None,
                                   name='{}'.format(feature),
                                   dtype=self.features[feature]['dtype'],
                                   sparse=False,
                                   tensor=None,
                                   ragged=False)

            inputs[feature] = input

        return inputs

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            if branch == 'global_view':
                input_branch = tf.keras.layers.Concatenate(name='global_input', axis=2)([self.inputs[input]
                                                                                         for input in self.inputs
                                                                                         if 'global' in input])
            else:
                input_branch = tf.keras.layers.Concatenate(name='local_input', axis=2)([self.inputs[input]
                                                                                        for input in self.inputs
                                                                                        if 'local' in input])

            # with tf.variable_scope('ConvNet_%s' % view):

            # get number of conv blocks for the given view
            n_blocks = 2 if 'local' in branch else 3

            # get pool size for the given view
            pool_size = 2

            # number of filters in each convolutional block
            num_filters = {'global_view': [16, 16, 32], 'local_view': [16, 16]}

            for conv_block_i in range(n_blocks):

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters[branch][conv_block_i],
                          'kernel_initializer': weight_initializer,
                          'kernel_size': 5,
                          'strides': 1,
                          'padding': "same"}

                for seq_conv_block_i in range(2):
                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None,
                                                 name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                                 **kwargs)(input_branch if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0
                                                           else net)

                    net = tf.keras.layers.ReLU()(net)

                if conv_block_i == n_blocks - 1:
                    net = tf.keras.layers.GlobalMaxPooling1D(name='globalmaxpooling{}{}'.format(branch,
                                                                                                conv_block_i))(net)
                else:
                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=2,
                                                       name='maxpooling{}{}'.format(branch, conv_block_i))(net)

            # Flatten
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv branches
        :return:
            model output before FC layers
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if 'scalar_params' in self.features:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)([
                pre_logits_concat, self.inputs['scalar_params']])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers.

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(1):

            fc_neurons = 512

            if self.config['decay_rate'] is not None:
                net = tf.keras.layers.Dense(units=fc_neurons,
                                            kernel_regularizer=regularizers.l2(
                                                self.config['decay_rate']),
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

            net = tf.keras.layers.LeakyReLU(alpha=0.01)(net) if self.config['non_lin_fn'] == 'prelu' \
                else tf.keras.layers.ReLU()(net)

            # TODO: investigate this, is it set automatically?
            # net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=keras.backend.learning_phase())
            net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)

        # create output FC layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')(logits)
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name='softmax')(logits)

        return output

    def build(self):
        """ Builds the model.

        :return:
        """

        # create convolutional branches
        cnn_layers = self.build_cnn_layers()

        # merge convolutional branches
        net = self.connect_segments(cnn_layers)

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


def create_inputs(features):
    """ Create input layers for the input features.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :return:
        inputs: dictionary, each key-value pair is a feature_name: feature
    """

    inputs = {}

    for feature in features:

        input = tf.keras.Input(shape=features[feature]['dim'],
                               batch_size=None,
                               name=f'{feature}',
                               dtype=features[feature]['dtype'],
                               sparse=False,
                               tensor=None,
                               ragged=False)

        inputs[feature] = input

    return inputs


def create_ensemble(features, models):
    """ Create a Keras ensemble.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :param models: list, list of Keras models
    :return:
        Keras ensemble
    """

    inputs = create_inputs(features=features)

    single_models_outputs = [model(inputs) for model in models]

    outputs = tf.keras.layers.Average()(single_models_outputs)

    return keras.Model(inputs=inputs, outputs=outputs)

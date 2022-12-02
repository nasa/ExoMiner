""" Implementation of models using Keras functional API. """

# 3rd party
import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, losses, optimizers
import numpy as np
import tensorflow.keras.backend as K


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
            self.output_size = len(config['label_map'])
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        if 'branches' not in self.config:
            self.branches = ['global_flux_view', 'local_flux_view']
        else:
            self.branches = self.config['branches']

        # self.is_training = None

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

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


class ExoMiner(object):

    def __init__(self, config, features):
        """ Initializes the CNN 1D Planet Finder v2 model. The core structure consists of separate convolutional
        branches for non-related types of input (flux, centroid, odd and even time series, ...).

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

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def build_conv_branches(self):
        """ Builds the convolutional branches.

        :return:
            conv_branches, dict with the different convolutional branches
        """

        odd_even_branch_name = 'local_odd_even'
        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in self.config['conv_branches']}
        for branch_i, branch in enumerate(self.config['conv_branches']):  # create a convolutional branch

            branch_view_inputs = [self.inputs[view_name] for view_name in self.config['conv_branches'][branch]['views']]

            branch_view_inputs = branch_view_inputs if len(branch_view_inputs) > 1 else branch_view_inputs[0]
            if branch == odd_even_branch_name:  # for odd and even views, inputs are combined as two channels

                branch_view_inputs = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format('branch'))(
                    branch_view_inputs)
                branch_view_inputs = tf.keras.layers.Permute((2, 1), name='permute_oe')(branch_view_inputs)
                branch_view_inputs = tf.expand_dims(branch_view_inputs, axis=-1, name='expanding_oe')

            # get number of conv blocks for the given view
            n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

            # get pool size for the given view
            pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

            for conv_block_i in range(n_blocks):  # create convolutional blocks

                num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': (1, self.config['kernel_size'])
                               if branch == odd_even_branch_name else self.config['kernel_size'],
                               'strides': (1, self.config['kernel_stride'])
                               if branch == odd_even_branch_name else self.config['kernel_stride'],
                               'padding': 'same'
                               }

                for seq_conv_block_i in range(self.config['conv_ls_per_block']):  # create convolutional block

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
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
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
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
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
                net = tf.split(net, 2, axis=1, name='split_oe')
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)
                # net = tf.keras.layers.Permute((2, 3, 1), name='permute2_oe')(net)  # needed for Conv2D

            # flatten output of the convolutional branch
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

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

            # if self.config['non_lin_fn'] == 'lrelu':
            #     net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'relu':
            #     net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'prelu':
            #     net = tf.keras.layers.PReLU(alpha_initializer='zeros',
            #                                 alpha_regularizer=None,
            #                                 alpha_constraint=None,
            #                                 shared_axes=[1],
            #                                 name='fc_prelu_{}'.format(branch))(net)
            #
            # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)  # needed for Conv2D
            #
            # net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

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

            # # concatenate corresponding scalar features to the flattened vector of features from the convolutional branch
            # if 'local_weak_secondary_view' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='wks_scalar_input')(
            #         [
            #             self.inputs['tce_maxmes_norm'],
            #             self.inputs['tce_albedo_stat_norm'],
            #             self.inputs['tce_ptemp_stat_norm'],
            #             self.inputs['wst_depth_norm'],
            #             # self.inputs['tce_period_norm'],
            #             # self.inputs['tce_prad_norm'],
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_centr_view' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')(
            #         [
            #             self.inputs['tce_dikco_msky_norm'],
            #             self.inputs['tce_dikco_msky_err_norm'],
            #             self.inputs['tce_dicco_msky_norm'],
            #             self.inputs['tce_dicco_msky_err_norm'],
            #             self.inputs['tce_fwm_stat_norm'],
            #             self.inputs['ruwe_norm'],
            #             # self.inputs['mag_norm'],
            #             # self.inputs['mag_cat'],
            #             # tf.cast(self.inputs['mag_cat'], tf.float32),
            #             self.inputs['mag_cat_norm']
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_flux_view' in branch:
            #     # scalar_input = self.inputs['transit_depth_norm']
            #
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='flux_scalar_input')(
            #         [
            #             self.inputs['transit_depth_norm'],
            #             self.inputs['tce_max_mult_ev_norm'],
            #             self.inputs['tce_max_sngle_ev_norm'],
            #             self.inputs['tce_robstat_norm'],
            #             self.inputs['tce_model_chisq_norm']
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_flux_oddeven_views' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='oddeven_scalar_input')(
            #         [
            #             # self.inputs['sigma_oot_odd'],
            #             # self.inputs['sigma_it_odd'],
            #             # self.inputs['sigma_oot_even'],
            #             # self.inputs['sigma_it_even'],
            #             self.inputs['odd_se_oot_norm'],
            #             self.inputs['even_se_oot_norm'],
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])

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

            scalar_fc_output = tf.keras.layers.Dense(units=4,
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
            scalar_branches_net[scalar_branch_name] = scalar_fc_output

        return scalar_branches_net

        # if 'stellar' in self.scalar_branches:
        #     stellar_scalar_input = tf.keras.layers.Concatenate(axis=1, name='stellar_scalar_input')(
        #         [
        #             self.inputs['tce_steff_norm'],
        #             self.inputs['tce_slogg_norm'],
        #             self.inputs['tce_smet_norm'],
        #             self.inputs['tce_sradius_norm'],
        #             self.inputs['tce_smass_norm'],
        #             self.inputs['tce_sdens_norm'],
        #         ])
        #
        #     stellar_scalar_fc_output = tf.keras.layers.Dense(units=4,
        #                                                      kernel_regularizer=regularizers.l2(
        #                                                          self.config['decay_rate']) if self.config[
        #                                                                                            'decay_rate'] is not None else None,
        #                                                      activation=None,
        #                                                      use_bias=True,
        #                                                      kernel_initializer='glorot_uniform',
        #                                                      bias_initializer='zeros',
        #                                                      bias_regularizer=None,
        #                                                      activity_regularizer=None,
        #                                                      kernel_constraint=None,
        #                                                      bias_constraint=None,
        #                                                      name='fc_stellar_scalar')(stellar_scalar_input)
        #
        #     if self.config['non_lin_fn'] == 'lrelu':
        #         stellar_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_stellar_scalar')(
        #             stellar_scalar_fc_output)
        #     elif self.config['non_lin_fn'] == 'relu':
        #         stellar_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_stellar_scalar')(stellar_scalar_fc_output)
        #     elif self.config['non_lin_fn'] == 'prelu':
        #         stellar_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
        #                                                          alpha_regularizer=None,
        #                                                          alpha_constraint=None,
        #                                                          shared_axes=[1],
        #                                                          name='fc_prelu_stellar_scalar')(
        #             stellar_scalar_fc_output)
        #
        #     all_branches.append(stellar_scalar_fc_output)
        #
        # if 'dv+tce_fit' in self.scalar_branches:
        #
        #     dv_scalar_input = tf.keras.layers.Concatenate(axis=1, name='dv_scalar_input')([
        #         self.inputs['tce_cap_stat_norm'],
        #         self.inputs['tce_hap_stat_norm'],
        #         # self.inputs['tce_cap_hap_stat_diff_norm'],
        #         self.inputs['tce_rb_tcount0n_norm'],
        #         # self.inputs['tce_rb_tcount0_norm'],
        #         self.inputs['boot_fap_norm'],
        #         self.inputs['tce_period_norm'],
        #         self.inputs['tce_prad_norm'],
        #         # self.inputs['tce_bin_oedp_stat_norm'],
        #         # self.inputs['koi_fpflag_ec'],
        #     ])
        #
        #     dv_scalar_fc_output = tf.keras.layers.Dense(units=4,
        #                                                 kernel_regularizer=regularizers.l2(
        #                                                     self.config['decay_rate']) if self.config[
        #                                                                                       'decay_rate'] is not None else None,
        #                                                 activation=None,
        #                                                 use_bias=True,
        #                                                 kernel_initializer='glorot_uniform',
        #                                                 bias_initializer='zeros',
        #                                                 bias_regularizer=None,
        #                                                 activity_regularizer=None,
        #                                                 kernel_constraint=None,
        #                                                 bias_constraint=None,
        #                                                 name='fc_dv_scalar')(dv_scalar_input)
        #
        #     if self.config['non_lin_fn'] == 'lrelu':
        #         dv_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_dv_scalar')(
        #             dv_scalar_fc_output)
        #     elif self.config['non_lin_fn'] == 'relu':
        #         dv_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_dv_scalar')(dv_scalar_fc_output)
        #     elif self.config['non_lin_fn'] == 'prelu':
        #         dv_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
        #                                                     alpha_regularizer=None,
        #                                                     alpha_constraint=None,
        #                                                     shared_axes=[1],
        #                                                     name='fc_prelu_dv_scalar')(dv_scalar_fc_output)
        #
        #     all_branches.append(dv_scalar_fc_output)
        #
        # # centroid_scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')([
        # #     self.inputs['tce_dikco_msky_norm'],
        # #     self.inputs['tce_dikco_msky_err_norm'],
        # #     # self.inputs['tce_dicco_msky_norm'],
        # #     # self.inputs['tce_dicco_msky_err_norm'],
        # # ])
        # #
        # # centroid_scalar_fc_output = tf.keras.layers.Dense(units=4,
        # #                             kernel_regularizer=regularizers.l2(
        # #                                 self.config['decay_rate']) if self.config['decay_rate'] is not None else None,
        # #                             activation=None,
        # #                             use_bias=True,
        # #                             kernel_initializer='glorot_uniform',
        # #                             bias_initializer='zeros',
        # #                             bias_regularizer=None,
        # #                             activity_regularizer=None,
        # #                             kernel_constraint=None,
        # #                             bias_constraint=None,
        # #                             name='fc_centroid_scalar')(centroid_scalar_input)
        # #
        # # if self.config['non_lin_fn'] == 'lrelu':
        # #     centroid_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_centroid_scalar')(centroid_scalar_fc_output)
        # # elif self.config['non_lin_fn'] == 'relu':
        # #     centroid_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_centroid_scalar')(centroid_scalar_fc_output)
        # # elif self.config['non_lin_fn'] == 'prelu':
        # #     centroid_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
        # #                                 alpha_regularizer=None,
        # #                                 alpha_constraint=None,
        # #                                 shared_axes=[1],
        # #                                 name='fc_prelu_centroid_scalar')(centroid_scalar_fc_output)

    def connect_segments(self, branches):
        """ Connect the different branches.

        :param branches: dict, branches to be concatenated
        :return:
            model output before FC layers
        """

        branches_to_concatenate = []
        for branch_name, branch in branches.items():
            # if 'conv' in len(branch_name) > 0:
            #     # Sort the hidden layers by name because the order of dictionary items is
            #     # nondeterministic between invocations of Python.
            #     time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))
            #
            #     # Concatenate the conv hidden layers.
            #     if len(time_series_hidden_layers) == 1:  # only one column
            #         cnn_branches = time_series_hidden_layers[0][1]  # how to set a name for the layer?
            #     else:  # more than one branch
            #         cnn_branches = tf.keras.layers.Concatenate(name='convbranch_concat', axis=-1)(
            #             [branch_output[1] for branch_output in time_series_hidden_layers])

            branches_to_concatenate.append(branch)

        if len(branches_to_concatenate) > 1:
            net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(branches_to_concatenate)
        else:
            net = branches_to_concatenate[0]

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
                                                self.config['decay_rate']) if self.config['decay_rate'] is not None else None,
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

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {branch_name: None for branch_name in self.config['conv_branches']}
        branches_net.update({branch_name: None for branch_name in self.config['scalar_branches']})

        # create convolutional branches
        conv_branches_net = self.build_conv_branches()
        branches_net.update(conv_branches_net)

        # create scalar branches
        scalar_branches_net = self.build_scalar_branches()
        branches_net.update(scalar_branches_net)

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


class ExoMinerParallel(object):

    def __init__(self, config, features):
        """ Initializes the CNN 1D Planet Finder Parallel model. The core structure consists of separate convolutional
        branches for non-related types of input (flux, centroid, odd and even time series, ...), except for local and
        global flux and centroid views, which are part of the same local/global branch. The respective scalar features
        are concatenated to the extracted features at the end.

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

        if 'branches' not in self.config:
            self.branches = ['global_flux_centroid_views', 'local_flux_centroid_views']
        else:
            self.branches = self.config['branches']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

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
        for branch_i, branch in enumerate(self.branches):  # create a convolutional branch

            if branch == 'local_flux_oddeven_views':  # for odd and even views, inputs are combined as two channels

                input_branch = \
                    tf.keras.layers.Concatenate(axis=2,
                                                name='input_{}'.format('local_view_oddeven'
                                                                       ))([self.inputs['local_flux_odd_view_fluxnorm'],
                                                                           self.inputs['local_flux_even_view_fluxnorm']]
                                                                          )
                input_branch = tf.keras.layers.Permute((2, 1), name='permute_oe')(input_branch)
                input_branch = tf.expand_dims(input_branch, axis=-1, name='expanding_oe')

            elif branch in ['global_flux_centroid_views', 'local_flux_centroid_views']:
                if 'global' in branch:
                    input_branch = \
                        tf.keras.layers.Concatenate(axis=2,
                                                    name='input_{}'.format('global_flux_centroid_views'
                                                                           ))([self.inputs['global_flux_view_fluxnorm'],
                                                                               self.inputs[
                                                                                   'global_centr_view_std_noclip']])
                else:
                    input_branch = \
                        tf.keras.layers.Concatenate(axis=2,
                                                    name='input_{}'.format('local_flux_centroid_views'
                                                                           ))([self.inputs['local_flux_view_fluxnorm'],
                                                                               self.inputs[
                                                                                   'local_centr_view_std_noclip']])
            else:
                input_branch = self.inputs[branch]

            # get number of conv blocks for the given view
            n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

            # get pool size for the given view
            pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

            for conv_block_i in range(n_blocks):  # create convolutional blocks

                num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters,
                          'kernel_initializer': weight_initializer,
                          'kernel_size': (1, self.config['kernel_size'])
                          if branch == 'local_flux_oddeven_views' else self.config['kernel_size'],
                          'strides': (1, self.config['kernel_stride'])
                          if branch == 'local_flux_oddeven_views' else self.config['kernel_stride'],
                          'padding': 'same'
                          }

                for seq_conv_block_i in range(self.config['conv_ls_per_block']):  # create convolutional block

                    if branch == 'local_flux_oddeven_views':
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

                if branch == 'local_flux_oddeven_views':
                    net = tf.keras.layers.MaxPooling2D(pool_size=(1, pool_size),
                                                       strides=(1, self.config['pool_stride']),
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)
                else:
                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size,
                                                       strides=self.config['pool_stride'],
                                                       name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)

            if branch == 'local_flux_oddeven_views':  # subtract extracted features for odd and even views branch
                net = tf.split(net, 2, axis=1, name='split_oe')
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)
                # net = tf.keras.layers.Permute((2, 3, 1), name='permute2_oe')(net)  # needed for Conv2D

            # flatten output of the convolutional branch
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

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

            # if self.config['non_lin_fn'] == 'lrelu':
            #     net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'relu':
            #     net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'prelu':
            #     net = tf.keras.layers.PReLU(alpha_initializer='zeros',
            #                                 alpha_regularizer=None,
            #                                 alpha_constraint=None,
            #                                 shared_axes=[1],
            #                                 name='fc_prelu_{}'.format(branch))(net)
            #
            # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)  # needed for Conv2D
            #
            # net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

            # concatenate corresponding scalar features to the flattened vector of features from the convolutional branch
            if 'local_weak_secondary_view' in branch:
                scalar_input = tf.keras.layers.Concatenate(axis=1, name='wks_scalar_input')(
                    [
                        self.inputs['tce_maxmes_norm'],
                        self.inputs['tce_albedo_stat_norm'],
                        self.inputs['tce_ptemp_stat_norm'],
                        self.inputs['wst_depth_norm'],
                        # self.inputs['tce_period_norm'],
                        # self.inputs['tce_prad_norm'],
                    ])

                net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_input
                ])

            elif 'local_flux_centroid_views' in branch:
                scalar_input = tf.keras.layers.Concatenate(axis=1, name='flux_centroid_scalar_input')(
                    [
                        self.inputs['tce_dikco_msky_norm'],
                        self.inputs['tce_dikco_msky_err_norm'],
                        self.inputs['tce_dicco_msky_norm'],
                        self.inputs['tce_dicco_msky_err_norm'],
                        self.inputs['tce_fwm_stat_norm'],
                        # self.inputs['mag_norm'],
                        self.inputs['transit_depth_norm']
                    ])

                net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_input
                ])

            elif 'alocal_flux_oddeven_views' in branch:
                scalar_input = tf.keras.layers.Concatenate(axis=1, name='oddeven_scalar_input')(
                    [
                        # self.inputs['sigma_oot_odd'],
                        # self.inputs['sigma_it_odd'],
                        # self.inputs['sigma_oot_even'],
                        # self.inputs['sigma_it_even'],
                        self.inputs['odd_se_oot'],
                        self.inputs['even_se_oot'],
                    ])

                net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_input
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

        scalar_branches = []

        if 'stellar' in self.branches:
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
                                                             name='fc_stellar_scalar')(stellar_scalar_input)

            if self.config['non_lin_fn'] == 'lrelu':
                stellar_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_stellar_scalar')(
                    stellar_scalar_fc_output)
            elif self.config['non_lin_fn'] == 'relu':
                stellar_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_stellar_scalar')(stellar_scalar_fc_output)
            elif self.config['non_lin_fn'] == 'prelu':
                stellar_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                                 alpha_regularizer=None,
                                                                 alpha_constraint=None,
                                                                 shared_axes=[1],
                                                                 name='fc_prelu_stellar_scalar')(stellar_scalar_fc_output)

            scalar_branches.append(stellar_scalar_fc_output)

        if 'dv+tce_fit' in self.branches:
            dv_scalar_input = tf.keras.layers.Concatenate(axis=1, name='dv_scalar_input')([
                self.inputs['tce_cap_stat_norm'],
                self.inputs['tce_hap_stat_norm'],
                # self.inputs['tce_cap_hap_stat_diff_norm'],
                self.inputs['tce_rb_tcount0n_norm'],
                # self.inputs['tce_rb_tcount0_norm'],
                self.inputs['boot_fap_norm'],
                self.inputs['tce_period_norm'],
                self.inputs['tce_prad_norm'],
                # self.inputs['tce_bin_oedp_stat_norm'],
            ])

            dv_scalar_fc_output = tf.keras.layers.Dense(units=4,
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
                                                            name='fc_prelu_dv_scalar')(dv_scalar_fc_output)

            scalar_branches.append(dv_scalar_fc_output)

        # centroid_scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')([
        #     self.inputs['tce_dikco_msky_norm'],
        #     self.inputs['tce_dikco_msky_err_norm'],
        #     # self.inputs['tce_dicco_msky_norm'],
        #     # self.inputs['tce_dicco_msky_err_norm'],
        # ])
        #
        # centroid_scalar_fc_output = tf.keras.layers.Dense(units=4,
        #                             kernel_regularizer=regularizers.l2(
        #                                 self.config['decay_rate']) if self.config['decay_rate'] is not None else None,
        #                             activation=None,
        #                             use_bias=True,
        #                             kernel_initializer='glorot_uniform',
        #                             bias_initializer='zeros',
        #                             bias_regularizer=None,
        #                             activity_regularizer=None,
        #                             kernel_constraint=None,
        #                             bias_constraint=None,
        #                             name='fc_centroid_scalar')(centroid_scalar_input)
        #
        # if self.config['non_lin_fn'] == 'lrelu':
        #     centroid_scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_centroid_scalar')(centroid_scalar_fc_output)
        # elif self.config['non_lin_fn'] == 'relu':
        #     centroid_scalar_fc_output = tf.keras.layers.ReLU(name='fc_relu_centroid_scalar')(centroid_scalar_fc_output)
        # elif self.config['non_lin_fn'] == 'prelu':
        #     centroid_scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
        #                                 alpha_regularizer=None,
        #                                 alpha_constraint=None,
        #                                 shared_axes=[1],
        #                                 name='fc_prelu_centroid_scalar')(centroid_scalar_fc_output)

        all_branches = [net] + scalar_branches
        net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(all_branches)

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


class MLPPlanetFinder(object):

    def __init__(self, config, features):
        """ Initializes the MLP Planet Finder model. MPL model that is fed scalar features which are diagnostics and
         statistics.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.features = features

        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(config['label_map'])
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # self.is_training = None

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def connect_segments(self):
        """ Connect the different scalar features.

        :return:
            model output before FC layers
        """

        tce_scalar_input = self.inputs['transit_depth_norm']
        # tce_scalar_input = tf.keras.layers.Concatenate(axis=1, name='tce_scalar_input')(
        #     [
        #         self.inputs['transit_depth_norm'],
        #     ])

        oddeven_scalar_input = tf.keras.layers.Concatenate(axis=1, name='oddeven_scalar_input')(
            [
                self.inputs['odd_se_oot'],
                self.inputs['even_se_oot'],
            ])

        centroid_scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')(
            [
                self.inputs['tce_dikco_msky_norm'],
                self.inputs['tce_dikco_msky_err_norm'],
                self.inputs['tce_dicco_msky_norm'],
                self.inputs['tce_dicco_msky_err_norm'],
                self.inputs['tce_fwm_stat_norm'],
                self.inputs['mag_norm'],
            ])

        wks_scalar_input = tf.keras.layers.Concatenate(axis=1, name='wks_scalar_input')(
            [
                self.inputs['tce_maxmes_norm'],
                self.inputs['tce_albedo_stat_norm'],
                self.inputs['tce_ptemp_stat_norm'],
                self.inputs['wst_depth_norm'],
                self.inputs['tce_period_norm'],
                self.inputs['tce_prad_norm'],
            ])

        stellar_scalar_input = tf.keras.layers.Concatenate(axis=1, name='stellar_scalar_input')(
            [
                self.inputs['tce_steff_norm'],
                self.inputs['tce_slogg_norm'],
                self.inputs['tce_smet_norm'],
                self.inputs['tce_sradius_norm'],
                self.inputs['tce_smass_norm'],
                self.inputs['tce_sdens_norm'],
            ])

        dv_scalar_input = tf.keras.layers.Concatenate(axis=1, name='dv_scalar_input')([
            # self.inputs['tce_cap_stat_norm'],
            # self.inputs['tce_hap_stat_norm'],
            self.inputs['tce_cap_hap_stat_diff_norm'],
            self.inputs['tce_rb_tcount0n_norm'],
            self.inputs['boot_fap_norm'],
            # self.inputs['tce_period_norm'],
            # self.inputs['tce_prad_norm']
        ])

        concat_scalar_inputs = tf.keras.layers.Concatenate(axis=1, name='all_scalar_inputs')([
            dv_scalar_input,
            stellar_scalar_input,
            wks_scalar_input,
            oddeven_scalar_input,
            tce_scalar_input,
            centroid_scalar_input,
        ])

        return concat_scalar_inputs

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

        # connect the different scalar features
        net = self.connect_segments()

        # create FC layers and output
        outputs = self.build_fc_layers(net)

        return outputs


class Astronet(object):

    def __init__(self, config, features):
        """ Initializes the Astronet model. The core structure consists of two convolutional
        branches - one for the global view flux time series and another one for the local flux view.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_flux_view_fluxnorm', 'local_flux_view_fluxnorm']

        # self.is_training = None

        # # class-label weights for weighted loss
        # # convert from numpy to tensor
        # self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)


    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        # weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
        #     else 'glorot_uniform'

        cnn_layers = {}
        for branch_i, branch in enumerate(self.branches):

            input_branch = self.inputs[branch]

            # with tf.variable_scope('ConvNet_%s' % view):

            # get number of conv blocks for the given view
            # n_blocks = 2 if 'local' in branch else 5
            n_blocks = self.config['num_loc_conv_blocks'] if 'local' in branch else self.config['num_glob_conv_blocks']

            # get pool size for the given view
            # pool_size = 7 if 'local' in branch else 5
            pool_size = self.config['pool_size_loc'] if 'local' in branch else self.config['pool_size_glob']

            # number of filters in each convolutional block
            # num_filters = [16, 32, 64, 128, 256] if 'global' in branch else [16, 32]
            num_filters = [2 ** (self.config['init_conv_filters'] + conv_block_i) for conv_block_i in range(n_blocks)]

            for conv_block_i in range(n_blocks):

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters[conv_block_i],
                          # 'kernel_initializer': weight_initializer,
                          'kernel_size': self.config['kernel_size'],
                          'strides': 1,
                          'padding': "same"}

                for seq_conv_block_i in range(self.config['conv_ls_per_block']):
                    net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                 activation=None,
                                                 use_bias=True,
                                                 kernel_initializer='glorot_uniform',
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

                net = tf.keras.layers.MaxPooling1D(pool_size=pool_size,
                                                   strides=2,
                                                   name='maxpooling{}{}'.format(branch, conv_block_i))(net)

            # Flatten
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

            cnn_layers[branch] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example).

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

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers.

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']  # 512

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

            net = tf.keras.layers.ReLU()(net)

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

    def __init__(self, config, features):
        """ Initializes the Exonet model. The core structure consists of two convolutional
        branches - one for the global view time series and another one for the local view ones.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_view', 'local_view']

        # self.is_training = None

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)


    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv branches
        """

        # weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
        #     else 'glorot_uniform'

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
            # n_blocks = 2 if 'local' in branch else 5
            n_blocks = self.config['num_loc_conv_blocks'] if 'local' in branch else self.config['num_glob_conv_blocks']

            # get pool size for the given view
            # pool_size = 7 if 'local' in branch else 5
            pool_size = self.config['pool_size_loc'] if 'local' in branch else self.config['pool_size_glob']

            # number of filters in each convolutional block
            # num_filters = [16, 32] if 'local' in branch else [16, 32, 64, 128, 256]
            num_filters = [2 ** (self.config['init_conv_filters'] + conv_block_i) for conv_block_i in range(n_blocks)]

            for conv_block_i in range(n_blocks):

                # set convolution layer parameters from config
                kwargs = {'filters': num_filters[conv_block_i],
                          'kernel_initializer': 'glorot_uniform',
                          'kernel_size': self.config['kernel_size'],
                          'strides': 1,
                          'padding': "same"}

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

                    net = tf.keras.layers.ReLU()(net)

                net = tf.keras.layers.MaxPooling1D(
                    pool_size=self.config['pool_size_loc'] if 'local' in branch else self.config['pool_size_glob'],
                    strides=2,
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

        # concatenate stellar parameters
        pre_logits_concat = tf.keras.layers.Concatenate(axis=1, name='pre_logits_concat_scalar_params')(
            [
                pre_logits_concat,
                self.inputs['tce_steff_norm'],
                self.inputs['tce_slogg_norm'],
                self.inputs['tce_smet_norm'],
                self.inputs['tce_sradius_norm'],
                self.inputs['tce_smass_norm'],
                self.inputs['tce_sdens_norm'],
            ])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers.

        :param net: model upstream the FC layers
        :return:
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config['num_fc_layers']):

            fc_neurons = self.config['init_fc_neurons']  # 512

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

            net = tf.keras.layers.ReLU()(net)

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
        """

        # model configuration (parameters and hyperparameters)
        self.config = config['config']
        self.features = features

        # binary classification with sigmoid output layer
        self.output_size = 1

        # global and local view branches
        self.branches = ['global_view', 'local_view']

        # self.is_training = None

        self.inputs = create_inputs(self.features, config['feature_map'])

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

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

        # concatenate stellar parameters
        pre_logits_concat = tf.keras.layers.Concatenate(axis=1, name='pre_logits_concat_scalar_params')(
            [
                pre_logits_concat,
                self.inputs['tce_steff_norm'],
                self.inputs['tce_slogg_norm'],
                self.inputs['tce_smet_norm'],
                self.inputs['tce_sradius_norm'],
                self.inputs['tce_smass_norm'],
                self.inputs['tce_sdens_norm'],
            ])

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


def create_inputs(features, feature_map=None):
    """ Create input layers for the input features.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :param feature_map: maps features' names to features names expected by the model
    :return:
        inputs: dictionary, each key-value pair is a feature_name: feature
    """

    if feature_map is None:
        feature_map = {}

    inputs = {}
    for feature in features:
        if feature not in feature_map:
            inputs[feature] = tf.keras.Input(shape=features[feature]['dim'],
                                             batch_size=None,
                                             name=feature,
                                             dtype=features[feature]['dtype'],
                                             sparse=False,
                                             tensor=None,
                                             ragged=False)
        else:
            inputs[feature_map[feature]] = tf.keras.Input(shape=features[feature]['dim'],
                                                          batch_size=None,
                                                          name=feature_map[feature],
                                                          dtype=features[feature]['dtype'],
                                                          sparse=False,
                                                          tensor=None,
                                                          ragged=False)

    return inputs


def create_ensemble(features, models, feature_map=None):
    """ Create a Keras ensemble.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :param models: list, list of Keras models
    :param feature_map: maps features' names to features names expected by the model

    :return:
        Keras average ensemble
    """

    inputs = create_inputs(features=features, feature_map=feature_map)

    single_models_outputs = [model(inputs) for model in models]

    if len(single_models_outputs) == 1:
        outputs = single_models_outputs
    else:
        outputs = tf.keras.layers.Average()(single_models_outputs)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_model(model, config, metrics_list):
    """ Compile model.

    :param model: Keras model
    :param config: dict, configuration parameters
    :param metrics_list: list, monitored metrics
    :return:
        compiled model
    """

    # set loss
    if config['config']['multi_class']:  # multiclass
        # model_loss = losses.SparseCategoricalCrossentropy(from_logits=False, name='sparse_categorical_crossentropy')
        model_loss = losses.CategoricalCrossentropy(from_logits=False, name='categorical_crossentropy')

    else:
        model_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='binary_crossentropy')

    # set optimizer
    if config['config']['optimizer'] == 'Adam':
        model_optimizer = optimizers.Adam(learning_rate=config['config']['lr'],
                                          beta_1=0.9,
                                          beta_2=0.999,
                                          epsilon=1e-8,
                                          amsgrad=False,
                                          name='Adam')
    else:  # SGD
        model_optimizer = optimizers.SGD(learning_rate=config['config']['lr'],
                                         momentum=config['config']['sgd_momentum'],
                                         nesterov=False,
                                         name='SGD')

    # compile model with chosen optimizer, loss and monitored metrics
    model.compile(optimizer=model_optimizer, loss=model_loss, metrics=metrics_list)

    return model


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


class TransformerExoMiner(object):

    def __init__(self, config, features):
        """ Initializes the Transformer ExoMiner. The core structure consists of one or more transformer branches that
        are fed as input a representation of the unfolded flux time series. This representation consists of a fixed set
        of phases (aka cycles) for different views (global, local). Other branches such as convolutional and scalar can
        be added.

        :param config: dict, model configuration for its parameters and hyperparameters
        :param features: dict, 'feature_name' : {'dim': tuple, 'dtype': (tf.int, tf.float, ...)}
        """

        # model configuration (parameters and hyper-parameters)
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

    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, name):
        """ Implements encoder-only transformer.

        :param inputs:
        :param head_size:
        :param num_heads:
        :ff_dim:
        dropout:
        name:
        :return:

        """

        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                               name=f'{name}-layer_normalization-1')(inputs)

        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            name=f'{name}-multi_head_attention'
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                               name=f'{name}-multi_normalization-2')(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu",
                                   name=f'{name}-conv1d-1')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1,
                                   name=f'{name}-conv1d-2')(x)
        return x + res

    def build_transformer_layers(self):
        """ Builds the transformer branches.

        :return:
            transformer_branches, dict with the different transformer branches
        """

        transformer_branches = {branch_name: None for branch_name in self.config['transformer_branches']}
        for branch_i, branch in enumerate(self.config['transformer_branches']):  # create a transformer branch

            branch_view_inputs = self.inputs[self.config['transformer_branches'][branch][0]]

            net = branch_view_inputs

            if self.config['time_encoding']:
                time2vec = Time2Vec(kernel_size=1)
                time_embedding = keras.layers.TimeDistributed(time2vec)(net)
                net = K.concatenate([net, time_embedding], -1)

            # generate lstm blocks
            for transformer_block_i in range(self.config['num_transformer_blocks']):
                net = self.transformer_encoder(net,
                                               self.config['head_size'],
                                               self.config['num_heads'],
                                               self.config['ff_dim'],
                                               self.config['dropout_rate_transformer'],
                                               f'transformer_{branch}_{transformer_block_i}')

            # channels last makes shape [301]
            # channels first makes shape [20]
            if self.config['transformer_output'] == 'bin_average_pooling':
                net = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(net)
            elif self.config['transformer_output'] == 'phase_average_pooling':
                net = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(net)
            elif self.config['transformer_output'] == 'flat':
                net = tf.reshape(net, (-1, net.shape[1] * net.shape[2]))

            if 'global' in branch:
                self.config['transformer_num_fc_layers'] = self.config['global-num_units_transformer_fc_layers']
            elif 'local' in branch:
                self.config['transformer_num_fc_layers'] = self.config['local-num_units_transformer_fc_layers']

            for fc_layer_i, num_units in enumerate(self.config[f'transformer_num_fc_layers']):
                # add FC layer that extracts features from the combined feature vector of features from the lstm
                # branch (flattened) and corresponding scalar features
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
                                            name=f'fc_{branch}_{fc_layer_i}')(net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1],
                                                name=f'fc_prelu_{branch}_{fc_layer_i}')(net)

                net = tf.keras.layers.Dropout(self.config['dropout_rate_trans_fc'])(net)

            transformer_branches[branch] = net

        return transformer_branches

    def build_conv_branches(self):
        """ Builds the convolutional branches.

        :return:
            conv_branches, dict with the different convolutional branches
        """

        odd_even_branch_name = 'local_odd_even'
        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'},
                         'kernel_size': {'global_view': 'kernel_size_glob', 'local_view': 'kernel_size_loc'},
                         # 'kernel_size': {'global_view': 'kernel_size', 'local_view': 'kernel_size'},
                         }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in self.config['conv_branches']}
        for branch_i, branch in enumerate(self.config['conv_branches']):  # create a convolutional branch

            branch_view_inputs = [self.inputs[view_name] for view_name in self.config['conv_branches'][branch]['views']]
            # branch_view_inputs = branch_view_inputs if len(branch_view_inputs) > 1 else branch_view_inputs[0]
            if len(branch_view_inputs) == 2:
                branch_view_inputs = tf.keras.layers.Concatenate(axis=2, name='input_{}'.format(branch))(
                    branch_view_inputs)
                if branch == odd_even_branch_name:  # for odd and even views, inputs are combined as two channels
                    branch_view_inputs = tf.keras.layers.Permute((2, 1), name='permute_oe')(branch_view_inputs)
                    branch_view_inputs = tf.expand_dims(branch_view_inputs, axis=-1, name='expanding_oe')
            else:
                branch_view_inputs = branch_view_inputs[0]

            # get number of conv blocks for the given view
            n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

            kernel_size = self.config[config_mapper['kernel_size'][('local_view', 'global_view')['global' in branch]]]

            # get pool size for the given view
            pool_size = self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

            for conv_block_i in range(n_blocks):  # create convolutional blocks

                num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': (1, kernel_size)  # (1, self.config['kernel_size'])
                               if branch == odd_even_branch_name else kernel_size,  # self.config['kernel_size'],
                               'strides': (1, self.config['kernel_stride'])
                               if branch == odd_even_branch_name else self.config['kernel_stride'],
                               'padding': 'same'
                               }

                for seq_conv_block_i in range(self.config['conv_ls_per_block']):  # create convolutional block

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
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
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
                                                     name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
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
                net = tf.split(net, 2, axis=1, name='split_oe')
                net = tf.keras.layers.Subtract(name='subtract_oe')(net)
                # net = tf.keras.layers.Permute((2, 3, 1), name='permute2_oe')(net)  # needed for Conv2D

            # flatten output of the convolutional branch
            net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

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

            # if self.config['non_lin_fn'] == 'lrelu':
            #     net = tf.keras.layers.LeakyReLU(alpha=0.01, name='fc_lrelu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'relu':
            #     net = tf.keras.layers.ReLU(name='fc_relu_{}'.format(branch))(net)
            # elif self.config['non_lin_fn'] == 'prelu':
            #     net = tf.keras.layers.PReLU(alpha_initializer='zeros',
            #                                 alpha_regularizer=None,
            #                                 alpha_constraint=None,
            #                                 shared_axes=[1],
            #                                 name='fc_prelu_{}'.format(branch))(net)
            #
            # net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)  # needed for Conv2D
            #
            # net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

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

            # # concatenate corresponding scalar features to the flattened vector of features from the convolutional branch
            # if 'local_weak_secondary_view' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='wks_scalar_input')(
            #         [
            #             self.inputs['tce_maxmes_norm'],
            #             self.inputs['tce_albedo_stat_norm'],
            #             self.inputs['tce_ptemp_stat_norm'],
            #             self.inputs['wst_depth_norm'],
            #             # self.inputs['tce_period_norm'],
            #             # self.inputs['tce_prad_norm'],
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_centr_view' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='centroid_scalar_input')(
            #         [
            #             self.inputs['tce_dikco_msky_norm'],
            #             self.inputs['tce_dikco_msky_err_norm'],
            #             self.inputs['tce_dicco_msky_norm'],
            #             self.inputs['tce_dicco_msky_err_norm'],
            #             self.inputs['tce_fwm_stat_norm'],
            #             self.inputs['ruwe_norm'],
            #             # self.inputs['mag_norm'],
            #             # self.inputs['mag_cat'],
            #             # tf.cast(self.inputs['mag_cat'], tf.float32),
            #             self.inputs['mag_cat_norm']
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_flux_view' in branch:
            #     # scalar_input = self.inputs['transit_depth_norm']
            #
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='flux_scalar_input')(
            #         [
            #             self.inputs['transit_depth_norm'],
            #             self.inputs['tce_max_mult_ev_norm'],
            #             self.inputs['tce_max_sngle_ev_norm'],
            #             self.inputs['tce_robstat_norm'],
            #             self.inputs['tce_model_chisq_norm']
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])
            #
            # elif 'local_flux_oddeven_views' in branch:
            #     scalar_input = tf.keras.layers.Concatenate(axis=1, name='oddeven_scalar_input')(
            #         [
            #             # self.inputs['sigma_oot_odd'],
            #             # self.inputs['sigma_it_odd'],
            #             # self.inputs['sigma_oot_even'],
            #             # self.inputs['sigma_it_even'],
            #             self.inputs['odd_se_oot_norm'],
            #             self.inputs['even_se_oot_norm'],
            #         ])
            #
            #     net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_{}'.format(branch))([
            #         net,
            #         scalar_input
            #     ])

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

            conv_branches[branch] = net

        return conv_branches

    def build_diff_img_branch(self):
        """ Builds the difference image branch.

        :return: dict with the difference image branch
        """

        # config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
        #                  'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'},
        #                  'kernel_size': {'global_view': 'kernel_size_glob', 'local_view': 'kernel_size_loc'},
        #                  }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        # branch_view_inputs = [self.inputs[view_name] for view_name in self.config['diff_img_branch']['imgs']]
        branch_view_inputs = [self.inputs['diff_img'], self.inputs['diff_img']]
        branch_view_inputs = [tf.expand_dims(l, axis=-1, name='expanding_diff_img') for l in branch_view_inputs]

        branch_view_inputs = tf.keras.layers.Concatenate(axis=4, name='input_diff_img_concat')(branch_view_inputs)

        # self.config.update(
        #     {
        #         'blocks_diff_img': 3,
        #         'kernel_size_diff_img': 3,
        #         'pool_size_diff_img'
        #         # 'kernel_size_fc':
        #     }
        # )

        # get number of conv blocks for the given view
        n_blocks = 1  # self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

        kernel_size = (3, 3, 1)  # self.config[config_mapper['kernel_size'][('local_view', 'global_view')['global' in branch]]]

        # get pool size for the given view
        pool_size = (2, 2, 1)  # self.config[config_mapper['pool_size'][('local_view', 'global_view')['global' in branch]]]

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (2 + conv_block_i)
            # num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': kernel_size,  # self.config['kernel_size'],
                           'strides': 1,  # (1, self.config['kernel_stride'])
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
                                                   name='maxpooling_diff_img_{}'.format(conv_block_i))(net)

        # flatten output of the convolutional branch
        net = tf.keras.layers.Permute((1, 2, 4, 3), name='permute_diff_img')(net)
        net = tf.keras.layers.Reshape((np.prod(net.shape[1:-1].as_list()), net.shape[-1]))(net)

        # add scalar features
        if self.config['diff_img_branch']['scalars'] is not None:
            scalar_inputs = [self.inputs[feature_name] for feature_name in self.config['diff_img_branch']['scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'diff_img_scalar_input')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]
            net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_diff_img')([
                net,
                scalar_inputs
            ])

        # 2D convolution with kernel size equal to feature map size
        net = tf.expand_dims(net, axis=-1, name='expanding_input_convfc')
        net = tf.keras.layers.Conv2D(filters=4,  # self.config['num_fc_conv_units'],
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

            net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

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
                                                self.config['decay_rate']) if self.config['decay_rate'] is not None else None,
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

        return net

    def build(self):
        """ Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {}

        if self.config['transformer_branches'] is not None:
            branches_net.update(self.build_transformer_layers())

        if self.config['scalar_branches'] is not None:
            branches_net.update(self.build_scalar_branches())

        if self.config['conv_branches'] is not None:
            branches_net.update(self.build_conv_branches())

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

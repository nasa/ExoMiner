""" Implementation of models using Keras functional API. """

# 3rd party
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import tensorflow.keras.backend as K

# local
from models.utils_models import create_inputs
from models.custom_layers import Time2Vec, MeanAttentionNormalization, SplitLayer, ReduceSumLayer, StdLayer, IdentityConv1DInitializer, IdentityConv2DInitializer

def attention_block(query, key_value, num_heads=2, key_dim=32, dropout_rate=0.1, return_attention=False):
    """ Create attention block.

        Args:
             :param query: TF Keras tensor, query [batch_size, n_tokens, n_features]
             :param key_value: TF Keras tensor, value/key [batch_size, n_tokens, n_features]
             :param num_heads: int, number of attention heads
             :param key_dim: int, dimensionality of learned query and value vectors
             :param dropout_rate: float, dropout rate
             :param return_attention: bool, if True returns also the attention scores

        Returns: output TF Keras tensor from attention block; if return_attention is True, also returns the mean attention scores
     """

    # apply multi-head self-attention
    if return_attention:
        attn_output, attn_scores = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                        key_dim=key_dim,
                                                        name='att_multihead')(query, key_value, return_attention_scores=True)
    else:
        attn_output, attn_scores = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                        key_dim=key_dim,
                                                        name='att_multihead')(query, key_value, return_attention_scores=True)
        
    attn_output = tf.keras.layers.Dropout(dropout_rate, name='att_dropout')(attn_output)
    # add residual
    out1 = tf.keras.layers.Add(name='att_skip_add_residual')([query, attn_output])
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='att_layer_norm')(out1)

    # add dense layer and non-linear activity function
    ffn_output = tf.keras.layers.Dense(query.shape[-1], activation='relu', name='att_dense')(out1)
    ffn_output = tf.keras.layers.Dropout(dropout_rate, name='att_dropout_dense')(ffn_output)
    ffn_output = tf.keras.layers.Add(name='att_skip_add_residual_dense')([out1, ffn_output])
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='att_layer_norm_dense')(ffn_output)

    out2 = tf.keras.layers.Reshape((np.prod(out2.shape[1:]),), name='cross_att_output_reshape')(out2)
    # out2 = tf.keras.layers.GlobalAveragePooling1D(name='cross_att_global_average_pool')(out2)
    
    if return_attention:  # compute mean attention scores over all heads
        # mean_attn = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='mean_attn')(attn_scores)
        # mean_attn_normalized = Lambda(lambda x: x / tf.reduce_sum(x, axis=-1, keepdims=True), name='normalized_mean_attn')(mean_attn)
        mean_attn_normalized = MeanAttentionNormalization(name='normalized_mean_attn')(attn_scores)
    else:
        mean_attn_normalized = None

    return out2, mean_attn_normalized


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

    # def build_scalar_inputs(self):
    #     """ Builds the scalar input features.
    #
    #     :return:
    #         scalar_input, TF keras tensor with the different scalar input features
    #     """
    #
    #
    #     scalar_inputs = [self.inputs[feature_name] for feature_name in
    #                      self.config['scalar_branches']['scalar_branch']]
    #     if len(scalar_inputs) > 1:
    #         scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'scalar_inputs_concat')(scalar_inputs)
    #     else:
    #         scalar_inputs = scalar_inputs[0]
    #
    #     return scalar_inputs

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

        Returns: TF Keras tensor with flattened unfolded local flux extracted features
     """

    # global pooling similar to other local fluxes
    # global_avg_pool_layer = tf.keras.layers.GlobalAveragePooling1D(name=f'local_fluxes_unfolded_flux_global_avg_pooling')
    flatten_layer = tf.keras.layers.Flatten(name=f'local_fluxes_unfolded_flux_flatten')
    unfolded_flux_extracted_features = tf.keras.layers.TimeDistributed(flatten_layer, name=f'local_fluxes_unfolded_flux_flatten_apply')(unfolded_flux_extracted_features)
    
    # split layer in preparation to avg/min/max
    unfolded_flux_extracted_features_split = SplitLayer(unfolded_flux_extracted_features.shape[1],
                     axis=1,
                     name='unfolded_flux_split')(unfolded_flux_extracted_features)

    # get stats for all extracted features of each flux phase
    merge_layer_avg = tf.keras.layers.Average(name='local_fluxes_unfolded_flux_avg')(unfolded_flux_extracted_features_split)
    merge_layer_min = tf.keras.layers.Minimum(name='local_fluxes_unfolded_flux_min')(unfolded_flux_extracted_features_split)
    merge_layer_max = tf.keras.layers.Maximum(name='local_fluxes_unfolded_flux_max')(unfolded_flux_extracted_features_split)
    # merge_layer_std = StdLayer(axis=1, name='unfolded_flux_std')(net)

    # concat 3 different layers
    unfolded_flux_extracted_features_merge_stats = (
        tf.keras.layers.Concatenate(axis=1, name='local_fluxes_unfolded_flux_concat_stats')([
        merge_layer_min,
        merge_layer_max,
        merge_layer_avg,
        # merge_layer_std
    ]))

    # set the 3 layers to be the channels
    input_conv = tf.keras.layers.Permute((2, 1),
                                  name='local_fluxes_unfolded_flux_permute_stats')(unfolded_flux_extracted_features_merge_stats)

    # convolve features of the different channels
    conv_kwargs = {'filters': num_filters_conv_stats,
                   'kernel_initializer': weight_initializer,
                   'kernel_size': kernel_size_conv_stats,
                   }
    output_conv = tf.keras.layers.Conv1D(
        dilation_rate=1,
        activation=None,
        use_bias=True,
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='local_fluxes_unfolded_flux_2conv',
        **conv_kwargs)(input_conv)
    
    flattened_output = tf.keras.layers.Flatten(name='local_flux_unfolded_flux_flatten')(output_conv)

    return flattened_output


def difference_img_quality_metric_attention_fusion(diffimg_features, quality_metrics, name_prefix="diff_imgs"):
    """ Applies attention-based fusion to a set of features from difference image using quality metric values.

    Args:
        diffimg_features: Tensor of shape (batch_size, num_sets, feature_dim) with difference image features for a set of images
        quality_metrics: Tensor of shape (batch_size, num_sets, 1) with the corresponding quality metric values
        name_prefix: Optional prefix for layer naming

    Returns:
        Tensor of shape (batch_size, feature_dim) — the fused feature vector
    """

    # # Optional: project quality scores to attention logits
    # attention_logits = tf.keras.layers.Dense(
    #     1,
    #     activation=None,
    #     name=f"{name_prefix}_attention_logits"
    # )(quality)  # shape: (batch, num_sets, 1)

    # # Normalize to get attention weights
    # attention_weights = tf.keras.layers.Softmax(axis=1, name=f"{name_prefix}_attention_weights")(quality_metrics)

    # Apply attention weights to features
    weighted_features = tf.keras.layers.Multiply(name=f"{name_prefix}_weighted_features")([diffimg_features, quality_metrics])

    # Sum across the sets to get a single feature vector
    # fused_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name=f"{name_prefix}_fused_output")(weighted_features)
    fused_output = ReduceSumLayer(name=f"{name_prefix}_fused_output")(weighted_features)

    return fused_output


class ExoMinerJointLocalFlux(object):
    """ExoMiner architecture in progress. """

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
        
        self.attn_scores = None  # placeholder for attention scores if needed

        if self.config['multi_class'] or \
                (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = len(np.unique(list(config['label_map'].values())))
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        self.inputs = create_inputs(self.features, config['feature_map'])

        self.outputs = self.build()
        
        # build the model
        if self.config['use_attention_scores']:  # currently being used for XAI studies
            
            # builds model that outputs both prediction and attention scores for each example
            self.kerasModel = keras.Model(inputs=self.inputs, outputs=[self.outputs, self.attn_scores])
            
        else:
            # builds model that outputs only prediction for each example
            self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)


    def prepare_joint_local_flux_inputs(self, conv_branches):
        """ Prepares local flux inputs to be processed by the same convolutional branch.

            Args:
                :param conv_branches: list, names of convolutional branches whose features should be processed together

            Returns: TF Keras tensor with concatenated local flux input features
         """

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

        # prepare inputs before concatenating them
        local_transit_views_inputs = []
        for local_transit_feature in local_transit_features:

            if 'unfolded' in local_transit_feature[0]:

                view_inputs = [tf.keras.layers.Reshape(self.inputs[view_name].shape[1:] + (1,),
                                                       name=f'local_fluxes_expanding_{view_name}_dim')(
                    self.inputs[view_name]) for view_name in local_transit_feature]
            else:
                view_inputs = [tf.keras.layers.Reshape((1,) + self.inputs[view_name].shape[1:],
                                                       name=f'local_fluxes_expanding_{view_name}_dim')(
                    self.inputs[view_name]) for view_name in local_transit_feature]

            if len(view_inputs) > 1:  # when adding var time series
                view_inputs = tf.keras.layers.Concatenate(axis=-1,
                                                          name=f'local_fluxes_concat_{local_transit_feature[0]}_'
                                                               f'with_var')(view_inputs)
            else:
                view_inputs = view_inputs[0]

            local_transit_views_inputs.append(view_inputs)

        # combine the local transits to put through conv block (dim = [view, bins, avg/var view])
        if len(local_transit_views_inputs) > 1:
            branch_view_inputs = tf.keras.layers.Concatenate(axis=1,
                                                             name='local_fluxes_concat_local_views')(
                local_transit_views_inputs)
        else:  # only one local flux
            branch_view_inputs = local_transit_views_inputs[0]

        return branch_view_inputs

    def build_conv_branches(self):
        """ Builds convolutional branches.

        :return:
            conv_branches, dict with the different convolutional branches
        """

        conv_branch_selected = [
            # global branches
            'global_flux',
            'flux_trend',

            # local branches
            'local_centroid',
            # 'momentum_dump',

            # periodogram branch
            'flux_periodogram',
        ]

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in conv_branch_selected
                         if branch_name in self.config['conv_branches']}
        if len(conv_branches) == 0:
            return {}

        for branch in conv_branches:  # create a convolutional branch

            branch_view_inputs = [self.inputs[view_name] for view_name in self.config['conv_branches'][branch]['views']]

            # add var time series
            if len(branch_view_inputs) > 1:
                branch_view_inputs = tf.keras.layers.Concatenate(axis=2, name=f'{branch}_input')(branch_view_inputs)
            else:
                branch_view_inputs = branch_view_inputs[0]

            # get init parameters for the given view
            if branch == 'flux_trend':
                branch_name_config = 'global_flux'  # uses same hyperparameters as global flux branch
            elif branch == 'local_centroid':
                branch_name_config = 'local_fluxes'  # uses same hyperparameters as local centroid branch
            else:
                branch_name_config = branch
            n_blocks = self.config[f'{branch_name_config}_num_conv_blocks']
            kernel_size = self.config[f'{branch_name_config}_kernel_size']
            kernel_stride = self.config[f'{branch_name_config}_kernel_stride']
            pool_size = self.config[f'{branch_name_config}_pool_size']
            pool_stride = self.config[f'{branch_name_config}_pool_stride']
            conv_ls_per_block = self.config[f'{branch_name_config}_num_conv_ls_per_block']
            init_conv_filters = self.config[f'{branch_name_config}_init_power_num_conv_filters']
            
            pool_kwargs = {
                    'pool_size': pool_size,
                    'strides': pool_stride
                }

            # create convolutional branches
            for conv_block_i in range(n_blocks):  # create convolutional blocks

                input_conv_block = branch_view_inputs if conv_block_i == 0 else net

                num_filters = 2 ** (init_conv_filters + conv_block_i)

                for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional block

                    # set convolution layer parameters from config
                    conv_kwargs = {'filters': num_filters,
                                   'kernel_initializer': weight_initializer,
                                   'kernel_size': kernel_size,
                                   'strides': kernel_stride if seq_conv_block_i == 0 else 1,
                                   'padding': 'same'  # 'valid' if seq_conv_block_i == 0 else 'same'
                                   }

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

                    if self.config['batch_norm_after_conv_layers']:
                    # if seq_conv_block_i == conv_ls_per_block - 1:
                        net = tf.keras.layers.BatchNormalization(
                            axis=-1,
                            momentum=0.99,
                            epsilon=0.001,
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
                            synchronized=False,
                            name=f'{branch}_conv_{conv_block_i}_{seq_conv_block_i}_batch_norm'
                        )(net)

                    if self.config['use_attention_after_conv_layers']:
                        net = tf.keras.layers.Attention(
                            use_scale=True,
                            score_mode='dot',
                            dropout=0.0,
                            seed=None,
                            name=f'{branch}_self-attention_{conv_block_i}_{seq_conv_block_i}')([net, net])

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

                    # add skip connection of branch inputs
                    if self.config['use_skip_connection_conv_block']:
                        if seq_conv_block_i == conv_ls_per_block - 1:
                            # apply conv to have same number of channels as extracted feature map
                            if input_conv_block.shape != net.shape:
                                input_conv_block = tf.keras.layers.Conv1D(
                                    filters=net.shape[-1],
                                    kernel_size=kernel_size,
                                    padding='valid',
                                    strides=kernel_stride,
                                    dilation_rate=1,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=IdentityConv1DInitializer(),
                                    name=f'{branch}_conv{conv_block_i}_input',
                                )(input_conv_block)
                            net = tf.keras.layers.Add(
                                name=f'{branch}_skip_connection_{conv_block_i}')([net, input_conv_block])

                # if conv_block_i != n_blocks - 1:  # do not add maxpooling layer before global maxpooling layer
                #     net = tf.keras.layers.MaxPooling1D(**pool_kwargs,
                #                                        name='{}_maxpooling_{}'.format(branch, conv_block_i))(net)
                net = tf.keras.layers.MaxPooling1D(**pool_kwargs,
                                                   name=f'{branch}_maxpooling_{conv_block_i}')(net)

            # net = tf.keras.layers.GlobalAveragePooling1D(name=f'{branch}_global_max_pooling')(net)
            net = tf.keras.layers.Flatten(name=f'{branch}_flatten')(net)

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

                # scalar_inputs = tf.keras.layers.BatchNormalization(name=f'{branch}_scalar_inputs_batch_norm')(scalar_inputs)
                
                net = tf.keras.layers.Concatenate(axis=1, name='{}_wscalar'.format(branch))([
                    net,
                    scalar_inputs
                ])

            # add FC layer that extracts features from the combined feature vector of features from the convolutional
            # branch (flattened) and corresponding scalar features
            if self.config['branch_num_fc_units'] > 0:
                net = tf.keras.layers.Dense(units=self.config['branch_num_fc_units'],
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
                
                # net = tf.keras.layers.BatchNormalization(name=f'{branch}_fc_batch_norm')(net)

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

                # net = tf.keras.layers.Dropout(self.config['branch_dropout_rate_fc'],
                                            #   name=f'{branch}_dropout_fc_conv')(net)

            # net = tf.keras.layers.LayerNormalization(
            # axis=-1,
            # epsilon=1e-3,
            # center=True,
            # scale=True,
            # beta_initializer="zeros",
            # gamma_initializer="ones",
            # beta_regularizer=None,
            # gamma_regularizer=None,
            # beta_constraint=None,
            # gamma_constraint=None,
            # name=f'{branch}_layer_norm')(net)
            
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

        def joint_local_conv_block(net, conv_kwargs, conv_block_i, seq_conv_block_i):
            
            conv1d_layer = tf.keras.layers.Conv1D(
                    dilation_rate=1,
                    activation=None,
                    use_bias=True,
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name=f'local_fluxes_conv{conv_block_i}_{seq_conv_block_i}',
                    **conv_kwargs
                    )
            
            net = tf.keras.layers.TimeDistributed(conv1d_layer, name=f'local_fluxes_conv{conv_block_i}_{seq_conv_block_i}_apply')(net)
            
            if self.config['batch_norm_after_conv_layers']:
                batchnorm_layer = tf.keras.layers.BatchNormalization(
                    axis=-1,
                    momentum=0.99,
                    epsilon=0.001,
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
                    synchronized=False,
                    name=f'local_fluxes_{conv_block_i}_{seq_conv_block_i}_batch_norm'
                    )
                
                net = tf.keras.layers.TimeDistributed(batchnorm_layer, name=f'local_fluxes_{conv_block_i}_{seq_conv_block_i}_batch_norm_apply')(net)
            
            if self.config['use_attention_after_conv_layers']:
                attention_layer = tf.keras.layers.Attention(
                    use_scale=True,
                    score_mode='dot',
                    dropout=0.0,
                    seed=None,
                    name=f'local_fluxes_{conv_block_i}_{seq_conv_block_i}_self-attention')
                
                net = tf.keras.layers.TimeDistributed(attention_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_self-attention_apply')([net, net])

            if self.config['non_lin_fn'] == 'lrelu':
                activation_layer = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                name='local_fluxes_lrelu_{}_{}'.format(conv_block_i,
                                                                                    seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'local_fluxes__conv{conv_block_i}_{seq_conv_block_i}_lrelu_apply')(net)
            elif self.config['non_lin_fn'] == 'relu':
                activation_layer = tf.keras.layers.ReLU(name='local_fluxes_relu_{}_{}'.format(conv_block_i,
                                                                            seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'local_fluxes_conv{conv_block_i}_{seq_conv_block_i}_relu_apply')(net)
            elif self.config['non_lin_fn'] == 'prelu':
                activation_layer = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1, 2],
                                            name='local_fluxes__{}_{}_prelu'.format(conv_block_i,
                                                                                seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'local_fluxes_conv{conv_block_i}_{seq_conv_block_i}_prelu_apply')(net)
        
            return net
        
        # list of branches to be combined + their sizes
        local_transit_branches = [
            'local_odd_even',
            'local_flux',
            'local_weak_secondary',
            'local_unfolded_flux',
        ]
        # set expected sizes of local fluxes after joint convolution processing
        local_transit_sz = {
            'local_odd_even': 2,
            'local_flux': 1,
            'local_weak_secondary': 1,
            'local_unfolded_flux': self.inputs['unfolded_local_flux_view_fluxnorm'].shape[1],
        }
        odd_even_branch_name = 'local_odd_even'  # specific to odd and even branch

        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        conv_branches = {branch_name: None for branch_name in local_transit_branches
                         if branch_name in self.config['conv_branches']}
        if len(conv_branches) == 0:
            return {}

        branch_view_inputs = self.prepare_joint_local_flux_inputs(conv_branches)
        
        # convolve inputs with convolutional blocks
        # get init parameters for the given view
        n_blocks = self.config['local_fluxes_num_conv_blocks']
        kernel_size = self.config['local_fluxes_kernel_size']
        kernel_stride = self.config['local_fluxes_kernel_stride']
        pool_stride = self.config['local_fluxes_pool_stride']
        pool_size = self.config['local_fluxes_pool_size']
        conv_ls_per_block = self.config['local_fluxes_num_conv_ls_per_block']
        init_conv_filters = self.config['local_fluxes_init_power_num_conv_filters']

        pool_kwargs = {
                'pool_size': pool_size,
                'strides': pool_stride
            }
            
        for conv_block_i in range(n_blocks):  # create convolutional blocks

            input_conv_block = branch_view_inputs if conv_block_i == 0 else net

            num_filters = 2 ** (init_conv_filters + conv_block_i)

            for seq_conv_block_i in range(conv_ls_per_block):  # create convolutional block

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': kernel_size,
                               'strides': kernel_stride if seq_conv_block_i == 0 else 1,
                               'padding': 'same'  # 'valid' if seq_conv_block_i == 0 else 'same'
                               }

                net = joint_local_conv_block(branch_view_inputs if conv_block_i == 0 and seq_conv_block_i == 0 else net, conv_kwargs, conv_block_i, seq_conv_block_i)
                
                # add skip connection of branch inputs
                if self.config['use_skip_connection_conv_block']:

                    if seq_conv_block_i == conv_ls_per_block - 1:
                        if input_conv_block.shape != net.shape:
                            projection_layer = tf.keras.layers.Conv1D(
                                filters=net.shape[-1],
                                kernel_size=kernel_size,
                                padding='valid',
                                strides=kernel_stride,
                                dilation_rate=1,
                                activation=None,
                                use_bias=False,
                                bias_initializer='zeros',
                                kernel_initializer=IdentityConv1DInitializer(),
                                name=f'local_fluxes_conv{conv_block_i}_input_proj'
                            )
                            input_conv_block = tf.keras.layers.TimeDistributed(projection_layer, name=f'local_fluxes_skip_connection_{conv_block_i}')(input_conv_block)

                        net = tf.keras.layers.Add(name=f'local_fluxes_{conv_block_i}_skip_connection_add')([net, input_conv_block])

            # if conv_block_i != n_blocks - 1:  # do not add maxpooling layer before global maxpooling layer
            #     net = tf.keras.layers.MaxPooling2D(**pool_kwargs,
            #                                        name='local_fluxes_maxpooling_{}'.format(conv_block_i))(net)
            net = tf.keras.layers.MaxPooling2D(**pool_kwargs,
                                               name='local_fluxes_maxpooling_{}'.format(conv_block_i))(net)

        # split up extracted features for each local transit view (flux, secondary, and odd-even)
        net = SplitLayer(net.shape[1], axis=1, name='local_fluxes_split')(net)

        # combine them again based on which branch they are in
        cur = 0
        for branch in conv_branches:
            sz = local_transit_sz[branch]
            if sz != 1:  # multiple views in branch
                conv_branches[branch] = (
                    tf.keras.layers.Concatenate(axis=1, name='local_fluxes_merge_{}'.format(branch))(net[cur:cur + sz]))
                cur += sz
            else:  # only one view in branch
                conv_branches[branch] = net[cur]
                cur += 1

        for _, branch in enumerate(conv_branches):  # process each branch separately

            net = conv_branches[branch]

            if branch == 'local_unfolded_flux':  # process unfolded flux
                net = process_extracted_conv_features_unfolded_flux(
                    net,
                    self.config['local_unfolded_flux_num_filters_stats'],
                    kernel_size,
                    weight_initializer,
                )
            elif branch == odd_even_branch_name:  # subtract extracted features for odd and even views branch
                net = process_extracted_conv_features_odd_even_flux(net)

            # global pooling of the convolutional branch
            if branch != 'local_unfolded_flux':
                # net = tf.keras.layers.GlobalAveragePooling2D(name=f'local_fluxes_{branch}_global_max_pooling')(net)
                net = tf.keras.layers.Flatten(name=f'local_fluxes_{branch}_flatten')(net)

            # concatenate scalar features with features extracted in the convolutional branch for the time series views
            if self.config['conv_branches'][branch]['scalars'] is not None:
                scalar_inputs = \
                    [self.inputs[feature_name] for feature_name in self.config['conv_branches'][branch]['scalars']]
                if len(scalar_inputs) > 1:
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'{branch}_scalar_input')(scalar_inputs)
                else:
                    scalar_inputs = scalar_inputs[0]

                # scalar_inputs = tf.keras.layers.BatchNormalization(name=f'{branch}_scalar_inputs_batch_norm')(scalar_inputs)
                
                net = tf.keras.layers.Concatenate(axis=1, name='local_fluxes_flatten_wscalar_{}'.format(branch))([
                    net,
                    scalar_inputs
                ])

            # add FC layer that extracts features from the combined feature vector of features from the convolutional
            # branch (flattened) and corresponding scalar features
            if self.config['branch_num_fc_units'] > 0:
                net = tf.keras.layers.Dense(units=self.config['branch_num_fc_units'],
                                            kernel_regularizer=None,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None,
                                            name='local_fluxes_{}_fc'.format(branch))(net)

                # net = tf.keras.layers.BatchNormalization(name=f'local_fluxes_{branch}_fc_batch_norm')(net)
                
                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01, name='local_fluxes_{}_fc_lrelu'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'relu':
                    net = tf.keras.layers.ReLU(name='local_fluxes_{}_fc_relu'.format(branch))(net)
                elif self.config['non_lin_fn'] == 'prelu':
                    net = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                alpha_regularizer=None,
                                                alpha_constraint=None,
                                                shared_axes=[1],
                                                name='local_fluxes_{}_fc_prelu'.format(branch))(net)

                # net = tf.keras.layers.Dropout(self.config['branch_dropout_rate_fc'],
                #                               name=f'local_fluxes_{branch}_dropout_fc_conv')(net)
            
            # net = tf.keras.layers.LayerNormalization(
            # axis=-1,
            # epsilon=1e-3,
            # center=True,
            # scale=True,
            # beta_initializer="zeros",
            # gamma_initializer="ones",
            # beta_regularizer=None,
            # gamma_regularizer=None,
            # beta_constraint=None,
            # gamma_constraint=None,
            # name=f'local_fluxes_{branch}_layer_norm')(net)

            conv_branches[branch] = net

        return conv_branches

    def build_diff_img_branch(self):
        """ Builds the difference image branch.

        :return: dict with the difference image branch
        """

        def diff_img_conv_block(net, conv_kwargs, conv_block_i, seq_conv_block_i):
            
            conv2d_layer = tf.keras.layers.Conv2D(
                    dilation_rate=1,
                    activation=None,
                    use_bias=True,
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}',
                    **conv_kwargs
                    )
            
            net = tf.keras.layers.TimeDistributed(conv2d_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_apply')(net)
            
            if self.config['batch_norm_after_conv_layers']:
                batchnorm_layer = tf.keras.layers.BatchNormalization(
                    axis=-1,
                    momentum=0.99,
                    epsilon=0.001,
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
                    synchronized=False,
                    name=f'diff_imgs_{conv_block_i}_{seq_conv_block_i}_batchnorm'
                    )
                
                net = tf.keras.layers.TimeDistributed(batchnorm_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_batchnorm_apply')(net)
            
            if self.config['use_attention_after_conv_layers']:
                attention_layer = tf.keras.layers.Attention(
                    use_scale=True,
                    score_mode='dot',
                    dropout=0.0,
                    seed=None,
                    name=f'diff_imgs_self-attention_{conv_block_i}_{seq_conv_block_i}')
                
                net = tf.keras.layers.TimeDistributed(attention_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_self-attention_apply')([net, net])

            if self.config['non_lin_fn'] == 'lrelu':
                activation_layer = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                name='diff_imgs_lrelu_{}_{}'.format(conv_block_i,
                                                                                    seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_lrelu_apply')(net)
            elif self.config['non_lin_fn'] == 'relu':
                activation_layer = tf.keras.layers.ReLU(name='diff_imgs_relu_{}_{}'.format(conv_block_i,
                                                                            seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_relu_apply')(net)
            elif self.config['non_lin_fn'] == 'prelu':
                activation_layer = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                            alpha_regularizer=None,
                                            alpha_constraint=None,
                                            shared_axes=[1, 2],
                                            name='diff_imgs_prelu_{}_{}'.format(conv_block_i,
                                                                                seq_conv_block_i))
                net = tf.keras.layers.TimeDistributed(activation_layer, name=f'diff_imgs_conv{conv_block_i}_{seq_conv_block_i}_prelu_apply')(net)
        
            return net
            
        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        branch_view_inputs = [self.inputs[view_name] for view_name in self.config['diff_img_branch']['imgs']]
        branch_view_inputs = [tf.keras.layers.Reshape(l.shape[1:] + (1,), name=f'diff_imgs_expanding_{l.name}_dims')(l)
                              for l in branch_view_inputs]

        branch_view_inputs = tf.keras.layers.Concatenate(axis=4, name='input_diff_img_concat')(branch_view_inputs)

        # get number of conv blocks, layers per block, and kernel and pool sizes for the branch
        n_blocks = self.config['diff_img_num_conv_blocks']
        kernel_size = (self.config['diff_img_kernel_size'], self.config['diff_img_kernel_size'])
        kernel_stride = (self.config['diff_img_kernel_stride'], self.config['diff_img_kernel_stride'])
        # pool_size = self.config['diff_img_pool_size']
        # pool_stride = self.config['diff_img_pool_stride']
        n_layers_per_block = self.config['diff_img_num_conv_ls_per_block']

        # pool_kwargs = {
        #     'pool_size': pool_size,
        #     'strides': pool_stride
        # }
        for conv_block_i in range(n_blocks):  # create convolutional blocks

            input_conv_block = branch_view_inputs if conv_block_i == 0 else net

            num_filters = 2 ** (self.config['diff_img_init_power_num_conv_filters'] + conv_block_i)

            for seq_conv_block_i in range(n_layers_per_block):  # create convolutional block

                # set convolution layer parameters from config
                conv_kwargs = {'filters': num_filters,
                               'kernel_initializer': weight_initializer,
                               'kernel_size': kernel_size,
                               'strides': kernel_stride if seq_conv_block_i == 0 else (1, 1),
                               'padding': 'same'  # 'valid' if seq_conv_block_i == 0 else 'same'
                               }

                net = diff_img_conv_block(branch_view_inputs if conv_block_i == 0 and seq_conv_block_i == 0 else net, conv_kwargs, conv_block_i, seq_conv_block_i)
                
                # add skip connection of branch inputs
                if self.config['use_skip_connection_conv_block']:

                    if seq_conv_block_i == n_layers_per_block - 1:
                        if input_conv_block.shape != net.shape:
                            projection_layer = tf.keras.layers.Conv2D(
                                filters=net.shape[-1],
                                kernel_size=kernel_size,
                                padding='same',  # 'valid',
                                strides=kernel_stride,
                                dilation_rate=1,
                                activation=None,
                                use_bias=False,
                                bias_initializer='zeros',
                                kernel_initializer=IdentityConv2DInitializer(),
                                name=f'diff_imgs_conv{conv_block_i}_proj'
                            )
                            input_conv_block = tf.keras.layers.TimeDistributed(projection_layer, name=f'diff_imgs_skip_connection_{conv_block_i}')(input_conv_block)

                        net = tf.keras.layers.Add(name=f'add_diff_imgs_skip_connection_{conv_block_i}')([net, input_conv_block])

            # if conv_block_i != n_blocks - 1:  # do not add maxpooling layer before global maxpooling layer
            #     net = tf.keras.layers.MaxPooling3D(**pool_kwargs,
            #                                        name=f'diff_imgs_maxpooling_{conv_block_i}')(net)

        # # apply global average pooling for all sectors/quarters
        # global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D(name=f'diff_imgs_global_avg_pooling')
        # diff_imgs_global_avg_pool = tf.keras.layers.TimeDistributed(global_avg_pooling, name='diff_imgs_global_avg_pooling_apply')(net)
        # instead of GlobalAveragePooling2D, use 1x1 Conv2D inside TimeDistributed
        conv_diff_1x1 = tf.keras.layers.Conv2D(
            filters=4,              # reduce channels to 4
            kernel_size=(1, 1),     # 1x1 convolution
            strides=(1, 1),
            padding='same',
            name='diff_imgs_1x1_conv'
        )
        # apply 1x1 conv across all sectors/quarters
        diff_imgs_conv_1x1 = tf.keras.layers.TimeDistributed(conv_diff_1x1, name='diff_imgs_1x1_conv_apply')(net)
        diff_imgs_conv_1x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.PReLU())(diff_imgs_conv_1x1)
 
        # spatial attention block (per image)
        attention_conv = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')
        attention = tf.keras.layers.TimeDistributed(attention_conv, name='diff_imgs_attention')(diff_imgs_conv_1x1)
        # apply attention weights
        diff_imgs_attention_applied = tf.keras.layers.Multiply(name='diff_imgs_attention_multiply')([diff_imgs_conv_1x1, attention])

        # flatten each time slice separately (preserve time dimension)
        diff_imgs_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), name='diff_imgs_flatten')(diff_imgs_attention_applied)

        # reshape global pooling features for all sectors/quarters
        input_conv_block = tf.keras.layers.Reshape(diff_imgs_flatten.shape[1:] + (1,),
                                                   name='diff_imgs_global_avg_pooling_expand_dims')(diff_imgs_flatten)

        # use difference image quality metric information
        net = difference_img_quality_metric_attention_fusion(input_conv_block, self.inputs['quality'])
        
        # # add per-image scalar features        
        # if self.config['diff_img_branch']['imgs_scalars'] is not None:

        #     # get quality metric values for the images
        #     qmetrics_inputs = self.inputs['quality']

        #     # # option 1: multiply extracted features by quality metrics
        #     # # repeat them to get same dimension of extracted feature maps
        #     # qmetrics_inputs = tf.keras.layers.RepeatVector(n=input_conv_block.shape[2],
        #     #                                                name='diff_imgs_repeat_qmetrics')(qmetrics_inputs)
        #     # # reshape to match same shape as extracted feature maps
        #     # qmetrics_inputs = tf.keras.layers.Permute((2, 1), name='diff_imgs_permute_qmetrics')(qmetrics_inputs)
        #     # # expand dims to match
        #     # qmetrics_inputs = tf.keras.layers.Reshape(qmetrics_inputs.shape[1:] + (1,),
        #     #                                           name='diff_imgs_qmetrics_expand_dims')(qmetrics_inputs)
        #     #
        #     # input_conv_block = tf.keras.layers.Multiply(name='diff_imgs_qmetrics_mult')([input_conv_block,
        #     #                                                                              qmetrics_inputs])

        #     # option 2: concatenate quality metric features
        #     qmetrics_inputs = tf.keras.layers.Reshape(qmetrics_inputs.shape[1:] + (1, 1),
        #                                               name='diff_imgs_qmetrics_expand_dims')(qmetrics_inputs)
        #     input_conv_block = tf.keras.layers.Concatenate(axis=2, name='diff_imgs_qmetrics_concat')([input_conv_block,
        #                                                                                               qmetrics_inputs])

        # #     scalar_inputs = [self.inputs[feature_name]
        # #                      if 'pixel' not in feature_name else self.inputs[feature_name]
        # #                      for feature_name in self.config['diff_img_branch']['imgs_scalars']]
        # #     if len(scalar_inputs) > 1:
        # #         scalar_inputs = tf.keras.layers.Concatenate(axis=0,
        # #                                                     name=f'diff_imgs_imgs_scalars_inputs_concat')(scalar_inputs)
        # #     else:
        # #         scalar_inputs = scalar_inputs[0]
        # #
        # #     net = tf.keras.layers.Concatenate(axis=2, name='diff_imgs_imgsscalars_concat')([net, scalar_inputs])
        # #
        # # input_with_img_scalars = tf.keras.layers.Reshape(net.shape[1:] + (1, ),
        # #                                                  name=f'diff_imgs_expanding_w_imgs_scalars')(net)

        # # compress features from image and scalar sector data into a set of features
        # net = tf.keras.layers.Conv2D(filters=self.config['diff_img_conv_scalar_num_filters'],
        #                              kernel_size=(1, input_conv_block.shape[2]),
        #                              strides=(1, 1),
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
        #                              name='diff_imgs_convfc',
        #                              )(input_conv_block)

        # if self.config['batch_norm_after_conv_layers']:
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
        #         name=f'diff_imgs_convfc_batch_norm'
        #     )(net)

        # if self.config['use_attention_after_conv_layers']:
        #     net = tf.keras.layers.Attention(
        #         use_scale=True,
        #         score_mode='dot',
        #         dropout=0.0,
        #         seed=None,
        #         name=f'diff_imgs_convfc_self-attention_')([net, net])

        # if self.config['non_lin_fn'] == 'lrelu':
        #     net = tf.keras.layers.LeakyReLU(alpha=0.01, name='diff_imgs_convfc_lrelu')(net)
        # elif self.config['non_lin_fn'] == 'relu':
        #     net = tf.keras.layers.ReLU(name='diff_imgs_convfc_relu')(net)
        # elif self.config['non_lin_fn'] == 'prelu':
        #     net = tf.keras.layers.PReLU(alpha_initializer='zeros',
        #                                 alpha_regularizer=None,
        #                                 alpha_constraint=None,
        #                                 shared_axes=[1],
        #                                 name='diff_imgs_convfc_prelu')(net)

        # # add skip connection of branch inputs
        # if self.config['use_skip_connection_conv_block']:
        #     # apply conv to have same number of channels as extracted feature map
        #     if input_conv_block.shape[-1] != net.shape[-1]:
        #         input_conv_block = tf.keras.layers.Conv2D(
        #             filters=net.shape[-1],
        #             kernel_size=1,
        #             padding='same',
        #             strides=1,
        #             dilation_rate=1,
        #             activation=None,
        #             use_bias=True,
        #             bias_initializer='zeros',
        #             kernel_regularizer=None,
        #             bias_regularizer=None,
        #             activity_regularizer=None,
        #             kernel_constraint=None,
        #             bias_constraint=None,
        #             name=f'diff_imgs_convfc_input',
        #         )(input_conv_block)
        #
        #     net = tf.keras.layers.Add(
        #         name=f'diff_imgs_convfc_skip_connection')([net, input_conv_block])

        net = tf.keras.layers.Flatten(data_format='channels_last', name='diff_imgs_flatten_convfc')(net)

        # add scalar features
        if self.config['diff_img_branch']['scalars'] is not None:

            scalar_inputs = [self.inputs[feature_name] for feature_name in self.config['diff_img_branch']['scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1,
                                                            name=f'diff_imgs_scalars_inputs_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            # scalar_inputs = tf.keras.layers.BatchNormalization(name='diff_imgs_scalars_inputs_batch_norm')(scalar_inputs)

            # concatenate scalar features with remaining features
            net = tf.keras.layers.Concatenate(axis=1,
                                              name='diff_imgs_flatten_w_scalar_inputs_concat')([net, scalar_inputs])

        # add FC layer that extracts features from the combined feature vector of features from the convolutional
        # branch (flattened) and corresponding scalar features
        if self.config['branch_num_fc_units'] > 0:
            net = tf.keras.layers.Dense(units=self.config['branch_num_fc_units'],
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

            # net = tf.keras.layers.BatchNormalization(name=f'diff_imgs_fc_batch_norm')(net)
            
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

            # net = tf.keras.layers.Dropout(self.config['branch_dropout_rate_fc'], name=f'diff_imgs_dropout_fc')(net)
            # net = tf.keras.layers.LayerNormalization(
            # axis=-1,
            # epsilon=1e-3,
            # center=True,
            # scale=True,
            # beta_initializer="zeros",
            # gamma_initializer="ones",
            # beta_regularizer=None,
            # gamma_regularizer=None,
            # beta_constraint=None,
            # gamma_constraint=None,
            # name='diff_imgs_layer_norm')(net)

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

            scalar_fc_output = tf.keras.layers.Dense(units=self.config['branch_num_fc_units'],
                                                     kernel_regularizer=regularizers.l2(self.config['decay_rate']) if
                                                     self.config['clf_head_fc_decay_rate'] is not None else None,
                                                     activation=None,
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     bias_initializer='zeros',
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name=f'{scalar_branch_name}_fc_summarize')(scalar_input)

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
            
            # scalar_fc_output = tf.keras.layers.BatchNormalization(name=f'{scalar_branch_name}_fc_sumarize_batch_norm')(scalar_fc_output)

            if self.config['non_lin_fn'] == 'lrelu':
                scalar_fc_output = tf.keras.layers.LeakyReLU(alpha=0.01, name=f'{scalar_branch_name}_fc_sumarize_lrelu')(
                    scalar_fc_output)
            elif self.config['non_lin_fn'] == 'relu':
                scalar_fc_output = tf.keras.layers.ReLU(name=f'{scalar_branch_name}_fc_summarize_relu')(scalar_fc_output)
            elif self.config['non_lin_fn'] == 'prelu':
                scalar_fc_output = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                                         alpha_regularizer=None,
                                                         alpha_constraint=None,
                                                         shared_axes=[1],
                                                         name=f'{scalar_branch_name}_fc_summarize_prelu')(scalar_fc_output)
            
            # scalar_fc_output = tf.keras.layers.Dropout(self.config['branch_dropout_rate_fc'], name=f'{scalar_branch_name}_fc_summarize_dropout')(scalar_fc_output)
            
            # scalar_fc_output = tf.keras.layers.LayerNormalization(
            # axis=-1,
            # epsilon=1e-3,
            # center=True,
            # scale=True,
            # beta_initializer="zeros",
            # gamma_initializer="ones",
            # beta_regularizer=None,
            # gamma_regularizer=None,
            # beta_constraint=None,
            # gamma_constraint=None,
            # name=f'{scalar_branch_name}_layer_norm')(scalar_fc_output)
            
            scalar_branches_net[scalar_branch_name] = scalar_fc_output

        return scalar_branches_net

    def connect_segments(self, branches):
        """ Connect the different branches.

        :param branches: dict, branches to be concatenated
        :return:
            model output before FC layers
        """

        branches_lst = [branch for _, branch in branches.items()]

        if len(branches_lst) == 1:  # only one convolutional branch output
            net = branches_lst[0]

        elif self.config['use_attention_before_classification_head']:  # perform multi-headed self-attention

            branches_lst = [tf.keras.layers.Reshape((1,) + branch.shape[1:],
                                                    name=f'reshape_for_cross_att_{branch.name.split("/")[0]}')(branch)
                            for branch in branches_lst]
            branches_concat = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(branches_lst)

            att_key_dim = min(branches_concat.shape[-1],
                                    self.config['attention_before_classification_head_max_key_dim'])
            att_n_heads = branches_concat.shape[-1] // att_key_dim

            net, mean_attn = attention_block(
                branches_concat,
                branches_concat,
                num_heads=att_n_heads,
                key_dim=att_key_dim,
                dropout_rate=0.1,
                return_attention=self.config['use_attention_scores']
                )
            if mean_attn is not None:
                self.mean_attn = mean_attn
        else:  # simple concatenation of outputs from convolutional branches
            net = tf.keras.layers.Concatenate(axis=1, name='convbranch_wscalar_concat')(branches_lst)
        
        # net = tf.keras.layers.BatchNormalization(name=f'input_clf_head_batch_norm')(net)
        # net = tf.keras.layers.LayerNormalization(
        # axis=-1,
        # epsilon=1e-3,
        # center=True,
        # scale=True,
        # beta_initializer="zeros",
        # gamma_initializer="ones",
        # beta_regularizer=None,
        # gamma_regularizer=None,
        # beta_constraint=None,
        # gamma_constraint=None,
        # name='input_clf_head_layer_norm')(net)

        return net

    def build_fc_block(self, net):
        """ Builds the FC block after the convolutional branches.

        :param net: model upstream the FC block
        :return:
            net: model with added FC block
        """

        for fc_layer_i in range(self.config['clf_head_num_fc_layers']):

            net = tf.keras.layers.Dense(units=self.config['clf_head_fc_neurons'],
                                        kernel_regularizer=regularizers.l2(self.config['clf_head_fc_decay_rate']),
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
            
            # net = tf.keras.layers.LayerNormalization(
            #         axis=-1,
            #         epsilon=1e-3,
            #         center=True,
            #         scale=True,
            #         beta_initializer="zeros",
            #         gamma_initializer="ones",
            #         beta_regularizer=None,
            #         gamma_regularizer=None,
            #         beta_constraint=None,
            #         gamma_constraint=None,
            #         name=f'fc{fc_layer_i}_layernorm')(net)

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

            net = tf.keras.layers.Dropout(self.config['clf_head_fc_dropout_rate'], name=f'dropout_fc{fc_layer_i}')(net)

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
                                             name='unfolded_flux_conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                             **conv_kwargs)(view_inputs if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0 else net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='unfolded_flux_lrelu{}_{}_{}'.format(branch, conv_block_i,
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
                                               name='unfolded_flux_maxpooling_{}_{}'.format(branch, conv_block_i))(net)

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
        conv_kwargs = {'filters': self.config['n_conv_filter_unfolded_stats'],
                       'kernel_initializer': weight_initializer,
                       'kernel_size': (self.config['kernel_size_unfolded_stats'], 1),
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
            # 'global_centroid',
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
                               if branch == odd_even_branch_name else kernel_size,  # self.config['kernel_size'],
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
            view_inputs = tf.keras.layers.Concatenate(axis=-1, name=f'local_flux_concat_{local_transit_feature[0]}_with_var')(view_inputs)

            local_transit_views_inputs.append(view_inputs)

        # combine the local transits to put through conv block (dim = [view, bins, avg/var view])
        if len(local_transit_views_inputs) > 1:
            branch_view_inputs = tf.keras.layers.Concatenate(axis=1, name='local_flux_concat_local_views')(local_transit_views_inputs)
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
                    tf.keras.layers.Concatenate(axis=1, name='local_flux_transit_merge_{}'.format(branch))(net[cur:cur + sz]))
                cur += sz
            else:  # only one view in branch
                conv_branches[branch] = net[cur]
                cur += 1

        for branch in conv_branches:  # create a convolutional branch

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
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'local_flux_{branch}_scalar_input')(scalar_inputs)
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
        n_blocks = self.config['num_diffimg_conv_blocks'] 
        kernel_size = (self.config['kernel_size_diffimg'], self.config['kernel_size_diffimg'], 1)

        # get pool size for the given view
        pool_size = (self.config['pool_size_diffimg'], self.config['pool_size_diffimg'], 1)  

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (self.config['init_diffimg_conv_filters'] + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': kernel_size,  
                           'strides': 1,
                           'padding': 'same'
                           }

            for seq_conv_block_i in range(self.config['diffimg_conv_ls_per_block']):  # create convolutional block

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

            scalar_inputs = [tf.keras.layers.Permute((2, 1), name=f'permute_{feature_name}')(self.inputs[feature_name])
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

class ExoMinerPlusPlusTemp(object):
    """ ExoMiner++ prototype for developing ideas."""

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

    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, name):
        """ Implements encoder-only transformer.

        :param inputs:
        :param head_size: int, head size
        :param num_heads: int, number of heads
        :ff_dim: kernel size
        :dropout: float, dropout rate
        :name: str, name for layers
        :return: encoder output

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
                self.config['num_units_transformer_fc_layers'] = self.config['global-num_units_transformer_fc_layers']
            elif 'local' in branch:
                self.config['num_units_transformer_fc_layers'] = self.config['local-num_units_transformer_fc_layers']

            for fc_layer_i, num_units in enumerate(self.config['num_units_transformer_fc_layers']):
                # add FC layer that extracts features from the combined feature vector of features from the lstm
                # branch (flattened) and corresponding scalar features
                net = tf.keras.layers.Dense(units=num_units,
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

    def build_conv_unfolded_flux_alt(self):
        """ Uses Conv2D as a way to process unfolded flux.

        Returns: net, output of this unfolded flux branch

        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'},
                         'kernel_size': {'global_view': 'kernel_size_glob', 'local_view': 'kernel_size_loc'},
                         }

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        # initialize branch
        view_inputs = self.inputs["unfolded_local_flux_view_fluxnorm"]

        # expand dims to prevent last dimension being "lost"
        view_inputs = tf.keras.layers.Reshape(view_inputs.shape + (1,), name='expanding_unfolded_flux_dim')(view_inputs)

        branch = "local_unfolded_flux"

        # get number of conv blocks for the given view
        n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

        # get pool size for the given view
        kernel_size = self.config[config_mapper['kernel_size']['local_view']]

        # get pool size for the given view
        pool_size = self.config[config_mapper['pool_size']['local_view']]

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (self.config['init_conv_filters'] + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {'filters': num_filters,
                           'kernel_initializer': weight_initializer,
                           'kernel_size': (kernel_size, kernel_size),  # self.config['kernel_size'],
                           'strides': (self.config['kernel_stride'], self.config['kernel_stride']),
                           'padding': 'same'
                           }

            for seq_conv_block_i in range(self.config['conv_ls_per_block']):  # create convolutional block
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
                                             **conv_kwargs)(view_inputs if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0 else net)

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

            net = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                               strides=(self.config['pool_stride'], self.config['pool_stride']),
                                               name='maxpooling_{}_{}'.format(branch, conv_block_i))(net)
        net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(branch))(net)

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

        net = tf.keras.layers.Dropout(self.config['dropout_rate_fc_conv'])(net)

        return net

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
                # net = tf.keras.layers.Conv1D(dilation_rate=1,
                #                              activation=None,
                #                              use_bias=True,
                #                              bias_initializer='zeros',
                #                              kernel_regularizer=None,
                #                              bias_regularizer=None,
                #                              activity_regularizer=None,
                #                              kernel_constraint=None,
                #                              bias_constraint=None,
                #                              name='conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                #                              **conv_kwargs)(view_inputs if conv_block_i == 0 and
                #                                                            seq_conv_block_i == 0 else net)
                net = tf.keras.layers.Conv2D(dilation_rate=1,
                                             activation=None,
                                             use_bias=True,
                                             bias_initializer='zeros',
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None,
                                             name='unfolded_flux_conv{}_{}_{}'.format(branch, conv_block_i, seq_conv_block_i),
                                             **conv_kwargs)(view_inputs if conv_block_i == 0 and
                                                                           seq_conv_block_i == 0 else net)

                if self.config['non_lin_fn'] == 'lrelu':
                    net = tf.keras.layers.LeakyReLU(alpha=0.01,
                                                    name='unfolded_flux_lrelu{}_{}_{}'.format(branch, conv_block_i,
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
                                               name='unfolded_flux_maxpooling_{}_{}'.format(branch, conv_block_i))(net)

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

        # net = tf.keras.layers.Conv1D(dilation_rate=1,
        #                              activation=None,
        #                              use_bias=True,
        #                              bias_initializer='zeros',
        #                              kernel_regularizer=None,
        #                              bias_regularizer=None,
        #                              activity_regularizer=None,
        #                              kernel_constraint=None,
        #                              bias_constraint=None,
        #                              name='conv{}_{}'.format(branch, 1),
        #                              **conv_kwargs)(net)
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
            # 'global_centroid',
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
                               if branch == odd_even_branch_name else kernel_size,  # self.config['kernel_size'],
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
            view_inputs = tf.keras.layers.Concatenate(axis=-1, name=f'local_flux_concat_{local_transit_feature[0]}_with_var')(view_inputs)

            local_transit_views_inputs.append(view_inputs)

        # combine the local transits to put through conv block (dim = [view, bins, avg/var view])
        if len(local_transit_views_inputs) > 1:
            branch_view_inputs = tf.keras.layers.Concatenate(axis=1, name='local_flux_concat_local_views')(local_transit_views_inputs)
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
                # net = tf.keras.layers.Conv1D(dilation_rate=1,
                #                              activation=None,
                #                              use_bias=True,
                #                              bias_initializer='zeros',
                #                              kernel_regularizer=None,
                #                              bias_regularizer=None,
                #                              activity_regularizer=None,
                #                              kernel_constraint=None,
                #                              bias_constraint=None,
                #                              name='conv{}_{}'.format(conv_block_i, seq_conv_block_i),
                #                              **conv_kwargs)(branch_view_inputs if conv_block_i == 0 and
                #                                                                   seq_conv_block_i == 0
                #                                             else net)
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
                    tf.keras.layers.Concatenate(axis=1, name='local_flux_transit_merge_{}'.format(branch))(net[cur:cur + sz]))
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
                    scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'local_flux_{branch}_scalar_input')(scalar_inputs)
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

        # config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
        #                  'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'},
        #                  'kernel_size': {'global_view': 'kernel_size_glob', 'local_view': 'kernel_size_loc'},
        #                  }

        weight_initializer = tf.keras.initializers.he_normal() \
            if self.config['weight_initializer'] == 'he' else 'glorot_uniform'

        branch_view_inputs = [self.inputs[view_name] for view_name in self.config['diff_img_branch']['imgs']]

        # expanding dimensionality
        branch_view_inputs = [tf.keras.layers.Reshape(l.shape[1:] + (1,), name=f'expanding_{l.name}_dims')(l)
                              for l in branch_view_inputs]

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
        n_blocks = 3  # self.config[config_mapper['blocks'][('local_view', 'global_view')['global' in branch]]]

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
                                                   name='maxpooling_diff_img_{}_{}'.format(conv_block_i,
                                                                                           seq_conv_block_i))(net)

        # flatten output of the convolutional branch
        net = tf.keras.layers.Permute((1, 2, 4, 3), name='permute_diff_imgs')(net)
        net = tf.keras.layers.Reshape((np.prod(list(net.shape[1:-1])), net.shape[-1]),
                                      name='flatten_diff_imgs')(net)

        # add per-image scalar features
        if self.config['diff_img_branch']['imgs_scalars'] is not None:

            scalar_inputs = [tf.keras.layers.Permute((2, 1), name=f'permute_{feature_name}')(self.inputs[feature_name])
                             if 'pixel' not in feature_name else self.inputs[feature_name]
                             for feature_name in self.config['diff_img_branch']['imgs_scalars']]
            if len(scalar_inputs) > 1:
                scalar_inputs = tf.keras.layers.Concatenate(axis=1, name=f'diff_img_imgsscalars_concat')(scalar_inputs)
            else:
                scalar_inputs = scalar_inputs[0]

            # concatenate per-image scalar features with extracted features from the difference images
            net = tf.keras.layers.Concatenate(axis=1, name='flatten_wscalar_diff_img_imgsscalars')([net, scalar_inputs])

        # 2D convolution with kernel size equal to feature map size
        # net = tf.expand_dims(net, axis=-1, name='expanding_input_convfc')
        # net = tf.keras.layers.Conv2D(filters=4,  # self.config['num_fc_conv_units'],
        #                              kernel_size=net.shape[1:-1],
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
        #                              name='convfc_{}'.format('diff_img'),
        #                              )(net)
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

        if self.config['transformer_branches'] is not None:
            branches_net.update(self.build_transformer_layers())

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

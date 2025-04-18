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



class StdLayer(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(StdLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):

        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)
        return tf.sqrt(variance)


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

    def call(self, inputs):
        """ Calls the custom split layer.

        Args:
            inputs: TF Keras layer, input to be split

        Returns: if num_or_size_splits is an int returns a list of num_or_size_splits Tensor objects; if
            num_or_size_splits is a 1-D list for 1-D Tensor returns num_or_size_splits.get_shape[0] Tensor objects
            resulting from splitting value.

        """

        return tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis)


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
        # branch_view_inputs = tf.keras.layers.Permute((2, 3, 1, 4), name='permute_inputs')(branch_view_inputs)

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
            global_max_pooling_img = tf.keras.layers.GlobalMaxPool2D(name=f'diff_imgs_global_max_pooling_{img_i}')(extracted_img)
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

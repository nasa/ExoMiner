import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, losses, optimizers

# from tensorflow.keras.layers import Lambda
import numpy as np
import tensorflow.keras.backend as K

# local
from utils_model import create_inputs


class ExoMiner_TESS_Transit_Detection(object):

    def __init__(self, config, features):
        """ """

        # model configuration (parameters and hyperparameters)
        self.config = config["config"]
        self.features = features

        self.output_size = 1  # binary classification with sigmoid output layer

        self.inputs = create_inputs(
            features=self.features, feature_map=config["feature_map"]
        )  # by default no feature map

        # build the model
        self.outputs = self.build()

        self.kerasModel = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def build_flux_window_branch(self):
        """
        Builds the flux window branch

        Returns: dict, flux window branch
        """
        # no config mapper used

        weight_initializer = (
            tf.keras.initializers.he_normal()
            if self.config["global_branch_settings"]["weight_initializer"] == "he"
            else "glorot_uniform"
        )

        # initialize branch
        branch_name = "flux_window"
        view_inputs = self.inputs[
            self.config["flux_window_branch_settings"]["window"][0]
        ]

        # expand dims to prevent last dimension being "lost"
        # (None, 100) -> (None, 100, 1) in (batch_size, steps, channels)
        view_inputs = tf.expand_dims(view_inputs, axis=-1, name="expanding_flux_window")
        # view_inputs = Lambda(
        #     lambda x: tf.expand_dims(x, axis=-1), name="expanding_flux_window"
        # )(view_inputs)

        n_blocks = self.config["flux_window_branch_settings"]["n_blocks"]
        kernel_size = self.config["flux_window_branch_settings"]["kernel_size"]
        kernel_stride = self.config["flux_window_branch_settings"]["kernel_stride"]
        pool_size = self.config["flux_window_branch_settings"]["pool_size"]
        pool_stride = self.config["flux_window_branch_settings"]["pool_stride"]
        init_conv_filters = self.config["flux_window_branch_settings"][
            "init_conv_filters"
        ]
        conv_ls_per_block = self.config["flux_window_branch_settings"][
            "conv_ls_per_block"
        ]  # self.config['conv_ls_per_block']

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (init_conv_filters + conv_block_i)

            # set convolution layer parameters from config
            conv_kwargs = {
                "filters": num_filters,
                "kernel_initializer": weight_initializer,
                "kernel_size": kernel_size,
                "strides": kernel_stride,
                "padding": "same",
            }

            for seq_conv_block_i in range(
                conv_ls_per_block
            ):  # create convolutional layers for the block
                net = tf.keras.layers.Conv1D(
                    dilation_rate=1,
                    activation=None,
                    use_bias=True,
                    bias_initializer="zeros",
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name="conv{}_{}_{}".format(
                        branch_name, conv_block_i, seq_conv_block_i
                    ),
                    **conv_kwargs,
                )(view_inputs if conv_block_i == 0 and seq_conv_block_i == 0 else net)

                if self.config["global_branch_settings"]["non_lin_fn"] == "lrelu":
                    net = tf.keras.layers.LeakyReLU(
                        alpha=0.01,
                        name="lrelu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        ),
                    )(net)
                elif self.config["global_branch_settings"]["non_lin_fn"] == "relu":
                    net = tf.keras.layers.ReLU(
                        name="relu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        )
                    )(net)
                elif self.config["global_branch_settings"]["non_lin_fn"] == "prelu":
                    net = tf.keras.layers.PReLU(
                        alpha_initializer="zeros",
                        alpha_regularizer=None,
                        alpha_constraint=None,
                        shared_axes=[1, 2],
                        name="prelu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        ),
                    )(net)
            if self.config["flux_window_branch_settings"]["pool_size"] not in (None, 0):
                net = tf.keras.layers.MaxPooling1D(
                    pool_size=pool_size,
                    strides=pool_stride,
                    name="maxpooling_{}_{}".format(branch_name, conv_block_i),
                )(net)

        # flatten output of the final conv block (None, 12, 16) -> (None, 192)
        net = tf.keras.layers.Flatten(
            data_format="channels_last", name="flatten_{}".format(branch_name)
        )(net)

        if self.config["global_branch_settings"]["num_fc_conv_units"] > 0:
            net = tf.keras.layers.Dense(
                units=self.config["global_branch_settings"]["num_fc_conv_units"],
                kernel_regularizer=None,
                activation=None,
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name="fc_{}".format(branch_name),
            )(net)

            if self.config["global_branch_settings"]["non_lin_fn"] == "lrelu":
                net = tf.keras.layers.LeakyReLU(
                    alpha=0.01, name="fc_lrelu_{}".format(branch_name)
                )(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "relu":
                net = tf.keras.layers.ReLU(name="fc_relu_{}".format(branch_name))(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "prelu":
                net = tf.keras.layers.PReLU(
                    alpha_initializer="zeros",
                    alpha_regularizer=None,
                    alpha_constraint=None,
                    shared_axes=[1],
                    name="fc_prelu_{}".format(branch_name),
                )(net)

            net = tf.keras.layers.Dropout(
                self.config["global_branch_settings"]["dropout_rate_fc_conv"],
                name="dropout_fc_conv_{}".format(branch_name),
            )(net)
            # keep at 0

        return {branch_name: net}

    def build_diff_img_branch(self):
        """Builds the difference image branch.

        :return: dict with the difference image branch
        """

        weight_initializer = (
            tf.keras.initializers.he_normal()
            if self.config["global_branch_settings"]["weight_initializer"] == "he"
            else "glorot_uniform"
        )

        branch_name = "diff_img"

        branch_view_inputs = [
            self.inputs[view_name]
            for view_name in self.config["diff_img_branch_settings"]["imgs"]
        ]  # inital shape per input is (None, 33, 33)
        branch_view_inputs = [
            tf.expand_dims(l, axis=-1, name=f"expanding_diff_img_{i}")
            for i, l in enumerate(branch_view_inputs)
        ]  # [(None, 33, 33, 1) ..]

        # branch_view_inputs = [
        #     Lambda(
        #         lambda l: tf.expand_dims(l, axis=-1),
        #         name=f"expanding_diff_img_{i}",
        #         # output_shape=(33, 33, 1),  # Needed for deserialization?
        #     )(l)
        #     for i, l in enumerate(branch_view_inputs)
        # ]

        branch_view_inputs = tf.keras.layers.Concatenate(
            axis=3, name="input_diff_img_concat"
        )(
            branch_view_inputs
        )  # (None, 33, 33, 4)

        n_blocks = self.config["diff_img_branch_settings"]["n_blocks"]
        kernel_size = self.config["diff_img_branch_settings"]["kernel_size"]
        kernel_stride = self.config["diff_img_branch_settings"]["kernel_stride"]
        pool_size = self.config["diff_img_branch_settings"]["pool_size"]
        pool_stride = self.config["diff_img_branch_settings"]["pool_stride"]
        init_conv_filters = self.config["diff_img_branch_settings"]["init_conv_filters"]
        conv_ls_per_block = self.config["diff_img_branch_settings"]["conv_ls_per_block"]

        for conv_block_i in range(n_blocks):  # create convolutional blocks

            num_filters = 2 ** (
                init_conv_filters + conv_block_i
            )  # exponential number of filters per block 4, 8, 16?

            # set convolution layer parameters from config
            conv_kwargs = {
                "filters": num_filters,
                "kernel_initializer": weight_initializer,
                "kernel_size": kernel_size,
                "strides": kernel_stride,
                "padding": "same",
            }

            for seq_conv_block_i in range(
                conv_ls_per_block
            ):  # create convolutional block

                net = tf.keras.layers.Conv2D(
                    dilation_rate=1,
                    activation=None,
                    use_bias=True,
                    bias_initializer="zeros",
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name="conv{}_{}_{}".format(
                        branch_name, conv_block_i, seq_conv_block_i
                    ),
                    **conv_kwargs,
                )(
                    branch_view_inputs
                    if conv_block_i == 0 and seq_conv_block_i == 0
                    else net
                )

                if self.config["global_branch_settings"]["non_lin_fn"] == "lrelu":
                    net = tf.keras.layers.LeakyReLU(
                        alpha=0.01,
                        name="lrelu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        ),
                    )(net)
                elif self.config["global_branch_settings"]["non_lin_fn"] == "relu":
                    net = tf.keras.layers.ReLU(
                        name="relu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        )
                    )(net)
                elif self.config["global_branch_settings"]["non_lin_fn"] == "prelu":
                    net = tf.keras.layers.PReLU(
                        alpha_initializer="zeros",
                        alpha_regularizer=None,
                        alpha_constraint=None,
                        shared_axes=[1, 2],
                        name="prelu{}_{}_{}".format(
                            branch_name, conv_block_i, seq_conv_block_i
                        ),
                    )(net)
            if self.config["diff_img_branch_settings"]["pool_size"] not in (None, 0):
                net = tf.keras.layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_stride,
                    name="maxpooling_{}_{}_{}".format(
                        branch_name, conv_block_i, seq_conv_block_i
                    ),
                )(net)
        # flatten output of the final conv block (None, 4, 4, 16) -> (None, 16, 16)

        # net = tf.keras.layers.Reshape((np.prod(net.shape[1:-1].as_list()), net.shape[-1]),
        #                               name='flatten_{}'.format(branch_name))(net)

        # flatten 2d feature map -> 1d array
        # approach 1: 1 filter, same kernel, conv 2d: (x, y, 1) 1d feature map -> flatten for fc layer input
        # approach 2: flatten feature map from conv blocks to x,y, num_filters in last conv block,

        net = tf.keras.layers.Flatten(
            data_format="channels_last", name="flatten_convfc_{}".format(branch_name)
        )(net)

        if self.config["global_branch_settings"]["num_fc_conv_units"] > 0:
            net = tf.keras.layers.Dense(
                units=self.config["global_branch_settings"]["num_fc_conv_units"],
                kernel_regularizer=None,
                activation=None,
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name="fc_{}".format(branch_name),
            )(net)

            if self.config["global_branch_settings"]["non_lin_fn"] == "lrelu":
                net = tf.keras.layers.LeakyReLU(
                    alpha=0.01, name="fc_lrelu_{}".format(branch_name)
                )(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "relu":
                net = tf.keras.layers.ReLU(name="fc_relu_{}".format(branch_name))(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "prelu":
                net = tf.keras.layers.PReLU(
                    alpha_initializer="zeros",
                    alpha_regularizer=None,
                    alpha_constraint=None,
                    shared_axes=[1],
                    name="fc_prelu_{}".format(branch_name),
                )(net)

            net = tf.keras.layers.Dropout(
                self.config["global_branch_settings"]["dropout_rate_fc_conv"],
                name="dropout_fc_{}".format(branch_name),
            )(net)

        return {branch_name: net}

    def connect_segments(self, branches):
        """Connect the different branches.

        :param branches: dict, branches to be concatenated
        :return:
            model output before FC layers
        """

        branches_to_concatenate = []
        for branch_name, branch in branches.items():
            branches_to_concatenate.append(branch)

        if len(branches_to_concatenate) > 1:
            net = tf.keras.layers.Concatenate(axis=1, name="convbranch_wscalar_concat")(
                branches_to_concatenate
            )  # todo: update name?
        else:
            net = branches_to_concatenate[0]

        if self.config["connect_segment_settings"]["batch_norm"]:  # go without
            net = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=1e-3,
                center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
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
                name="batch_norm_convbranch_wscalar_concat",
            )(net)

        return net

    def build_fc_block(self, net):
        """Builds the FC block after the convolutional branches.

        :param net: model upstream the FC block
        :return:
            net: model with added FC block
        """

        # with tf.variable_scope('FcNet'):

        for fc_layer_i in range(self.config["fc_block_settings"]["num_fc_layers"]):

            fc_neurons = self.config["fc_block_settings"]["init_fc_neurons"]

            if self.config["fc_block_settings"]["decay_rate"] is not None:
                net = tf.keras.layers.Dense(
                    units=fc_neurons,
                    kernel_regularizer=(
                        regularizers.l2(self.config["fc_block_settings"]["decay_rate"])
                        if self.config["fc_block_settings"]["decay_rate"] is not None
                        else None
                    ),
                    activation=None,
                    use_bias=True,
                    kernel_initializer=self.config["global_branch_settings"][
                        "kernel_initializer"
                    ],
                    bias_initializer=self.config["global_branch_settings"][
                        "bias_initializer"
                    ],
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name="fc{}".format(fc_layer_i),
                )(net)
            else:
                net = tf.keras.layers.Dense(
                    units=fc_neurons,
                    kernel_regularizer=None,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=self.config["global_branch_settings"][
                        "kernel_initializer"
                    ],
                    bias_initializer=self.config["global_branch_settings"][
                        "bias_initializer"
                    ],
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name="fc{}".format(fc_layer_i),
                )(net)

            if self.config["global_branch_settings"]["non_lin_fn"] == "lrelu":
                net = tf.keras.layers.LeakyReLU(
                    alpha=0.01, name="fc_lrelu{}".format(fc_layer_i)
                )(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "relu":
                net = tf.keras.layers.ReLU(name="fc_relu{}".format(fc_layer_i))(net)
            elif self.config["global_branch_settings"]["non_lin_fn"] == "prelu":
                net = tf.keras.layers.PReLU(
                    alpha_initializer="zeros",
                    alpha_regularizer=None,
                    alpha_constraint=None,
                    shared_axes=[1],
                    name="fc_prelu{}".format(fc_layer_i),
                )(net)

            net = tf.keras.layers.Dropout(
                self.config["fc_block_settings"]["dropout_rate"],
                name=f"dropout_fc{fc_layer_i}",
            )(net)

        return net

    def build(self):
        """Builds the model.

        :return:
            output: full model, from inputs to outputs
        """

        branches_net = {}

        if self.config["flux_window_branch_settings"] is not None:
            branches_net.update(self.build_flux_window_branch())

        if self.config["diff_img_branch_settings"] is not None:
            branches_net.update(self.build_diff_img_branch())

        # merge branches
        net = self.connect_segments(branches_net)

        # create FC layers
        net = self.build_fc_block(net)

        # create output layer
        logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        if self.output_size == 1:
            output = tf.keras.layers.Activation(tf.nn.sigmoid, name="sigmoid")(
                logits
            )  # currently used
        else:
            output = tf.keras.layers.Activation(tf.nn.softmax, name="softmax")(logits)

        return output

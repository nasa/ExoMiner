"""
Custom estimator built using the estimator API from TensorFlow.

[1] Shallue, Christopher J., and Andrew Vanderburg. "Identifying exoplanets with deep learning: A five-planet resonant
chain around kepler-80 and an eighth planet around kepler-90." The Astronomical Journal 155.2 (2018): 94.
[2] Ansdell, Megan, et al. "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning."
The Astrophysical Journal Letters 869.1 (2018): L7.

"""

# 3rd party
import tensorflow as tf
import copy
import operator
import os
import tempfile
import _pickle as pickle
import numpy as np
import itertools

# local
from src.utils_train import phase_inversion, phase_shift, add_whitegaussiannoise


class InputFn(object):
    """Class that acts as a callable input function for the Estimator."""

    def __init__(self, file_pattern, batch_size, mode, label_map, features_set, data_augmentation=False,
                 filter_data=None, scalar_params_idxs=None):
        """Initializes the input function.

        :param file_pattern: File pattern matching input TFRecord files, e.g. "/tmp/train-?????-of-00100".
        May also be a comma-separated list of file patterns.
        :param batch_size: int, batch size
        :param mode: A tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        :param label_map: dict, map between class name and integer value
        :param features_set: dict of the features to be extracted from the dataset, the key is the feature name and the
        value is a dict with the dimension 'dim' of the feature and its data type 'dtype'
        (can the dimension and data type be inferred from the tensors in the dataset?)
        :param data_augmentation: bool, if True data augmentation is performed
        :param filter_data:
        :param scalar_params_idxs: list, indexes of features to extract from the scalar features Tensor
        :return:
        """

        self._file_pattern = file_pattern
        self._mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        self.features_set = features_set
        self.data_augmentation = data_augmentation and self._mode == tf.estimator.ModeKeys.TRAIN
        self.scalar_params_idxs = scalar_params_idxs

        self.filter_data = filter_data

    def __call__(self, config, params):
        """ Builds the input pipeline.

        :param config: dict, parameters and hyperparameters for the model
        :param params:
        :return:
            a tf.data.Dataset with features and labels

        TODO: does the call method need config and parameters?
        """

        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors.

            :param serialized_example: a single tf.Example
            :return:
                tuple, feature and label tensors
            """

            # get features names, shapes and data types to be extracted from the TFRecords
            data_fields = {feature_name: tf.FixedLenFeature([feature_info['dim']], feature_info['dtype'])
                           for feature_name, feature_info in self.features_set.items()}

            # # get auxiliary data from TFRecords required to perform data augmentation
            # if self.data_augmentation:
            #     for feature_name in self.features_set:
            #         if 'view' in feature_name:
            #             data_fields[feature_name + '_rmsoot'] = tf.FixedLenFeature([], tf.float32)

            # get labels if in TRAIN or EVAL mode
            # FIXME: change the feature name to 'label' - standardization of TCE feature names across different
            #  TFRecords (Kepler/TESS) sources - remember that for backward compatibility we need to keep
            #  av_training_set
            if include_labels:
                data_fields['av_training_set'] = tf.FixedLenFeature([], tf.string)

            # initialize filtering data fields
            if self.filter_data is not None:
                data_fields['kepid'] = tf.FixedLenFeature([], tf.int64)
                data_fields['tce_plnt_num'] = tf.FixedLenFeature([], tf.int64)

            # Parse the features.
            parsed_features = tf.parse_single_example(serialized_example, features=data_fields)

            # data augmentation - time axis flipping
            # TODO: do we need this check?
            # if reverse_time_series_prob > 0:
            if self.data_augmentation:
                # Randomly reverse time series features with probability reverse_time_series_prob.
                should_reverse = tf.less(tf.random_uniform([], 0, 1), 0.5, name="should_reverse")
                bin_shift = [-5, 5]
                shift = tf.random.uniform(shape=(), minval=bin_shift[0], maxval=bin_shift[1],
                                          dtype=tf.dtypes.int32, name='randuniform')

            # initialize feature output
            output = {'time_series_features': {}}
            if self.filter_data is not None:
                output['filt_features'] = {}

            label_id = tf.cast(0, dtype=tf.int32, name='cast_label_to_int32')

            for feature_name, value in parsed_features.items():

                if 'oot' in feature_name:
                    continue
                # label
                # FIXME: change the feature name to 'label' - standardization of TCE feature names across different
                #  TFRecords (Kepler/TESS) sources - remember that for backward compatibility we need to keep
                #  av_training_set
                elif include_labels and feature_name == 'av_training_set':

                    # map label to integer
                    label_id = label_to_id.lookup(value)

                    # Ensure that the label_id is non negative to verify a successful hash map lookup.
                    assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.cast(0, dtype=tf.int32)),
                                                   ["Unknown label string:", value], name='assert_non-negativity')

                    with tf.control_dependencies([assert_known_label]):
                        label_id = tf.identity(label_id)

                # filtering features
                elif self.filter_data is not None and feature_name == 'kepid':
                    output['filt_features']['kepid'] = value
                elif self.filter_data is not None and feature_name == 'tce_plnt_num':
                    output['filt_features']['tce_n'] = value

                # scalar features (e.g, stellar, TCE, transit fit parameters)
                elif feature_name == 'scalar_params':
                    if self.scalar_params_idxs is None:
                        output['scalar_params'] = value
                    else:  # choose only some of the scalar_params based on their indexes
                        output['scalar_params'] = tf.gather(value, indices=self.scalar_params_idxs, axis=0)

                # time-series features
                else:  # input_config.features[feature_name].is_time_series:

                    # data augmentation
                    if self.data_augmentation:

                        # with tf.variable_scope('input_data/data_augmentation'):

                        # invert phase
                        value = phase_inversion(value, should_reverse)

                        # phase shift some bins
                        value = phase_shift(value, shift)

                        # add white gaussian noise
                        value = add_whitegaussiannoise(value, parsed_features[feature_name + '_meanoot'],
                                                       parsed_features[feature_name + '_stdoot'])

                    output['time_series_features'][feature_name] = value

            # FIXME: should it return just output when in PREDICT mode? Would have to change predict.py yielding part
            return output, label_id

        def filt_func(x, y):
            """ Utility function used to filter examples from the dataset based on their target ID + TCE planet number

            :param x: feature tensor
            :param y: label tensor
            :return:
                boolean tensor, True for valid examples, False otherwise
            """

            z1 = tf.as_string(x['filt_features']['target_id'])
            z_ = tf.convert_to_tensor('_')
            z2 = tf.as_string(x['filt_features']['tce_plnt_num'])
            zj = tf.strings.join([z1, z_, z2])

            return tf.math.reduce_any(tf.math.equal(zj,
                                                    tf.convert_to_tensor(self.filter_data['target_id+tce_plnt_num'])))

        def get_features_and_labels(x, y):
            """ Utility function used to remove the features used to filter the dataset.

            :param x: feature tensor
            :param y: label tensor
            :return:
                tuple, dict with features tensor, and label tensor
            """

            return {'time_series_features': x['time_series_features']}, y

        with tf.variable_scope('input_data'):

            # Create a HashTable mapping label strings to integer ids.
            table_initializer = tf.contrib.lookup.KeyValueTensorInitializer(
                keys=list(self.label_map.keys()),
                values=list(self.label_map.values()),
                key_dtype=tf.string,
                value_dtype=tf.int32)

            label_to_id = tf.contrib.lookup.HashTable(table_initializer, default_value=-1)

            include_labels = self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

            file_patterns = self._file_pattern.split(",")
            filenames = []
            for p in file_patterns:
                matches = tf.gfile.Glob(p)
                if not matches:
                    raise ValueError("Found no input files matching {}".format(p))
                filenames.extend(matches)

            tf.logging.info("Building input pipeline from %d files matching patterns: %s", len(filenames), file_patterns)

            # create filename dataset based on the list of tfrecords filepaths
            filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

            # map a TFRecordDataset object to each tfrecord filepath
            dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

            # shuffle the dataset if training or evaluating
            # FIXME: for perfect sampling, the buffer_size should be larger than the size of the dataset. Can we handle it?
            #        set variables for buffer size and shuffle seed?
            # if self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            if self._mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(27000, seed=None)

            # do not repeat the dataset
            dataset = dataset.repeat(1)

            # map the example parser across the tfrecords dataset to extract the examples and manipulate them
            # (e.g., real-time data augmentation, shuffling, ...)
            # dataset = dataset.map(_example_parser, num_parallel_calls=4)
            # number of parallel calls is set dynamically based on available CPU; it defines number of parallel calls to
            # process asynchronously
            dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # filter the dataset based on the filtering features
            if self.filter_data is not None:
                dataset = dataset.filter(filt_func)

                # remove the filtering features from the dataset
                dataset = dataset.map(get_features_and_labels)

            # creates batches by combining consecutive elements
            dataset = dataset.batch(self.batch_size)

            # prefetches batches determined by the buffer size chosen
            # parallelized processing in the CPU with model computations in the GPU
            dataset = dataset.prefetch(max(1, int(256 / self.batch_size)))

        return dataset


class ModelFn(object):
    """ Class that acts as a callable model function for the Estimator. """

    def __init__(self, model_class, config):
        """ Initialized the model function.

        :param model_class: type of model
        :param config: dict, parameters and hyperparameters for the model
        :return:
        """

        self._model_class = model_class
        self._base_config = config

    def __call__(self, features, labels, mode):
        """

        :param features: feature tensor
        :param labels: label tensor
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        :return:
            EstimatorSpec
        """

        config = copy.deepcopy(self._base_config)

        # initialize model instance of the class
        model = self._model_class(features, labels, config, mode)

        # create train op if training mode
        train_op = self.create_train_op(model) if mode == tf.estimator.ModeKeys.TRAIN else None

        # create metrics if mode is not predict
        metrics = None if mode == tf.estimator.ModeKeys.PREDICT else self.create_metrics(model)

        logging_hook = None  # if mode == tf.estimator.ModeKeys.TRAIN else None

        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     tf.summary.scalar('accuracy', metrics['accuracy'][1])
        #     tf.summary.scalar('precision', metrics['precision'][1])
        #     s = tf.summary.merge_all()
        #     logging_hook = [tf.train.SummarySaverHook(save_steps=100, output_dir=config.model_dir_custom, summary_op=s)]

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=model.predictions,
                                          loss=model.total_loss,
                                          train_op=train_op,
                                          eval_metric_ops=metrics,
                                          training_hooks=logging_hook,
                                          )

    @staticmethod
    def create_metrics(model):
        """Builds TensorFlow operations to compute model evaluation metrics.

        Args:
            model
        # labels: Tensor with shape [batch_size].
        # predictions: Tensor with shape [batch_size, output_dim].
        # weights: Tensor with shape [batch_size].
        # batch_losses: Tensor with shape [batch_size].
        # output_dim: Dimension of model output

        Returns:
        A dictionary {metric_name: (metric_value, update_op).
        """

        # initialize dictionary to save metrics
        metrics = {}

        if model.output_size == 1:  # 1 output

            assert model.predictions.shape[1] == 1

            # removes dimensions of size 1 from the shape of a tensor.
            predictions = tf.squeeze(model.predictions, axis=[1], name='squeezed_predictions')

            # thresholding scores at 0.5
            predicted_labels = tf.cast(tf.greater(predictions, 0.5, name='thresholding'), name="predicted_labels",
                                       dtype=tf.int32)

        else:  # 2 or more outputs

            # num_samples x num_classes, 2 dimensions
            assert len(model.predictions.shape) == 2

            predictions = model.predictions

            # index with the largest score across the output (class) axis
            # num_samples x 1
            predicted_labels = tf.argmax(model.predictions, axis=1, name="predicted_labels", output_type=tf.int32)

            # TODO: implement one-hot encoding
            # one_hot_labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)

        labels = model.labels

        num_classes = max(model.config['label_map'].values()) + 1

        # compute metrics

        # across-class accuracy
        metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predicted_labels, name='accuracy')
        metrics['mean_per_class_accuracy'] = tf.metrics.mean_per_class_accuracy(labels=labels,
                                                                                predictions=predicted_labels,
                                                                                num_classes=num_classes,
                                                                                name='mean_per_class_accuracy')

        # TODO: implement these metrics for multiclass classification
        if model.output_size == 1:  # metrics that can only be computed for binary classification

            metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predicted_labels, name='precision')
            metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predicted_labels, name='recall')

            # define the number of thresholds used
            metrics['prec thr'] = tf.metrics.precision_at_thresholds(labels, predictions,
                                                                     np.linspace(0, 1, num=1000, endpoint=True,
                                                                                 dtype='float32'),
                                                                     name='precision_at_thresholds')
            metrics['rec thr'] = tf.metrics.recall_at_thresholds(labels, predictions,
                                                                 np.linspace(0, 1, num=1000, endpoint=True,
                                                                             dtype='float32'),
                                                                 name='recall_at_thresholds')

            metrics["roc auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                                summation_method='careful_interpolation', curve='ROC', name='roc_auc')
            metrics["pr auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                               summation_method='careful_interpolation', curve='PR', name='pr_auc')

            # TODO: make mean avg precision work
            # metrics['avg prec'] = tf.metrics.average_precision_at_k(labels, predictions,
            #                                                         labels.get_shape().as_list()[1])

        # auxiliary functions for computing confusion matrix
        def _metric_variable(name, shape, dtype):
            """ Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
            return tf.get_variable(
                name,
                initializer=tf.zeros(shape, dtype),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])

        def _count_condition(name, labels_value, predicted_value):
            """ Creates a counter for given values of predictions and labels. """
            count = _metric_variable(name, [], tf.float32)
            is_equal = tf.cast(tf.logical_and(tf.equal(labels, labels_value),
                                                  tf.equal(predicted_labels, predicted_value)), dtype=tf.float32)
            update_op = tf.assign_add(count, tf.reduce_sum(tf.ones_like(model.labels, dtype=tf.float32) * is_equal))
            return count.read_value(), update_op

        # confusion matrix metrics
        num_labels = 2 if not model.config['multi_class'] else num_classes
        for label in range(num_labels):
            for pred_label in range(num_labels):
                metric_name = "label_{}_pred_{}".format(label, pred_label)
                metrics[metric_name] = _count_condition(metric_name, labels_value=label, predicted_value=pred_label)

        return metrics

    @staticmethod
    def create_train_op(model):

        # if model.config['lr_scheduler'] in ["inv_exp_fast", "inv_exp_slow"]:
        #     global_step = tf.train.get_global_step()
        #     # global_step = tf.Variable(0, trainable=False)
        #     # global_step = tf.train.get_or_create_global_step()
        #
        #     # decay learning rate every 'decay_epochs' by 'decay_rate'
        #     decay_rate = 0.5
        #     decay_epochs = 32 if model.config['lr_scheduler'] == "inv_exp_fast" else 64
        #
        #     learning_rate = tf.train.exponential_decay(
        #         learning_rate=model.config['lr'],
        #         global_step=global_step,
        #         decay_steps=int(decay_epochs * model.n_train / self._base_config['batch_size']),
        #         decay_rate=decay_rate,
        #         staircase=True)
        #
        # elif model.config['lr_scheduler'] == "constant":
        #     learning_rate = model.config['lr']
        # else:
        #     learning_rate = None
        #     NotImplementedError('Learning rate scheduler not recognized')

        # if model.config.optimizer == 'Adam':
        if model.config['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=model.config['lr'], beta1=0.9, beta2=0.999, epsilon=1e-8)
        else:
            optimizer = tf.train.MomentumOptimizer(model.config['lr'], model.config['sgd_momentum'],
                                                   # use_nesterov=True
                                                   )

        return tf.contrib.training.create_train_op(total_loss=model.total_loss, optimizer=optimizer)


class CNN1dModel(object):

    def __init__(self, features, labels, config, mode):
        """ Initializes the CNN 1D model. The core structure follows Shallue & Vandenburg [1], one convolutional branch
        per view (global and local), each one with one or more channels (flux, centroid, odd and even, ...)

        :param features: feature tensor
        :param labels: label tensor
        :param config: dict, model configuration for its parameters and hyperparameters
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.mode = mode  # TRAIN, EVAL or PREDICT

        # features
        self.time_series_features = features['time_series_features']
        if 'scalar_params' in features:
            self.scalar_params = features['scalar_params']
        else:
            self.scalar_params = None

        self.labels = labels  # labels

        # self.is_training = None
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        self.logits = None
        self.predictions = None

        # losses
        # total loss adds possible regularization terms
        self.batch_losses = None
        self.total_loss = None

        # if doing multiclassification or using softmax as output layer, the output has to be equal to the number of
        # classes
        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # class-label weights for weighted loss
        # convert from numpy to tensor
        self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        # build the model
        self.build()

    def build_cnn_layers(self):
        """ Builds the convolutiuonal branches.

        :return:
            cnn_layers, dict with the convolutional branches
        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        # set weight initializer
        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        for view in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s' % view):

                # add the different time series for a view as channels in the same convolutional branch
                input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                if view in feature_name], axis=-1, name='input_{}'.format(view))

                # set number of convolutional blocks for the convolutional branch
                n_blocks = self.config[config_mapper['blocks'][view]]

                # get pool size for the maxpooling layers for this convolutional branch
                pool_size = self.config[config_mapper['pool_size'][view]]

                # create convolutional blocks
                for conv_block_i in range(n_blocks):

                    # number of filters in a specific convolutional block
                    num_filters = self.config['init_conv_filters'] * (2 ** conv_block_i)

                    kwargs = {'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
                              'padding': "same"}

                    # create convolutional layers for each convolutional block
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
                                                     name='conv{}_{}'.format(view, conv_block_i, seq_conv_block_i),
                                                     **kwargs)(input if conv_block_i == 0 and seq_conv_block_i == 0
                                                                                             else net)

                        # set activation
                        net = tf.keras.layers.LeakyReLU(alpha=0.01)(net) if self.config['non_lin_fn'] == 'prelu' \
                            else tf.keras.layers.ReLU()(net)

                    # create maxpooling layer
                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=self.config['pool_stride'],
                                                       name='maxpooling{}_{}'.format(view, conv_block_i))(net)

                    # # set batch normalization for the output of the convolutional branch
                    # if self.config['batch_norm'] and seq_conv_block_i == n_blocks - 1:
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
                net.get_shape().assert_has_rank(3)
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(view))(net)

            # create dictionary with key/value pairs for each convolutional branch
            cnn_layers[view] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv branches
        :return:
            pre_logits_concat: model before logits
        """

        # Sort the hidden layers by dictionary key (name) because the order of dictionary items is
        # nondeterministic between invocations of Python.
        # this is now a list of tuples (key, val)
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one branch
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if self.scalar_params is not None:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)(
                [pre_logits_concat, self.scalar_params])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        with tf.variable_scope('FcNet'):

            # create sequence of FC layers
            for fc_layer_i in range(self.config['num_fc_layers']):

                # fc_neurons = self.config.init_fc_neurons / (2 ** fc_layer_i)
                fc_neurons = self.config['init_fc_neurons']

                if self.config['decay_rate'] is not None:
                    net = tf.keras.layers.Dense(units=fc_neurons,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
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

                # only during training this layer does something
                net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=self.is_training)

            # create output FC layer
            logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        self.logits = logits

    def build(self):
        """ Builds the model.

        :return:
        """

        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     self.is_training = tf.placeholder_with_default(True, [], "is_training")
        # else:
        #     self.is_training = False

        # create convolutional columns for local and global views
        cnn_layers = self.build_cnn_layers()

        # merge columns
        net = self.connect_segments(cnn_layers)

        # create FC layers
        self.build_fc_layers(net)

        # transform outputs to predictions - map logits to "probabilities"; softmax for multiclass classification or
        # sigmoid for the binary case
        # TODO: are these functions equivalent to the corresponding loss functions in build_losses()?
        prediction_fn = tf.nn.softmax if self.config['multi_class'] else tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        # build loss if in TRAIN or EVAL modes
        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        """ Build the loss. Weighted or non-weighted cross-entropy. Sigmoid for binary classification, softmax for
        multiclassification.

        :return:
        """

        # map class weights to samples
        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.cast(self.labels, dtype=tf.int32)))

        if self.output_size == 1:  # sigmoid CE
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, dtype=tf.float32),
                                                                   logits=tf.squeeze(self.logits, [1]),
                                                                   name='batch_losses')
        else:  # softmax CE
            # the sparse version does not use the one-hot encoding; the probability of a given label is exclusive
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                          name='batch_losses')

        # Compute the weighted mean cross entropy loss and add it to the LOSSES collection
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES collection (e.g. regularization losses)
        self.total_loss = tf.losses.get_total_loss(name='total_loss')

        self.batch_losses = batch_losses


class CNN1dPlanetFinderv1(object):

    def __init__(self, features, labels, config, mode):
        """ Initializes the CNN 1D Planet Finder v1 model. The core structure consists of separate convolutional
        branches for non-related types of input (flux, centroid, odd and even time series, ...).

        :param features: feature tensor
        :param labels: label tensor
        :param config: dict, model configuration for its parameters and hyperparameters
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.mode = mode  # TRAIN, EVAL or PREDICT

        # features
        self.time_series_features = features['time_series_features']
        if 'scalar_params' in features:
            self.scalar_params = features['scalar_params']
        else:
            self.scalar_params = None

        self.labels = labels  # labels

        self.is_training = None

        self.logits = None
        self.predictions = None

        # losses
        # total loss adds possible regularization terms
        self.batch_losses = None
        self.total_loss = None

        # if doing multiclassification or using softmax as output layer, the output has to be equal to the number of
        # classes
        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # class-label weights for weighted loss
        # convert from numpy to tensor
        self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        # build the model
        self.build()

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv columns
        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' \
            else 'glorot_uniform'

        cnn_layers = {}
        # TODO: NEED TO CHANGE THIS MANUALLY...
        # for view in ['global_view', 'local_view', 'local_view_centr', 'global_view_centr', 'global_view_oddeven',
        # 'local_view_oddeven']:
        for view in ['global_view', 'local_view']:

            with tf.variable_scope('ConvNet_%s' % view):

                # add the different time series for a view as channels
                if 'oddeven' not in view:  # for branches that do not involve odd-even views
                    input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                     if view == feature_name], axis=-1, name='input_{}'.format(view))
                elif view == 'local_oddeven':  # local odd-even view
                    input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                      if feature_name in ['local_view_odd', 'local_view_even']],
                                     axis=-1,
                                     name='input_{}'.format(view))
                else:  # 'global_oddeven'
                    input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                      if feature_name in ['global_view_odd', 'global_view_even']],
                                     axis=-1,
                                     name='input_{}'.format(view))

                # get number of conv blocks for the given view
                n_blocks = self.config[config_mapper['blocks'][('local_view', 'global_view')['global_view' in view]]]

                # get pool size for the given view
                pool_size = self.config[
                    config_mapper['pool_size'][('local_view', 'global_view')['global_view' in view]]]

                for conv_block_i in range(n_blocks):

                    num_filters = self.config['init_conv_filters'] * (2 ** conv_block_i)

                    # set convolution layer parameters from config
                    kwargs = {'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
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
                                                     name='conv{}_{}_{}'.format(view, conv_block_i, seq_conv_block_i),
                                                     **kwargs)(input if conv_block_i == 0 and seq_conv_block_i == 0
                                                                                             else net)

                        net = tf.keras.layers.LeakyReLU(alpha=0.01)(net) if self.config['non_lin_fn'] == 'prelu' \
                        else tf.keras.layers.ReLU()(net)

                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=self.config['pool_stride'],
                                                       name='maxpooling{}{}'.format(view, conv_block_i))(net)

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
                net.get_shape().assert_has_rank(3)
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(view))(net)
                # net_shape = net.get_shape().as_list()
                # output_dim = net_shape[1] * net_shape[2]
                # net = tf.reshape(net, [-1, output_dim], name="flatten")

            cnn_layers[view] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv columns/branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv columns
        :return:
            model before logits
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one column
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])
            # pre_logits_concat = tf.concat([layer[1] for layer in time_series_hidden_layers],
            #                               axis=1, name="pre_logits_concat")

        # concatenate scalar params
        if self.scalar_params is not None:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)(
                [pre_logits_concat, self.scalar_params])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        with tf.variable_scope('FcNet'):

            for fc_layer_i in range(self.config['num_fc_layers']):

                # fc_neurons = self.config.init_fc_neurons / (2 ** fc_layer_i)
                fc_neurons = self.config['init_fc_neurons']

                if self.config['decay_rate'] is not None:
                    net = tf.keras.layers.Dense(units=fc_neurons,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
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

                net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net, training=self.is_training)

            # create output FC layer
            logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        self.logits = logits

    def build(self):
        """ Builds the model.

        :return:
        """

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        # create convolutional columns for local and global views
        cnn_layers = self.build_cnn_layers()

        # merge columns
        net = self.connect_segments(cnn_layers)

        # create FC layers
        self.build_fc_layers(net)

        # transform outputs to predictions
        prediction_fn = tf.nn.softmax if self.config['multi_class'] else tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        # build loss if in TRAIN or EVAL modes
        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        """ Build the loss. Weighted or non-weighted cross-entropy. Sigmoid for binary classification, softmax for
        multiclassification.

        :return:
        """

        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.cast(self.labels, dtype=tf.int32)))

        if self.output_size == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, dtype=tf.float32),
                                                                   logits=tf.squeeze(self.logits, [1]),
                                                                   name='batch_losses')
        else:
            # the sparse version does not use the one-hot encoding; the probability of a given label is exclusive
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                          name='batch_losses')

        # Compute the weighted mean cross entropy loss and add it to the LOSSES collection
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES collection (e.g. regularization losses)
        self.total_loss = tf.losses.get_total_loss(name='total_loss')

        self.batch_losses = batch_losses


class Exonet(object):

    def __init__(self, features, labels, config, mode):
        """ Initializes the Exonet model [2]. The core structure is based on [1] and consists of two branches - local
        and global view - with centroid time series as additional channel besides the flux time series. Stellar
        parameters are concatenated with the outputs from the two convolutional branches before the FC layers.

        :param features: feature tensor
        :param labels: label tensor
        :param config: dict, model configuration for its parameters and hyperparameters
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.mode = mode  # TRAIN, EVAL or PREDICT

        # features
        self.time_series_features = features['time_series_features']
        if 'scalar_params' in features:
            self.scalar_params = features['scalar_params']
        else:
            self.scalar_params = None

        self.labels = labels  # labels

        self.is_training = None

        self.logits = None
        self.predictions = None

        # losses
        # total loss adds possible regularization terms
        self.batch_losses = None
        self.total_loss = None

        # if doing multiclassification or using softmax as output layer, the output has to be equal to the number of
        # classes
        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # class-label weights for weighted loss
        # convert from numpy to tensor
        self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        # build the model
        self.build()

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv columns
        """

        weight_initializer = 'glorot_uniform'

        num_filters = {'local_view': [16, 32], 'global_view': [16, 32, 64, 128, 256]}
        pool_size = {'local_view': 7, 'global_view': 5}
        conv_ls_per_block = {'local_view': 2, 'global_view': 2}

        cnn_layers = {}
        for view in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s' % view):

                # add the different time series for a view as channels
                input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                 if view in feature_name],
                                 axis=-1,
                                 name='input_{}'.format(view))

                for conv_block_i in range(len(num_filters[view])):

                    kwargs = {'filters': num_filters[view][conv_block_i],
                              'kernel_initializer': weight_initializer,
                              'kernel_size': 5,
                              'strides': 1,
                              'padding': "same"}

                    for seq_conv_block_i in range(conv_ls_per_block[view]):

                        net = tf.keras.layers.Conv1D(dilation_rate=1,
                                                     activation=None,
                                                     use_bias=True,
                                                     bias_initializer='zeros',
                                                     kernel_regularizer=None,
                                                     bias_regularizer=None,
                                                     activity_regularizer=None,
                                                     kernel_constraint=None,
                                                     bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(conv_block_i, 0, view),
                                                     **kwargs)(input if seq_conv_block_i == 0 else net)

                        net = tf.keras.layers.ReLU()(net)

                    net = tf.keras.layers.MaxPooling1D(pool_size=pool_size[view],
                                                       strides=2,
                                                       name='maxpooling{}_{}'.format(view, conv_block_i))(net)

                # Flatten
                net.get_shape().assert_has_rank(3)
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(view))(net)

            cnn_layers[view] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv columns/branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv columns
        :return:
            model before logits
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one column
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if self.scalar_params is not None:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)(
                [pre_logits_concat, self.scalar_params])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        fc_neurons = 512

        with tf.variable_scope('FcNet'):

            for fc_layer_i in range(4):

                if self.config['decay_rate'] is not None:
                    net = tf.keras.layers.Dense(units=fc_neurons,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
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

            # create output FC layer
            logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        self.logits = logits

    def build(self):
        """ Builds the model.

        :return:
        """

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        # create convolutional columns for local and global views
        cnn_layers = self.build_cnn_layers()

        # merge columns
        net = self.connect_segments(cnn_layers)

        # create FC layers
        self.build_fc_layers(net)

        # transform outputs to predictions
        prediction_fn = tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        # build loss if in TRAIN or EVAL modes
        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        """ Build the loss. Weighted or non-weighted cross-entropy. Sigmoid for binary classification, softmax for
        multiclassification.

        :return:
        """

        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.cast(self.labels, dtype=tf.int32)))

        if self.output_size == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, dtype=tf.float32),
                                                                   logits=tf.squeeze(self.logits, [1]),
                                                                   name='batch_losses')
        else:
            # the sparse version does not use the one-hot encoding; the probability of a given label is exclusive
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                          name='batch_losses')

        # Compute the weighted mean cross entropy loss and add it to the LOSSES collection
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES collection (e.g. regularization losses)
        self.total_loss = tf.losses.get_total_loss(name='total_loss')

        self.batch_losses = batch_losses


class Exonet_XS(object):

    def __init__(self, features, labels, config, mode):
        """ Initializes the Exonet-XS model [2]. Shallower version of Exonet that converts the convolutional features
        to a single featuere per view by using global max pooling layers at the end of the convolutional branches.

        :param features: feature tensor
        :param labels: label tensor
        :param config: dict, model configuration for its parameters and hyperparameters
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        """

        # model configuration (parameters and hyperparameters)
        self.config = config
        self.mode = mode  # TRAIN, EVAL or PREDICT

        # features
        self.time_series_features = features['time_series_features']
        if 'scalar_params' in features:
            self.scalar_params = features['scalar_params']
        else:
            self.scalar_params = None

        self.labels = labels  # labels

        self.is_training = None

        self.logits = None
        self.predictions = None

        # losses
        # total loss adds possible regularization terms
        self.batch_losses = None
        self.total_loss = None

        # if doing multiclassification or using softmax as output layer, the output has to be equal to the number of
        # classes
        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:  # binary classification with sigmoid output layer
            self.output_size = 1

        # class-label weights for weighted loss
        # convert from numpy to tensor
        self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        # build the model
        self.build()

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv columns
        """

        weight_initializer = 'glorot_uniform'

        num_filters = {'local_view': [16, 16], 'global_view': [16, 16, 32]}
        pool_size = {'local_view': 2, 'global_view': 2}
        conv_ls_per_block = {'local_view': 1, 'global_view': 1}

        cnn_layers = {}
        for view in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s' % view):

                # add the different time series for a view as channels
                input = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                 if view in feature_name], -1)

                for conv_block_i in range(len(num_filters[view])):

                    kwargs = {'filters': num_filters[view][conv_block_i],
                              'kernel_initializer': weight_initializer,
                              'kernel_size': 5,
                              'strides': 1,
                              'padding': "same"}

                    for seq_conv_block_i in range(conv_ls_per_block[view]):

                        net = tf.keras.layers.Conv1D(dilation_rate=1, activation=None, use_bias=True,
                                                     bias_initializer='zeros', kernel_regularizer=None,
                                                     bias_regularizer=None, activity_regularizer=None,
                                                     kernel_constraint=None, bias_constraint=None,
                                                     name='conv{}_{}_{}'.format(view, conv_block_i, seq_conv_block_i),
                                                     **kwargs)(input if conv_block_i == 0 and seq_conv_block_i == 0
                                                                                             else net)

                        net = tf.keras.layers.ReLU()(net)

                    if conv_block_i < len(num_filters[view]) - 1:
                        net = tf.keras.layers.MaxPooling1D(pool_size=pool_size[view], strides=2,
                                                           name='maxpooling{}_{}'.format(view, conv_block_i))(net)
                    else:
                        net = tf.keras.layers.GlobalMaxPooling1D(name='globalmaxpooling{}}'.format(view))(net)

                # Flatten
                net.get_shape().assert_has_rank(3)
                net = tf.keras.layers.Flatten(data_format='channels_last', name='flatten_{}'.format(view))(net)

            cnn_layers[view] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):
        """ Connect the different conv columns/branches; also has the option to concatenate additional features
        (stellar params for example)

        :param cnn_layers: dict with the different conv columns
        :return:
            model before logits
        """

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the conv hidden layers.
        if len(time_series_hidden_layers) == 1:  # only one column
            pre_logits_concat = time_series_hidden_layers[0][1]  # how to set a name for the layer?
        else:  # more than one column
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat', axis=-1)(
                [branch_output[1] for branch_output in time_series_hidden_layers])

        # concatenate scalar params
        if self.scalar_params is not None:
            pre_logits_concat = tf.keras.layers.Concatenate(name='pre_logits_concat_scalar_params', axis=-1)(
                [pre_logits_concat, self.scalar_params])

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        fc_neurons = 512

        with tf.variable_scope('FcNet'):

            for fc_layer_i in range(1):

                if self.config['decay_rate'] is not None:
                    net = tf.keras.layers.Dense(units=fc_neurons,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
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

            # create output FC layer
            logits = tf.keras.layers.Dense(units=self.output_size, name="logits")(net)

        self.logits = logits

    def build(self):
        """ Builds the model.

        :return:
        """

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        # create convolutional columns for local and global views
        cnn_layers = self.build_cnn_layers()

        # merge columns
        net = self.connect_segments(cnn_layers)

        # create FC layers
        self.build_fc_layers(net)

        # transform outputs to predictions
        prediction_fn = tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        # build loss if in TRAIN or EVAL modes
        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        """ Build the loss. Weighted or non-weighted cross-entropy. Sigmoid for binary classification, softmax for
        multiclassification.

        :return:
        """

        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.cast(self.labels, dtype=tf.int32)))

        if self.output_size == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, dtype=tf.float32),
                                                                   logits=tf.squeeze(self.logits, [1]),
                                                                   name='batch_losses')
        else:
            # the sparse version does not use the one-hot encoding; the probability of a given label is exclusive
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                          name='batch_losses')

        # Compute the weighted mean cross entropy loss and add it to the LOSSES collection
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES collection (e.g. regularization losses)
        self.total_loss = tf.losses.get_total_loss(name='total_loss')

        self.batch_losses = batch_losses


def get_model_dir(path):
    """Returns a randomly named, non-existing model file folder.

    :param path: str, root directory for saved model directories
    :return:
        model_dir_custom: str, model directory
    """

    def _gen_dir():
        return os.path.join(path, tempfile.mkdtemp().split('/')[-1])

    model_dir_custom = _gen_dir()

    while os.path.isdir(model_dir_custom):  # try until there is no current directory with that name
        model_dir_custom = _gen_dir()

    return model_dir_custom


def get_ce_weights(label_map, tfrec_dir, datasets=['train'], label_fieldname='label', verbose=False):
    """ Compute class cross-entropy weights based on the amount of labels for each class.

    :param label_map: dict, map between class name and integer value
    :param tfrec_dir: str, filepath to directory with the tfrecords
    :param datasets: list, datasets used to compute the CE weights
    :param label_fieldname: str, name of the label field in the TFRecords
    :param verbose: bool
    :return:
        ce_weights: list, weight for each class (class 0, class 1, ...)
    """

    max_label_val = max(label_map.values())

    # assumes TFRecords filenames (train, val, test)-xxxxx
    filenames = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir) if file.split('-')[0] in datasets]

    label_vec, example = [], tf.train.Example()

    n_samples = {dataset: 0 for dataset in datasets}
    for file in filenames:

        file_dataset = file.split('/')[-1].split('-')[0]

        record_iterator = tf.python_io.tf_record_iterator(path=file)
        try:
            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)
                label = example.features.feature[label_fieldname].bytes_list.value[0].decode("utf-8")
                label_vec.append(label_map[label])

                n_samples[file_dataset] += 1

        except tf.errors.DataLossError as err:
            print("Oops: " + str(err))

    # count instances for each class based on their indexes
    label_counts = [label_vec.count(category) for category in range(max_label_val + 1)]

    # give more weight to classes with less instances
    ce_weights = [max(label_counts) / max(count_i, 1e-7) for count_i in label_counts]

    if verbose:
        for dataset in datasets:
            print('Number of examples for dataset {}: {}'.format(dataset, label_counts))
        print('CE weights: {}'.format(ce_weights))

    return ce_weights


# TODO: do we need this function at all?
def picklesave(path, savedict):
    """

    :param path:
    :param savedict:
    :return:
    """

    p = pickle.Pickler(open(path, "wb+"))
    p.fast = True
    p.dump(savedict)


def get_num_samples(label_map, tfrec_dir, datasets, label_fieldname='label'):
    """ Compute number of samples in the datasets for each class.

    :param label_map: dict, map between class name and integer value
    :param tfrec_dir: str, filepath to directory with the tfrecords
    :param datasets: list, datasets to be counted. It follows that convention that the tfrecords have in their name
    train, val, or test if they contain examples pertaining to the training, validation or test sets, respectively.
    :return:
        n_samples: int, number of samples in the datasets for each class
    """

    n_samples = {dataset: {label: 0 for label in label_map.values()} for dataset in datasets}

    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir) if
                 any([dataset in file for dataset in datasets])]

    for file in filenames:

        file_dataset = file.split('/')[-1]
        curr_dataset = datasets[np.where([dataset in file_dataset for dataset in datasets])[0][0]]

        record_iterator = tf.python_io.tf_record_iterator(path=file)
        try:
            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)
                try:
                    label = example.features.feature[label_fieldname].bytes_list.value[0].decode("utf-8")
                except ValueError as e:
                    print('No label field found on the example. Ignoring it.')
                    print('Error output:', e)
                    continue

                n_samples[curr_dataset][label_map[label]] += 1

        except tf.errors.DataLossError as err:
            print("Oops: " + str(err))

    return n_samples


def get_data_from_tfrecord(tfrecord, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a tfrecord file.

    :param tfrecord: str, tfrecord filepath
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field

    # TODO: add lookup table
    #       deal with errors
    """

    # valid fields and features in the TFRecords
    # main fields: ['target_id', 'tce_plnt_num', 'label']
    FIELDS = ['tce_period', 'tce_duration', 'tce_time0bk', 'mes', 'ra', 'dec', 'mag', 'mag_uncert', 'tce_time0bk_err',
              'tce_period_err', 'tce_duration_err', 'transit_depth', 'transit_depth_err', 'sectors']
    TIMESERIES = ['global_view', 'local_view', 'local_view_centr', 'global_view_centr', 'global_view_odd',
                  'local_view_odd']

    if filt is not None:
        union_fields = np.union1d(data_fields, list(filt.keys()))  # get fields that are in both
        if 'target_id+tce_plnt_num' in list(filt.keys()):  # add target_id and tce_plnt_num
            union_fields = np.concatenate((union_fields, ['target_id', 'tce_plnt_num']))
    else:
        union_fields = data_fields

    # initialize data dict
    data = {field: [] for field in union_fields}

    if filt is not None:
        data['selected_idxs'] = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord)
    try:
        for string_record in record_iterator:

            example = tf.train.Example()
            example.ParseFromString(string_record)

            if filt is not None:
                data['selected_idxs'].append(False)

            # extracting data fields
            datum = {}
            if 'label' in union_fields:
                label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                if label_map is not None:  # map from label to respective integer
                    datum['label'] = label_map[label]

            if 'original_label' in union_fields:
                datum['original_label'] = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            if 'target_id' in union_fields:
                datum['target_id'] = example.features.feature['target_id'].int64_list.value[0]

            if 'tce_plnt_num' in union_fields:
                datum['tce_plnt_num'] = example.features.feature['tce_plnt_num'].int64_list.value[0]

            if 'sectors' in union_fields:  # for TESS data
                datum['sectors'] = example.features.feature['sectors'].bytes_list.value[0].decode("utf-8")

            # float parameters
            for field in FIELDS:
                if field in union_fields:
                    datum[field] = example.features.feature[field].float_list.value[0]

            # time-series features
            for timeseries in TIMESERIES:
                if timeseries in union_fields:
                    datum[timeseries] = example.features.feature[timeseries].float_list.value

            # filtering
            if filt is not None:

                if 'label' in filt.keys() and datum['label'] not in filt['label'].values:
                    continue

                if 'original label' in filt.keys() and datum['original label'] not in filt['original label'].values:
                    continue

                if coupled:
                    if 'target_id+tce_plnt_num' in filt.keys() and \
                            '{}_{}'.format(datum['target_id'], datum['tce_plnt_num']) not in filt['target_id+tce_plnt']:
                        continue
                else:
                    if 'target_id' in filt.keys() and datum['target_id'] not in filt['target_id']:
                        continue
                    if 'tce_plnt_num' in filt.keys() and datum['tce_plnt_num'] not in filt['tce_plnt_num']:
                        continue

                if 'tce_period' in filt.keys() and \
                        not filt['tce_period'][0] <= datum['tce_period'] <= filt['tce_period'][1]:
                    continue

                if 'tce_duration' in filt.keys() and \
                        not filt['tce_duration'][0] <= datum['tce_duration'] <= filt['tce_duration'][1]:
                    continue

                if 'tce_time0bk' in filt.keys() and \
                        not filt['tce_time0bk'][0] <= datum['tce_time0bk'] <= filt['tce_time0bk'][1]:
                    continue

                if 'MES' in filt.keys() and not filt['mes'][0] <= datum['mes'] <= filt['mes'][1]:
                    continue

                data['selected_idxs'][-1] = True

            # add example
            for field in data_fields:
                data[field].append(datum[field])

    except:
        print('Corrupted TFRecord: {}'.format(tfrecord))

    return data


def get_data_from_tfrecord_kepler(tfrecord, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a tfrecord file.

    :param tfrecord: str, tfrecord filepath
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field

    # TODO: add lookup table
    #       right now, it assumes that all examples have the same fields
    #       deal with errors
    """

    EPHEMERIS = {'tce_period': 'tce_period', 'tce_duration': 'tce_duration', 'epoch': 'tce_time0bk', 'MES': 'mes'}
    TIMESERIES = ['global_view', 'local_view', 'local_view_centr', 'global_view_centr', 'global_view_odd',
                  'local_view_odd']

    if filt is not None:
        union_fields = np.union1d(data_fields, list(filt.keys()))
        if 'kepid+tce_n' in list(filt.keys()):
            union_fields = np.concatenate((union_fields, ['kepid', 'tce_n']))
    else:
        union_fields = data_fields

    data = {field: [] for field in data_fields}

    if filt is not None:
        data['selected_idxs'] = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord)
    try:
        for string_record in record_iterator:

            example = tf.train.Example()
            example.ParseFromString(string_record)

            if filt is not None:
                data['selected_idxs'].append(False)

            # extracting data fields
            datum = {}
            if 'label' in union_fields:
                datum['label'] = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
                if label_map is not None:
                    datum['label'] = label_map[datum['label']]
                    if 'original label' in union_fields:
                        datum['original label'] = example.features.feature['av_training_set'].bytes_list.value[0].\
                            decode("utf-8")

            if 'kepid' in union_fields:
                datum['kepid'] = example.features.feature['kepid'].int64_list.value[0]

            if 'tce_n' in union_fields:
                datum['tce_n'] = example.features.feature['tce_plnt_num'].int64_list.value[0]

            # ephemeris data
            for ephem in EPHEMERIS:
                if ephem in union_fields:
                    datum[ephem] = example.features.feature[EPHEMERIS[ephem]].float_list.value[0]

            # time series
            for timeseries in TIMESERIES:
                if timeseries in union_fields:
                    datum[timeseries] = example.features.feature[timeseries].float_list.value

            # filtering
            if filt is not None:

                if 'label' in filt.keys() and datum['label'] not in filt['label'].values:
                    continue

                if 'original label' in filt.keys() and datum['original label'] not in filt['original label'].values:
                    continue

                if coupled:
                    if 'kepid+tce_n' in filt.keys() and '{}_{}'.format(datum['kepid'], datum['tce_n']) \
                            not in filt['kepid+tce_n']:
                        continue
                else:
                    if 'kepid' in filt.keys() and datum['kepid'] not in filt['kepid']:
                        continue
                    if 'tce_n' in filt.keys() and datum['tce_n'] not in filt['tce_n']:
                        continue

                if 'tce_period' in filt.keys() and not filt['tce_period'][0] <= datum['tce_period'] <= filt['tce_period'][1]:
                    continue

                if 'tce_duration' in filt.keys() and not filt['tce_duration'][0] <= datum['tce_duration'] <= \
                                                         filt['tce_duration'][1]:
                    continue

                if 'epoch' in filt.keys() and not filt['epoch'][0] <= datum['epoch'] <= filt['epoch'][1]:
                    continue

                if 'MES' in filt.keys() and not filt['MES'][0] <= datum['MES'] <= filt['MES'][1]:
                    continue

                data['selected_idxs'][-1] = True

            # add example
            for field in data_fields:
                data[field].append(datum[field])

    except:
        print('Corrupted TFRecord: {}'.format(tfrecord))

    return data


def get_data_from_tfrecords(tfrecords, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a set of tfrecord files.

    :param tfrecords: list of tfrecords filepaths.
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field
    """

    data = {field: [] for field in data_fields}

    if filt is not None:
        data['selected_idxs'] = []

    for tfrecord in tfrecords:
        data_aux = get_data_from_tfrecord(tfrecord, data_fields, label_map=label_map, filt=filt, coupled=coupled)
        for field in data_aux:
            data[field].extend(data_aux[field])

    return data


def create_filtered_tfrecord(src_tfrecord, save_dir, filt, append_name='', kw_filt_args=None):
    """ Create filtered tfrecord from a source tfrecord file.

    :param: src_tfrecord:
    :param save_dir: str, directory in which the new tfrecords are saved
    :param filt: dict, containing as keys the fields used to filter the examples. For 'label', 'kepid' and 'tce_n' the
    values should be a list; for the other data_fields, it should be a two element list that defines the interval of
    acceptable values
    :param append_name: str, string added to the name of the new filtered tfrecord
    :param kw_filt_args: dict, keyword parameters for function get_data_from_tfrecord
    :return:
    """

    if kw_filt_args is None:
        kw_filt_args = {}

    # get boolean indexes for the examples in the tfrecord file
    filt_idx = get_data_from_tfrecord(src_tfrecord, [], label_map=None, filt=filt, **kw_filt_args)['selected_idxs']

    # tfrecord destination filepath
    dest_tfrecord = save_dir + src_tfrecord.split['/'][-1] + append_name

    # write new tfrecord
    with tf.python_io.TFRecordWriter(dest_tfrecord) as writer:
        # create iterator for source tfrecord 
        record_iterator = tf.python_io.tf_record_iterator(path=src_tfrecord)
        # go over the examples in the source tfrecord
        for i, string_record in enumerate(record_iterator):
            if not filt_idx[i]:  # filter out examples
                continue

            # add example to the new tfrecord
            example = tf.train.Example()
            example.ParseFromString(string_record)
            if example is not None:
                writer.write(example.SerializeToString())


if __name__ == '__main__':

    import paths
    from src.config import label_map

    # get number of samples in the datasets
    multi_class = False
    satellite = 'kepler'
    label_map = label_map[satellite][multi_class]
    tfrec_dir = '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_dr25_manual_2dkepler_centroid_oddeven_' \
                'normsep_nonwhitened_gapped_2001-201'  # '/data5/tess_project/Data/tfrecords/180k_tfrecord'
    # nsamples = get_num_samples(label_map, tfrec_dir, ['predict'])
    # print(nsamples)

    # assert that there are no repeated examples in the datasets based on KeplerID and TCE number
    # multi_class = False
    # satellite = 'kepler'
    # label_map = label_map[satellite][multi_class]
    # tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_dr25_manual_2dkeplerwhitened_2001-201'
    tfrecords = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir)]
    # data_fields = ['kepid', 'tce_period', 'tce_duration', 'global_view', 'local_view', 'MES', 'epoch', 'label']
    data_fields = ['kepid']
    data_dict = get_data_from_tfrecords(tfrecords, data_fields, label_map=None, filt=None)
    print(len(data_dict['kepid']))
    # tces = []
    # for i in range(len(data_dict['kepid'])):
    #     tces.append('{}_{}'.format(data_dict['kepid'][i], data_dict['tce_n'][i]))
    # unique_tces = np.unique(tces)
    # print(len(unique_tces), len(data_dict['kepid']))
    #
    # # create new tfrecord files by filtering source tfrecord files
    # filt_data = np.load('').item()
    # save_dir = ''
    # filt_dataset_name = ''
    # add_params = {'coupled': True}
    # for tfrec in os.list_dir(tfrec_dir):
    #     create_filtered_tfrecord(os.path.join(tfrec_dir, tfrec), save_dir, filt, append_name=filt_dataset_name,
    #     kw_filt_args=add_params)

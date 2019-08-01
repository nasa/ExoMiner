"""
Custom estimator built using the estimator API from TensorFlow.

# TODO: update model to use Keras API instead of the Estimator API?
        remove centr_flag (only there for backward compatibility)

"""

# 3rd party
import tensorflow as tf
import copy
import operator
import os
import tempfile
import _pickle as pickle
import numpy as np


class InputFn(object):
    """Class that acts as a callable input function for the Estimator."""

    def __init__(self, file_pattern, batch_size, mode, label_map, features_set=None, centr_flag=False, filter_data=None):
        """Initializes the input function.

        :param file_pattern: File pattern matching input TFRecord files, e.g. "/tmp/train-?????-of-00100".
        May also be a comma-separated list of file patterns.
        :param batch_size: int, batch size
        :param mode: A tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        :param label_map:
        :param features_set: dict of the features to be extracted from the dataset, the key is the feature name and the
        value is a dict with the dimension 'dim' of the feature and its data type 'dtype'
        (can the dimension and data type be inferred from the tensors in the dataset?)
        :param filter_data:
        :return:
        """

        self._file_pattern = file_pattern
        self._mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        if features_set is None:
            # by default, assume there are global and local views
            self.features_set = {'global_view': {'dim': 2001, 'dtype': tf.float32},
                                'local_view': {'dim': 201, 'dtype': tf.float32}}
            if self.centr_flag:
                features_set['global_view_centr'] = {'dim': 2001, 'dtype': tf.float32}
                features_set['local_view_centr'] = {'dim': 201, 'dtype': tf.float32}
        else:
            self.features_set = features_set
        self.filter_data = filter_data

        self.centr_flag = centr_flag

        # if self._mode == tf.estimator.ModeKeys.PREDICT and pred_feat is not None:
        #     self.pred_feat = pred_feat

    def __call__(self, config, params):
        """Builds the input pipeline.

        :param config: dict, parameters and hyperparameters for the model
        :param params:
        :return:
            a tf.data.Dataset with features and labels
        """

        reverse_time_series_prob = 0.5 if self._mode == tf.estimator.ModeKeys.TRAIN else 0

        table_initializer = tf.contrib.lookup.KeyValueTensorInitializer(
            keys=list(self.label_map.keys()),
            values=list(self.label_map.values()),
            key_dtype=tf.string,
            value_dtype=tf.int32)

        label_to_id = tf.contrib.lookup.HashTable(table_initializer, default_value=-1)

        include_labels = (self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL])

        file_patterns = self._file_pattern.split(",")
        filenames = []
        for p in file_patterns:
            matches = tf.gfile.Glob(p)
            if not matches:
                raise ValueError("Found no input files matching {}".format(p))
            filenames.extend(matches)

        tf.logging.info("Building input pipeline from %d files matching patterns: %s", len(filenames), file_patterns)

        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors.

            :param serialized_example: a single tf.Example
            :return:
                tuple, feature and label tensors
            """

            # data_fields = {feature_name: tf.FixedLenFeature([length], tf.float32)
            #                for feature_name, length in feature_size_dict.items()}

            # IT MAKES THE CENTR_FLAG IRRELEVANT
            data_fields = {feature_name: tf.FixedLenFeature([feature_info['dim']], feature_info['dtype'])
                           for feature_name, feature_info in self.features_set.items()}

            # get labels if in TRAIN or PREDICT mode
            if include_labels:
                data_fields['av_training_set'] = tf.FixedLenFeature([], tf.string)

            # initialize filtering data fields
            if self.filter_data is not None:
                data_fields['kepid'] = tf.FixedLenFeature([], tf.int64)
                data_fields['tce_plnt_num'] = tf.FixedLenFeature([], tf.int64)

            # Parse the features.
            parsed_features = tf.parse_single_example(serialized_example, features=data_fields)

            # data augmentation - time axis flipping
            if reverse_time_series_prob > 0:
                # Randomly reverse time series features with probability
                # reverse_time_series_prob.
                should_reverse = tf.less(
                    tf.random_uniform([], 0, 1),
                    reverse_time_series_prob,
                    name="should_reverse")

            output = {'time_series_features': {}}
            if self.filter_data is not None:
                output['filt_features'] = {}
            label_id = tf.cast(0, dtype=tf.int32)
            for feature_name, value in parsed_features.items():

                # label
                if include_labels and feature_name == 'av_training_set':
                    label_id = label_to_id.lookup(value)
                    # Ensure that the label_id is non negative to verify a successful hash map lookup.
                    assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.cast(0, dtype=tf.int32)),
                                                   ["Unknown label string:", value])
                    with tf.control_dependencies([assert_known_label]):
                        label_id = tf.identity(label_id)

                # get filtering features
                elif self.filter_data is not None and feature_name == 'kepid':
                    output['filt_features']['kepid'] = value
                elif self.filter_data is not None and feature_name == 'tce_plnt_num':
                    output['filt_features']['tce_n'] = value

                # features
                else:  # input_config.features[feature_name].is_time_series:
                    # Possibly reverse.
                    if reverse_time_series_prob > 0:
                        # pylint:disable=cell-var-from-loop
                        value = tf.cond(should_reverse, lambda: tf.reverse(value, axis=[0]),
                                        lambda: tf.identity(value))

                    output['time_series_features'][feature_name] = value

            return output, label_id

        def filt_func(x, y):
            """ Utility function used to filter examples from the dataset based on their Kepler ID + TCE planet number

            :param x: feature tensor
            :param y: label tensor
            :return:
                boolean tensor, True for valid examples, False otherwise
            """

            z1 = tf.as_string(x['filt_features']['kepid'])
            z_ = tf.convert_to_tensor('_')
            z2 = tf.as_string(x['filt_features']['tce_n'])
            zj = tf.strings.join([z1, z_, z2])

            return tf.math.reduce_any(tf.math.equal(zj, tf.convert_to_tensor(self.filter_data['kepid+tce_n'])))

        def get_features_and_labels(x, y):
            """ Utility function used to remove the features used to filter the dataset.

            :param x: feature tensor
            :param y: label tensor
            :return:
                tuple, dict with features tensor, and label tensor
            """

            return {'time_series_features': x['time_series_features']}, y

        # create filename dataset based on the list of tfrecords filepaths
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

        # map a TFRecordDataset object to each tfrecord filepath
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

        # shuffle the dataset if training or evaluating
        if self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            dataset = dataset.shuffle(1024)

        # ????????
        dataset = dataset.repeat(1)

        # map the example parser across the tfrecords dataset to extract the examples
        dataset = dataset.map(_example_parser, num_parallel_calls=4)

        # filter the dataset based on the filtering features
        if self.filter_data is not None:
            dataset = dataset.filter(filt_func)

            # remove the filtering features from the dataset
            dataset = dataset.map(get_features_and_labels)

        # creates batches by combining consecutive elements
        dataset = dataset.batch(self.batch_size)

        # prefetches batches determined by the buffer size chosen - is it needed?
        dataset = dataset.prefetch(max(1, int(256 / self.batch_size)))  # Better to set at None?

        return dataset


class ModelFn(object):
    """Class that acts as a callable model function for the Estimator."""

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
        # metrics = self.create_metrics(model)  # None if mode == tf.estimator.ModeKeys.PREDICT else self.create_metrics(model)
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
        labels: Tensor with shape [batch_size].
        predictions: Tensor with shape [batch_size, output_dim].
        weights: Tensor with shape [batch_size].
        batch_losses: Tensor with shape [batch_size].
        output_dim: Dimension of model output

        Returns:
        A dictionary {metric_name: (metric_value, update_op).
        """
        metrics = {}

        assert len(model.predictions.shape) == 2
        if model.output_size == 1:
            assert model.predictions.shape[1] == 1
            predictions = tf.squeeze(model.predictions, axis=[1])
            predicted_labels = tf.cast(tf.greater(predictions, 0.5), name="predicted_labels", dtype=tf.int32)
        else:
            predicted_labels = tf.argmax(model.predictions, 1, name="predicted_labels", output_type=tf.int32)
            predictions = model.predictions
            # labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)

        if model.config['force_softmax']:
            labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)
        else:
            labels = model.labels

        metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predicted_labels)
        metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predicted_labels)
        metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predicted_labels)

        metrics['prec thr'] = tf.metrics.precision_at_thresholds(model.labels, predictions,
                                                         np.linspace(0, 1, num=1000, endpoint=True, dtype='float32'))
        metrics['rec thr'] = tf.metrics.recall_at_thresholds(model.labels, predictions,
                                                     np.linspace(0, 1, num=1000, endpoint=True, dtype='float32'))

        def _metric_variable(name, shape, dtype):
            """Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
            return tf.get_variable(
                name,
                initializer=tf.zeros(shape, dtype),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])

        def _count_condition(name, labels_value, predicted_value):
            """Creates a counter for given values of predictions and labels."""
            count = _metric_variable(name, [], tf.float32)
            is_equal = tf.cast(tf.logical_and(tf.equal(labels, labels_value),
                                                  tf.equal(predicted_labels, predicted_value)), dtype=tf.float32)
            update_op = tf.assign_add(count, tf.reduce_sum(tf.ones_like(model.labels, dtype=tf.float32) * is_equal))
            return count.read_value(), update_op

        # Confusion matrix metrics.
        num_labels = 2 if not model.config['multi_class'] else max(model.config['label_map'].values()) + 1
        for label in range(num_labels):
            for pred_label in range(num_labels):
                metric_name = "label_{}_pred_{}".format(label, pred_label)
                metrics[metric_name] = _count_condition(metric_name, labels_value=label, predicted_value=pred_label)

        if not model.config['multi_class']:
            labels = tf.cast(labels, dtype=tf.bool)
            metrics["roc auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                                summation_method='careful_interpolation', curve='ROC')
            metrics["pr auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                               summation_method='careful_interpolation', curve='PR')

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
            # optimizer = tf.train.AdamOptimizer(learning_rate=model.config.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
            optimizer = tf.train.AdamOptimizer(learning_rate=model.config['lr'], beta1=0.9, beta2=0.999, epsilon=1e-8)
        else:
            # optimizer = tf.train.MomentumOptimizer(model.config.lr, model.config.sgd_momentum,
            #                                        # use_nesterov=True
            #                                        )
            optimizer = tf.train.MomentumOptimizer(model.config['lr'], model.config['sgd_momentum'],
                                                   # use_nesterov=True
                                                   )

        return tf.contrib.training.create_train_op(total_loss=model.total_loss, optimizer=optimizer)


class CNN1dModel(object):

    def __init__(self, features, labels, config, mode):
        """ Initializes the CNN 1D model.

        :param features: feature tensor
        :param labels: label tensor
        :param config: dict, model configuration for its parameters and hyperparameters
        :param mode: a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        """

        self.config = config
        self.mode = mode
        self.time_series_features = features['time_series_features']

        self.is_training = None
        self.labels = labels
        self.weights = None
        self.logits = None
        self.predictions = None
        self.batch_losses = None
        self.total_loss = None

        if self.config['multi_class'] or (not self.config['multi_class'] and self.config['force_softmax']):
            self.output_size = max(config['label_map'].values()) + 1
        else:
            self.output_size = 1

        self.ce_weights = tf.constant(self.config['ce_weights'], dtype=tf.float32)

        self.build()

    def build_cnn_layers(self):
        """ Builds the conv columns/branches.

        :return:
            cnn_layers, dict with the different conv columns
        """

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' else None

        cnn_layers = {}
        for view in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s' % view):

                # add the different time series for a view as channels
                net = tf.stack([feature for feature_name, feature in self.time_series_features.items()
                                if view in feature_name], -1)

                # if 'global_view_centr' in self.time_series_features:
                #     net = tf.stack([self.time_series_features[view], self.time_series_features[view + '_centr']], -1)
                # else:
                #     net = tf.expand_dims(self.time_series_features[view], -1)  # [batch, length, channels]

                n_blocks = self.config[config_mapper['blocks'][view]]
                pool_size = self.config[config_mapper['pool_size'][view]]

                for conv_block_i in range(n_blocks):
                    num_filters = self.config['init_conv_filters'] * (2 ** conv_block_i)
                    kwargs = {'inputs': net,
                              'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
                              'padding': "same"}

                    # net = tf.keras.layers.Conv1D(**kwargs)
                    net = tf.layers.conv1d(**kwargs)
                    net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' else tf.nn.relu(net)

                    for seq_conv_block_i in range(self.config['conv_ls_per_block'] - 1):
                        # net = tf.keras.layers.Conv1D(**kwargs)
                        net = tf.layers.conv1d(**kwargs)
                        net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' \
                            else tf.nn.relu(net)

                    # net = tf.keras.layers.MaxPooling1D(inputs=net, pool_size=pool_size,
                    # strides=self.config['pool_stride'])
                    net = tf.layers.max_pooling1d(inputs=net, pool_size=pool_size, strides=self.config['pool_stride'])

                    if self.config['batch_norm']:
                        net = tf.layers.batch_normalization(inputs=net)

                # Flatten
                net.get_shape().assert_has_rank(3)
                net_shape = net.get_shape().as_list()
                output_dim = net_shape[1] * net_shape[2]
                net = tf.reshape(net, [-1, output_dim], name="flatten")

            cnn_layers[view] = net

        return cnn_layers

    # @staticmethod
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
            pre_logits_concat = tf.concat([layer[1] for layer in time_series_hidden_layers],
                                          axis=1, name="pre_logits_concat")

        # concatenate stellar params
        if 'stellar_params' in self.time_series_features:
            pre_logits_concat = tf.concat([pre_logits_concat, self.time_series_features['stellar_params']], axis=1,
                                          name="pre_logits_concat")

        return pre_logits_concat

    def build_fc_layers(self, net):
        """ Builds the FC layers

        :param net: model upstream the FC layers
        :return:
        """

        with tf.variable_scope('FcNet'):

            for fc_layer_i in range(self.config['num_fc_layers'] - 1):
                # fc_neurons = self.config.init_fc_neurons / (2 ** fc_layer_i)
                fc_neurons = self.config['init_fc_neurons']
                if self.config['decay_rate'] is not None:
                    # net = tf.keras.layers.Dense(inputs=net, units=fc_neurons,
                    #                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config['decay_rate']))
                    net = tf.layers.dense(inputs=net, units=fc_neurons,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                    self.config['decay_rate']))
                else:
                    # net = tf.keras.layers.Dense(inputs=net, units=fc_neurons)
                    net = tf.layers.dense(inputs=net, units=fc_neurons)

                net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' else tf.nn.relu(net)

                # net = tf.keras.layers.Dropout(net, self.config['dropout_rate'], training=self.is_training)
                net = tf.layers.dropout(net, self.config['dropout_rate'], training=self.is_training)

            # ?????????????
            # tf.identity(net, "final")

            # create output FC layer
            # logits = tf.keras.layers.Dense(inputs=net, units=self.output_size, name="logits")
            logits = tf.layers.dense(inputs=net, units=self.output_size, name="logits")

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

        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.cast(self.labels, dtype=tf.int32)))

        if self.output_size == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, dtype=tf.float32),
                                                                   logits=tf.squeeze(self.logits, [1]))
        else:
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

        # Compute the weighted mean cross entropy loss and add it to the LOSSES collection
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES collection (e.g. regularization losses)
        self.total_loss = tf.losses.get_total_loss()

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


def get_ce_weights(label_map, tfrec_dir):
    """ Compute class cross-entropy weights based on the amount of labels for each class.

    :param label_map: dict, map between class name and integer value
    :param tfrec_dir: str, filepath to directory with the tfrecords
    :return:
        ce_weights: list, weight for each class (class 0, class 1, ...)
        n_train: int, number of training samples
    """

    max_label_val = max(label_map.values())

    # takes into account samples from the validation and train tfrecords???
    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir)
                 if not (file.startswith('test') or file.startswith('predict'))]
    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir)
                 if file.startswith('train')]

    label_vec, example = [], tf.train.Example()
    # n_train = 0
    # n_samples = {'train': 0, 'val': 0, 'test': 0, 'predict': 0}
    n_samples = {'train': 0, 'val': 0}
    for file in filenames:
        # upd_c = True if file.split('/')[-1].startswith('train') else False  # update counter only if train files
        file_dataset = file.split('/')[-1]
        if 'train' in file_dataset:
            dataset = 'train'
        elif 'val' in file_dataset:
            dataset = 'val'
        # elif 'test' in file_dataset:
        #     dataset = 'test'
        # elif 'predict' in file_dataset:
        #     dataset = 'predict'

        record_iterator = tf.python_io.tf_record_iterator(path=file)
        try:
            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)
                label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
                label_vec.append(label_map[label])

                n_samples[dataset] += 1
                # if upd_c:
                #     n_train += 1

        except tf.errors.DataLossError as err:
            print("Oops: " + str(err))

    # count instances for each class based on their indexes
    label_counts = [label_vec.count(category) for category in range(max_label_val + 1)]

    # give more weight to classes with less instances
    ce_weights = [max(label_counts) / max(count_i, 1e-7) for count_i in label_counts]

    print('Train and validation samples: {}, {}'.format(n_samples['train'], n_samples['val']))

    return ce_weights, n_samples['train']  # n_train  # , 'global_view_centr' in example.features.feature


def picklesave(path, savedict):

    p = pickle.Pickler(open(path, "wb+"))
    p.fast = True
    p.dump(savedict)


def get_num_samples(label_map, tfrec_dir, datasets):
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

    # label_vec, example = [], tf.train.Example()

    for file in filenames:

        file_dataset = file.split('/')[-1]
        curr_dataset = datasets[np.where([dataset in file_dataset for dataset in datasets])[0][0]]

        record_iterator = tf.python_io.tf_record_iterator(path=file)
        try:
            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)
                try:
                    label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
                except ValueError as e:
                    print('No label field found on the example. Ignoring it.')
                    print('Error output:', e)

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
    #       right now, it assumes that all examples have the same fields
    #       deal with errors
    """

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

        if 'kepid' in union_fields:
            datum['kepid'] = example.features.feature['kepid'].int64_list.value[0]

        if 'tce_n' in union_fields:
            datum['tce_n'] = example.features.feature['tce_plnt_num'].int64_list.value[0]

        if 'tce_period' in union_fields:
            datum['tce_period'] = example.features.feature['tce_period'].float_list.value[0]

        if 'tce_duration' in union_fields:
            datum['tce_duration'] = example.features.feature['tce_duration'].float_list.value[0]

        if 'epoch' in union_fields:
            datum['epoch'] = example.features.feature['tce_time0bk'].float_list.value[0]

        if 'MES' in union_fields:
            datum['MES'] = example.features.feature['mes'].float_list.value[0]

        if 'global_view' in union_fields:
            datum['global_view'] = example.features.feature['global_view'].float_list.value

        if 'local_view' in union_fields:
            datum['local_view'] = example.features.feature['local_view'].float_list.value

        if 'global_view_centr' in union_fields:
            datum['global_view_centr'] = example.features.feature['global_view_centr'].float_list.value

        if 'local_view_centr' in union_fields:
            datum['local_view_centr'] = example.features.feature['local_view_centr'].float_list.value

        # filtering
        if filt is not None:

            if 'label' in filt.keys() and datum['label'] not in filt['label'].values:
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
    filt_idx = get_data_from_tfrecord(src_tfrecord, [], label_map=None, filt=filt, **kw_filt_args)

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
    tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/Pre_processor/tfrecords/tfrecord_dr25_manual_2d_few180k_keplernonwhitened'  # '/data5/tess_project/Data/tfrecords/180k_tfrecord'
    # nsamples = get_num_samples(label_map, tfrec_dir, ['predict'])
    # print(nsamples)

    # assert that there are no repeated examples in the datasets based on KeplerID and TCE number
    # multi_class = False
    # satellite = 'kepler'
    # label_map = label_map[satellite][multi_class]
    # tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_dr25_manual_2dkeplerwhitened_2001-201'
    tfrecords = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir)]
    data_fields = ['kepid', 'tce_period', 'tce_duration', 'global_view', 'local_view', 'MES', 'epoch', 'label']
    data_dict = get_data_from_tfrecords(tfrecords, data_fields, label_map=None, filt=None)
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
    #     create_filtered_tfrecord(os.path.join(tfrec_dir, tfrec), save_dir, filt, append_name=filt_dataset_name, kw_filt_args=add_params)


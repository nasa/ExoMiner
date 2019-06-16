import tensorflow as tf
import copy
import operator
import os
import tempfile
import _pickle as pickle
import numpy as np


class InputFn(object):
    """Class that acts as a callable input function for Estimator train / eval."""

    def __init__(self, file_pattern, batch_size, mode, label_map, centr_flag=False):
        """Initializes the input function.

        Args:
            file_pattern: File pattern matching input TFRecord files, e.g.
            "/tmp/train-?????-of-00100". May also be a comma-separated list of file patterns.
            mode: A tf.estimator.ModeKeys.
        """

        self._file_pattern = file_pattern
        self._mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        self.centr_flag = centr_flag

    def __call__(self, config, params):
        """Builds the input pipeline."""

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

        # filenames = filenames[:1]  # for prototyping

        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors."""

            feature_size_dict = {'global_view': 2001, 'local_view': 201}

            if self.centr_flag:
                feature_size_dict = {**feature_size_dict, 'global_view_centr': 2001, 'local_view_centr': 201}

            data_fields = {feature_name: tf.FixedLenFeature([length], tf.float32)
                           for feature_name, length in feature_size_dict.items()}
            if include_labels:
                data_fields['av_training_set'] = tf.FixedLenFeature([], tf.string)

            # Parse the features.
            parsed_features = tf.parse_single_example(serialized_example, features=data_fields)

            if reverse_time_series_prob > 0:
                # Randomly reverse time series features with probability
                # reverse_time_series_prob.
                should_reverse = tf.less(
                    tf.random_uniform([], 0, 1),
                    reverse_time_series_prob,
                    name="should_reverse")

            output = {'time_series_features': {}}
            # label_id = tf.to_int32(0)
            for feature_name, value in parsed_features.items():
                if include_labels and feature_name == 'av_training_set':
                    label_id = label_to_id.lookup(value)
                    # Ensure that the label_id is non negative to verify a successful hash map lookup.
                    assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.to_int32(0)),
                                                   ["Unknown label string:", value])
                    with tf.control_dependencies([assert_known_label]):
                        label_id = tf.identity(label_id)

                else:  # input_config.features[feature_name].is_time_series:
                    # Possibly reverse.
                    if reverse_time_series_prob > 0:
                        # pylint:disable=cell-var-from-loop
                        value = tf.cond(should_reverse, lambda: tf.reverse(value, axis=[0]),
                                        lambda: tf.identity(value))

                    output['time_series_features'][feature_name] = value

            # return output
            return output, label_id

        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat(1)

        dataset = dataset.map(_example_parser, num_parallel_calls=4)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(max(1, int(256 / self.batch_size)))  # Better to set at None?

        return dataset


class ModelFn(object):
    def __init__(self, model_class, config):
        self._model_class = model_class
        self._base_config = config

    def __call__(self, features, labels, mode):
        config = copy.deepcopy(self._base_config)
        # initialize model instance of the class
        model = self._model_class(features, labels, config, mode)

        train_op = self.create_train_op(model) if mode == tf.estimator.ModeKeys.TRAIN else None

        # metrics = self.create_metrics(model)  # None if mode == tf.estimator.ModeKeys.PREDICT else self.create_metrics(model)
        metrics = None if mode == tf.estimator.ModeKeys.PREDICT else self.create_metrics(model)

        logging_hook = None
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
            predicted_labels = tf.to_int32(tf.greater(predictions, 0.5), name="predicted_labels")
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
            is_equal = tf.to_float(tf.logical_and(tf.equal(labels, labels_value),
                                                  tf.equal(predicted_labels, predicted_value)))
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

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        weight_initializer = tf.keras.initializers.he_normal() if self.config['weight_initializer'] == 'he' else None

        cnn_layers = {}
        for name in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s' % name):

                if 'global_view_centr' in self.time_series_features:
                    net = tf.stack([self.time_series_features[name], self.time_series_features[name + '_centr']], -1)
                else:
                    net = tf.expand_dims(self.time_series_features[name], -1)  # [batch, length, channels]

                n_blocks = self.config[config_mapper['blocks'][name]]
                pool_size = self.config[config_mapper['pool_size'][name]]
                # n_blocks = getattr(self.config, config_mapper['blocks'][name])
                # pool_size = getattr(self.config, config_mapper['pool_size'][name])

                for conv_block_i in range(n_blocks):
                    num_filters = self.config['init_conv_filters'] * (2 ** conv_block_i)
                    kwargs = {'inputs': net,
                              'filters': num_filters,
                              'kernel_initializer': weight_initializer,
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
                              'padding': "same"}

                    net = tf.layers.conv1d(**kwargs)
                    # net = tf.nn.leaky_relu(net, alpha=0.01)
                    net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' else tf.nn.relu(net)
                    # net = tf.nn.relu(net)

                    for seq_conv_block_i in range(self.config['conv_ls_per_block'] - 1):
                        net = tf.layers.conv1d(**kwargs)
                        net = tf.nn.leaky_relu(net, alpha=0.01)
                        net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' else tf.nn.relu(net)
                        # net = tf.nn.relu(net)

                    net = tf.layers.max_pooling1d(inputs=net, pool_size=pool_size, strides=self.config['pool_stride'])

                    if self.config['batch_norm']:
                        net = tf.layers.batch_normalization(inputs=net)

                # Flatten
                net.get_shape().assert_has_rank(3)
                net_shape = net.get_shape().as_list()
                output_dim = net_shape[1] * net_shape[2]
                net = tf.reshape(net, [-1, output_dim], name="flatten")

            cnn_layers[name] = net

        return cnn_layers

    @staticmethod
    def connect_segments(cnn_layers):

        # Sort the hidden layers by name because the order of dictionary items is
        # nondeterministic between invocations of Python.
        time_series_hidden_layers = sorted(cnn_layers.items(), key=operator.itemgetter(0))

        # Concatenate the hidden layers.
        if len(time_series_hidden_layers) == 1:
            pre_logits_concat = time_series_hidden_layers[0][1]
        else:
            pre_logits_concat = tf.concat([layer[1] for layer in time_series_hidden_layers],
                                          axis=1, name="pre_logits_concat")

        return pre_logits_concat

    def build_fc_layers(self, net):
        with tf.variable_scope('FcNet'):
            for fc_layer_i in range(self.config['num_fc_layers'] - 1):
                # fc_neurons = self.config.init_fc_neurons / (2 ** fc_layer_i)
                fc_neurons = self.config['init_fc_neurons']
                if self.config['decay_rate'] is not None:
                    net = tf.layers.dense(inputs=net, units=fc_neurons,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config['decay_rate']))
                else:
                    net = tf.layers.dense(inputs=net, units=fc_neurons)
                # net = tf.nn.leaky_relu(net, alpha=0.01)
                net = tf.nn.leaky_relu(net, alpha=0.01) if self.config['non_lin_fn'] == 'prelu' else tf.nn.relu(net)
                # net = tf.nn.relu(net)
                net = tf.layers.dropout(net, self.config['dropout_rate'], training=self.is_training)

            tf.identity(net, "final")

            logits = tf.layers.dense(inputs=net, units=self.output_size, name="logits")

        self.logits = logits

    def build(self):
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        cnn_layers = self.build_cnn_layers()
        net = self.connect_segments(cnn_layers)
        self.build_fc_layers(net)

        prediction_fn = tf.nn.softmax if self.config['multi_class'] else tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        weights = (1.0 if self.config['satellite'] == 'kepler' and not self.config['use_kepler_ce']
                   else tf.gather(self.ce_weights, tf.to_int32(self.labels)))

        if self.output_size == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.labels),
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
    """Returns a randomly named, non-existing model file folder"""

    def _gen_dir():
        # return os.path.join(parent_dir, 'models/', tempfile.mkdtemp().split('/')[-1])
        return os.path.join(path, tempfile.mkdtemp().split('/')[-1])

    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    model_dir_custom = _gen_dir()

    while os.path.isdir(model_dir_custom):  # try until there is no current directory with that name
        model_dir_custom = _gen_dir()

    return model_dir_custom


def get_ce_weights(label_map, tfrec_dir):

    max_label_val = max(label_map.values())

    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir)
                 if not file.startswith('test')]

    label_vec, example = [], tf.train.Example()
    n_train = 0
    for file in filenames:
        upd_c = True if file.split('/')[-1].startswith('train') else False  # update counter only if train files

        record_iterator = tf.python_io.tf_record_iterator(path=file)
        try:
            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)
                label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
                label_vec.append(label_map[label])

                if upd_c:
                    n_train += 1

        except tf.errors.DataLossError as err:
            print("Oops: " + str(err))

    label_counts = [label_vec.count(category) for category in range(max_label_val + 1)]
    ce_weights = [max(label_counts) / count_i for count_i in label_counts]

    return ce_weights, 'global_view_centr' in example.features.feature, n_train


def picklesave(path, savedict):
    p = pickle.Pickler(open(path, "wb+"))
    p.fast = True
    p.dump(savedict)


def get_labels(tfrecord, label_map):

    labels = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
        labels.append(label_map[label])

        # kepid = example.features.feature['kepid'].int64_list.value[0]
        # tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
        # period = example.features.feature['tce_period'].float_list.value[0]
        # duration = example.features.feature['tce_duration'].float_list.value[0]
        # epoch = example.features.feature['tce_time0bk'].float_list.value[0]
        # MES = example.features.feature['mes'].float_list.value[0]
        # ephem_vec += [{'period': period, 'duration': duration, 'epoch': epoch}]
        # glob_view = example.features.feature['global_view'].float_list.value
        # loc_view = example.features.feature['local_view'].float_list.value
        # glob_view_centr = example.features.feature['global_view_centr'].float_list.value
        # loc_view_centr = example.features.feature['local_view_centr'].float_list.value

        # kepid_vec.append(kepid)
        # glob_vec += [glob_view]
        # loc_vec += [loc_view]
        # glob_centrvec += [glob_view_centr]
        # loc_centrvec += [loc_view_centr]
        # mes_vec += [MES]

    return labels


# def input_fn_pred(features):
#
#     dataset = tf.data.Dataset.from_tensor_slices(features)
#     dataset.repeat(1)
#     dataset = dataset.map(parser)
#
#     return dataset
#
#
# # def parser(localview, globalview, localview_centr, globalview_centr):
# def parser(localview, globalview):
#
#     # output = {"time_series_features": {'local_view': tf.to_float(localview),
#     #                                    'global_view': tf.to_float(globalview),
#     #                                    'global_view_centr': tf.to_float(localview_centr),
#     #                                    'local_view_centr': tf.to_float(globalview_centr)}}
#     output = {"time_series_features": {'local_view': tf.to_float(localview),
#                                        'global_view': tf.to_float(globalview)}}
#     return output

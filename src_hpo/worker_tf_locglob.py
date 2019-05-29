import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import copy
import sys
from os.path import join as pathj
from os.path import isdir#, isfile, dirname
# from os import unlink, listdir
import tempfile
import operator
import shutil

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import numpy as np
# import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# if 'nobackup' in dirname(__file__):
#     # For running on Pleiades without matplotlib fig() error
#     plt.switch_backend('agg')


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
            for feature_name, value in parsed_features.items():
                if include_labels and feature_name == 'av_training_set':
                    label_id = label_to_id.lookup(value)
                    # Ensure that the label_id is nonnegative to verify a successful hash
                    # map lookup.
                    assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.to_int32(0)),
                                                   ["Unknown label string:", value])
                    with tf.control_dependencies([assert_known_label]):
                        label_id = tf.identity(label_id)

                    # We use the plural name "labels" in the output due to batching.
                    # output["labels"] = label_id
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
        model = self._model_class(features, labels, config, mode)

        train_op = self.create_train_op(model) if mode == tf.estimator.ModeKeys.TRAIN else None

        metrics = self.create_metrics(model)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=model.predictions,
                                          loss=model.total_loss,
                                          train_op=train_op,
                                          eval_metric_ops=metrics,
                                          )

    def create_metrics(self, model):
        """ Builds TensorFlow operations to compute model evaluation metrics.

        Args:
        labels: Tensor with shape [batch_size].
        predictions: Tensor with shape [batch_size, output_dim].
        weights: Tensor with shape [batch_size].
        batch_losses: Tensor with shape [batch_size].
        output_dim: Dimension of model output

        Returns:
        A dictionary {metric_name: (metric_value, update_op).
        """

        assert len(model.predictions.shape) == 2
        if not model.multi_class:
            assert model.predictions.shape[1] == 1
            predictions = tf.squeeze(model.predictions, axis=[1])
            predicted_labels = tf.to_int32(tf.greater(predictions, 0.5), name="predicted_labels")
            labels = model.labels
        else:
            predicted_labels = tf.argmax(model.predictions, 1, name="predicted_labels", output_type=tf.int32)
            # labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)
            labels = model.labels

        # evaluation metrics
        acc_op = tf.metrics.accuracy(labels=labels, predictions=predicted_labels)
        prec_op = tf.metrics.precision(labels=labels, predictions=predicted_labels)
        rec_op = tf.metrics.recall(labels=labels, predictions=predicted_labels)
        if not model.multi_class:
            rocauc_op = tf.metrics.auc(labels=model.labels, predictions=predictions, num_thresholds=1000, curve='ROC',
                                       summation_method='careful_interpolation')
            prauc_op = tf.metrics.auc(labels=model.labels, predictions=predictions, num_thresholds=1000, curve='PR',
                                      summation_method='careful_interpolation')
        else:
            rocauc_op, prauc_op = None, None

        return {'accuracy': acc_op, 'precision': prec_op, 'pr auc': prauc_op, 'recall': rec_op, 'roc auc': rocauc_op}

    def create_train_op(self, model):

        if model.config['lr_scheduler'] in ["inv_exp_fast", "inv_exp_slow"]:
            global_step = tf.train.get_global_step()
            # global_step = tf.Variable(0, trainable=False)
            # global_step = tf.train.get_or_create_global_step()

            # decay learning rate every 'decay_epochs' by 'decay_rate'
            decay_rate = 0.5
            decay_epochs = 32 if model.config['lr_scheduler'] == "inv_exp_fast" else 64

            learning_rate = tf.train.exponential_decay(
                learning_rate=model.config['lr'],
                global_step=global_step,
                decay_steps=int(decay_epochs * model.n_train / self._base_config['batch_size']),
                decay_rate=decay_rate,
                staircase=True)

        elif model.config['lr_scheduler'] == "constant":
            learning_rate = model.config['lr']
        else:
            learning_rate = None
            NotImplementedError('Learning rate scheduler not recognized')

        if model.config['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate, model.config['sgd_momentum'],
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
        self.ce_weights = config['ce_weights']
        self.logits = None
        self.predictions = None
        self.batch_losses = None
        self.total_loss = None

        self.multi_class = config['multi_class']
        self.output_size = 1 if not self.multi_class else max(config['label_map'].values()) + 1

        self.build()

    def build_cnn_layers(self):

        config_mapper = {'blocks': {'global_view': 'num_glob_conv_blocks', 'local_view': 'num_loc_conv_blocks'},
                         'pool_size': {'global_view': 'pool_size_glob', 'local_view': 'pool_size_loc'}}

        cnn_layers = {}
        for name in ['global_view', 'local_view']:
            with tf.variable_scope('ConvNet_%s_%s' % (name, self.config['worker_id_custom'])):

                if 'global_view_centr' in self.time_series_features:
                    net = tf.stack([self.time_series_features[name], self.time_series_features[name + '_centr']], -1)

                    # time_series = tf.stack([self.time_series_features[name],
                    #                         self.time_series_features[name + '_centr']], 2)
                    # net = tf.expand_dims(time_series, -2)  # net.shape: [batch, in_height, in_width, in_channels]
                else:
                    net = tf.expand_dims(self.time_series_features[name], -1)  # [batch, length, channels]

                n_blocks = self.config[config_mapper['blocks'][name]]
                pool_size = self.config[config_mapper['pool_size'][name]]

                for conv_block_i in range(n_blocks):
                    num_filters = 2 ** (conv_block_i + self.config['init_conv_filters'])
                    kwargs = {'inputs': net,
                              'filters': num_filters,
                              # 'kernel_initializer': tf.keras.initializers.he_normal(),
                              'kernel_size': self.config['kernel_size'],
                              'strides': self.config['kernel_stride'],
                              'padding': "same"}

                    net = tf.layers.conv1d(**kwargs)
                    net = tf.nn.relu(net)
                    # net = tf.nn.leaky_relu(net, alpha=0.01)

                    for seq_conv_block_i in range(self.config['conv_ls_per_block'] - 1):
                        net = tf.layers.conv1d(**kwargs)
                        net = tf.nn.relu(net)
                        # net = tf.nn.leaky_relu(net, alpha=0.01)

                    net = tf.layers.max_pooling1d(inputs=net, pool_size=pool_size, strides=self.config['pool_stride'])
                    # net = tf.layers.batch_normalization(inputs=net)

                # Flatten
                net.get_shape().assert_has_rank(3)
                net_shape = net.get_shape().as_list()
                output_dim = net_shape[1] * net_shape[2]
                net = tf.reshape(net, [-1, output_dim], name="flatten")

            cnn_layers[name] = net

        return cnn_layers

    def connect_segments(self, cnn_layers):

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
        with tf.variable_scope('FcNet_%s' % self.config['worker_id_custom']):
            for fc_layer_i in range(self.config['num_fc_layers'] - 1):
                # fc_neurons = self.config['init_fc_neurons'] / (2 ** fc_layer_i)
                fc_neurons = self.config['init_fc_neurons']
                net = tf.layers.dense(inputs=net, units=fc_neurons,
                                      activation=tf.nn.relu,
                                      # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config['decay_rate'])
                                      )
                # net = tf.nn.relu(net)
                # net = tf.nn.leaky_relu(net, alpha=0.01)
                if self.config['dropout_rate'] > 0.001:
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

        prediction_fn = tf.nn.softmax if self.multi_class else tf.sigmoid
        self.predictions = prediction_fn(self.logits, name="predictions")

        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

    def build_losses(self):
        if self.ce_weights is None:
            weights = 1.0
        else:
            self.ce_weights = tf.constant(self.ce_weights, dtype=tf.float32)
            weights = tf.gather(self.ce_weights, tf.to_int32(self.labels))

        # if self.binary_classification:
        #     weights = tf.gather(self.ce_weights, tf.to_int32(self.labels))
        #     batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_double(self.labels),
        #                                                            logits=tf.squeeze(tf.to_double(self.logits), [1]))
        # else:
        #     weights = tf.gather(self.ce_weights, tf.argmax(self.labels, axis=1, output_type=tf.int32))
        #     batch_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)

        if not self.multi_class:
            # Binary classification.

            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.labels),
                                                                   logits=tf.squeeze(self.logits, [1]))
        else:
            # Multi-class classification.
            batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

        # Compute the weighted mean cross entropy loss and add it to the LOSSES
        # collection.
        tf.losses.compute_weighted_loss(losses=batch_losses,
                                        weights=weights,
                                        reduction=tf.losses.Reduction.MEAN)

        # Compute the total loss, including any other losses added to the LOSSES
        # collection (e.g. regularization losses).
        total_loss = tf.losses.get_total_loss()

        self.batch_losses = batch_losses
        self.total_loss = total_loss


class TransitClassifier(Worker):
    """ Transit Classifier based on the HPBandster Worker."""

    def __init__(self, config_args, worker_id_custom=1, **kwargs):
        super().__init__(**kwargs)

        self.worker_id_custom = str(worker_id_custom)

        self.models_directory = config_args.models_directory
        self.results_directory = config_args.results_directory

        self.label_map = config_args.label_map
        self.centr_flag = config_args.centr_flag
        self.tfrec_dir = config_args.tfrec_dir
        self.multi_class = config_args.multi_class
        self.label_map = config_args.label_map

        # use of cross-entropy class weights
        if config_args.satellite == 'kepler' and not config_args.use_kepler_ce:
            self.ce_weights = None
        else:
            self.ce_weights = config_args.ce_weights

    @staticmethod
    def get_model_dir(path):
    # def get_model_dir():
        """Returns a randomly named, non-existing model file folder"""

        def _gen_dir():
            # return pathj(dirname(__file__), 'models', tempfile.mkdtemp().split('/')[-1])
            return pathj(path, tempfile.mkdtemp().split('/')[-1])

        model_dir_custom = _gen_dir()

        while isdir(model_dir_custom):
            model_dir_custom = _gen_dir()

        return model_dir_custom

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):

        config['ce_weights'] = self.ce_weights
        config['multi_class'] = self.multi_class
        config['label_map'] = self.label_map
        config['worker_id_custom'] = self.worker_id_custom

        # create folder for model
        model_dir_custom = self.get_model_dir(self.models_directory)  # pathj(dirname(__file__), 'models')  # None  # self.get_model_dir()
        # print('model directory: ', self.models_directory)
        # print('results directory: ', self.results_directory)
        # print('working directory: ', working_directory)

        config_sess = None
        # if 'nobackup' in dirname(__file__):  # if running on Pleiades
        config_sess = tf.ConfigProto(log_device_placement=False)
        # config_sess.gpu_options.force_gpu_compatible = True  # Force pinned memory
        # config_sess.intra_op_parallelism_threads = 1
        # config_sess.gpu_options.visible_device_list = "0"

        # try:

        classifier = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                            config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                          session_config=config_sess),
                                            model_dir=model_dir_custom)

        input_fn_train = InputFn(file_pattern=self.tfrec_dir + '/train*', batch_size=config['batch_size'],
                                 mode=tf.estimator.ModeKeys.TRAIN, label_map=self.label_map, centr_flag=self.centr_flag)

        input_fn_val = InputFn(file_pattern=self.tfrec_dir + '/val*', batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.EVAL, label_map=self.label_map, centr_flag=self.centr_flag)

        input_fn_test = InputFn(file_pattern=self.tfrec_dir + '/test*', batch_size=config['batch_size'],
                                mode=tf.estimator.ModeKeys.EVAL, label_map=self.label_map, centr_flag=self.centr_flag)

        res = {k: [] for k in ['training loss', 'training auc', 'validation loss', 'validation auc',
                               'validation precision', 'validation recall', 'test auc', 'test recall', 'test precision',
                               'epochs', 'test loss', 'test accuracy', 'validation accuracy']}
        for epoch_i in range(int(budget)):
            # train model
            printstr = "Training worker %s: Epoch %d of %d" % (self.worker_id_custom, epoch_i + 1, int(budget))
            print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')
            sys.stdout.flush()
            _ = classifier.train(input_fn_train)
            # evaluate model on the training set
            res_train = classifier.evaluate(input_fn_train)
            res['training loss'].append(res_train['loss'])
            res['training auc'].append(res_train['roc auc'])
            res['epochs'].append(epoch_i + 1)

            # evaluate model on the validation set
            res_val = classifier.evaluate(input_fn_val)
            res['validation loss'].append(res_val['loss'])
            res['validation accuracy'].append(res_val['accuracy'])
            res['validation auc'].append(res_val['roc auc'])
            res['validation precision'].append(res_val['precision'])
            res['validation recall'].append(res_val['recall'])

        # res['validation tpr'] = res_val['tp'] / (res_val['tp'] + res_val['fn'])
        # res['validation fpr'] = res_val['fp'] / (res_val['fp'] + res_val['tn'])

            # evaluate model on the test set
            res_test = classifier.evaluate(input_fn_test)
            res['test loss'].append(res_test['loss'])
            res['test accuracy'].append(res_test['accuracy'])
            res['test auc'].append(res_test['roc auc'])
            res['test precision'].append(res_test['precision'])
            res['test recall'].append(res_test['recall'])
        # res['test tpr'] = res_test['tp'] / (res_test['tp'] + res_test['fn'])
        # res['test fpr'] = res_test['fp'] / (res_test['fp'] + res_test['tn'])

        # delete saved model data but keep the folder so the random string is not used again (work on this...)
        # or delete model's folder and its contents
        self.del_savedmodel(model_dir_custom)

        # draw loss and evaluation metric plots for the model on this given budget
        self.draw_plots(res, config_id)

        # select which epoch to report from
        ep = -1  # np.argmax(res['validation auc'])
        res_hpo = {'loss': 1 - float(res_val['roc auc']),  # choose metric/loss to optimize
                   'info': {'validation loss': float(res['validation loss'][ep]),
                            'validation accuracy': float(res['validation accuracy'][ep]),
                            'validation precision': float(res['validation precision'][ep]),
                            'validation recall': float(res['validation recall'][ep]),
                            'validation auc': float(res['validation auc'][ep]),
                            'test loss': float(res['test loss'][ep]),
                            'test accuracy': float(res['test accuracy'][ep]),
                            'test precision': float(res['test precision'][ep]),
                            'test recall': float(res['test recall'][ep]),
                            'test auc': float(res['test auc'][ep])},
                   'model subfolder': model_dir_custom.split('/')[-1]}
        # res_hpo = {'loss': 1 - float(res_val['roc auc']),
        #            'info': {'validation loss': float(res_val['loss']),
        #                     'validation accuracy': float(res_val['accuracy']),
        #                     'validation precision': float(res_val['precision']),
        #                     'validation recall': float(res_val['recall']),
        #                     'validation auc': float(res_val['recall']),
        #                     'test loss': float(res_val['roc auc']),
        #                     'test accuracy': float(res_test['accuracy']),
        #                     'test precision': float(res_test['precision']),
        #                     'test recall': float(res_test['recall']),
        #                     'test auc': float(res_test['roc auc'])},
        #            'model subfolder': model_dir_custom.split('/')[-1]}

        print('#' * 100)
        for k in res_hpo:
            if k != 'info':
                print(k + ': ', res_hpo[k])
            else:
                for l in res_hpo[k]:
                    print(l + ': ', res_hpo[k][l])
        print('#' * 100)
        sys.stdout.flush()

        return (res_hpo)

        # except Exception as e:
        #     print('*' * 100)
        #     print('error: ', e)
        #     print('*' * 100)
        #     self.del_savedmodel(model_dir_custom)
        #     raise ValueError('ERRRORRRR!')

    def del_savedmodel(self, savedir):
        """ Delete data pertaining to the model trained and evaluated.

        :param savedir: str, filepath of the folder where the data was stored.
        :return:
        """

        # removes model's folder and its contents
        shutil.rmtree(savedir, ignore_errors=True)

        # removes model's folder contents
        # for content in listdir(savedir):
        #     fp = pathj(savedir, content)
        #     if isfile(fp):  # delete file
        #         unlink(fp)
        #     elif isdir(fp):  # delete folder
        #         shutil.rmtree(fp, ignore_errors=True)

    def draw_plots(self, res, config_id):
        """ Draw loss and evaluation metric plots.

        :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
        for the test set.
        :param config_id: tuple, config id
        :return:
        """

        res['epochs'] = np.array(res['epochs'], dtype='int')
        max_val_auc, ep_idx = np.max(res['validation auc']), np.argmax(res['validation auc'])

        # Loss and evaluation metric plots
        f, ax = plt.subplots(1, 2)
        ax[0].plot(res['epochs'], res['training loss'], label='Training')
        ax[0].plot(res['epochs'], res['validation loss'], label='Validation', color='r')
        # min_val_loss, ep_idx = np.min(res['validation loss']), np.argmin(res['validation loss'])
        ax[0].scatter(res['epochs'][ep_idx], res['validation loss'][ep_idx], c='r')
        ax[0].scatter(res['epochs'][ep_idx], res['test loss'][ep_idx], c='k', label='Test')
        ax[0].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[0].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Categorical cross-entropy\nVal/Test %.4f/%.4f' % (res['validation loss'][ep_idx],
                                                                           res['test loss'][ep_idx]))
        ax[0].legend(loc="upper right")
        ax[0].grid('on')
        ax[1].plot(res['epochs'], res['training auc'], label='Training')
        ax[1].plot(res['epochs'], res['validation auc'], label='Validation', color='r')
        # max_val_auc, ep_idx = np.max(res['validation auc']), np.argmax(res['validation auc'])
        ax[1].scatter(res['epochs'][ep_idx], max_val_auc, c='r')
        ax[1].scatter(res['epochs'][ep_idx], res['test auc'][ep_idx], label='Test', c='k')
        ax[1].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[1].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        ax[1].set_ylim([0.85, 1.05])
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('AUC')
        ax[1].set_title('Evaluation Metric\nVal/Test %.4f/%.4f' % (max_val_auc, res['test auc'][ep_idx]))
        ax[1].legend(loc="lower right")
        ax[1].grid('on')
        f.suptitle('Config {}|Budget = {:.0f}(Best val:{:.0f})'.format(config_id, res['epochs'][-1],
                                                                       res['epochs'][ep_idx]))
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        # f.savefig(model_dir + '/plotseval_budget%.0f.png' % res['epochs'][-1])
        f.savefig(self.results_directory + '/plotseval_{}budget{:.0f}.png'.format(config_id, res['epochs'][-1]))

        # Precision and Recall plots
        f, ax = plt.subplots()
        ax.plot(res['epochs'], res['validation precision'], color='r', label='Val. Pre.')
        ax.plot(res['epochs'], res['validation recall'], color='r', label='Val. Rec.', linestyle='--')
        ax.plot(res['epochs'], res['test precision'], color='k', label='Test Pre.')
        ax.plot(res['epochs'], res['test recall'], color='k', label='Test Rec.', linestyle='--')
        ax.set_ylabel('Metric value')
        ax.set_xlabel('Epochs')
        ax.set_xlim([0.0, res['epochs'][-1] + 1])
        # ax.set_ylim([0, 1])
        ax.set_ylim([0.75, 1])
        ax.legend(loc="lower right")
        ax.grid('on')
        ax.set_title('Precision and Recall')
        f.suptitle('Config {}|Budget = {:.0f}(Best val:{:.0f})'.format(config_id, res['epochs'][-1],
                                                                       res['epochs'][ep_idx]))
        f.savefig(self.results_directory + '/prec_rec_{}budget{:.0f}.png'.format(config_id, res['epochs'][-1]))

        # TODO: plot ROC curve; need tpr and fpr
        # # ROC curve
        # f, ax = plt.subplots()
        # ax.plot(res['validation tpr'], res['validation fpr'], color='r', label='Validation')
        # ax.plot(res['test tpr'], res['test fpr'], color='k', label='Test')
        # ax.set_ylabel('TPR')
        # ax.set_xlabel('FPR')
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
        # ax.legend(loc="lower right")
        # ax.set_title('ROC')
        # f.suptitle('Config {}|Budget = {:.0f}'.format(config_id, res['epochs'][-1]))
        # f.savefig(save_path + '/roc_{}budget{:.0f}.png'.format(config_id, res['epochs'][-1]))

    @staticmethod
    def get_configspace():
        """ Build the hyperparameter configuration space

        :return: ConfigurationsSpace-Object
        """
        config_space = CS.ConfigurationSpace()

        # use_softmax = CSH.CategoricalHyperparameter('use_softmax', [True, False])

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # batch_norm = CSH.CategoricalHyperparameter('batch_norm', [True, False])
        # non_lin_fn = CSH.CategoricalHyperparameter('non_lin_fn', ['relu', 'prelu'])
        # weight_initializer = CSH.CategoricalHyperparameter('weight_initializer', ['he', 'glorot'])

        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp_fast', 'inv_exp_slow'])
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.001, upper=0.99, default_value=0.9,
                                                      log=True)
        # batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=4, upper=9, default_value=6)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64, 128, 256], default_value=32)
        # dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.7, default_value=0.2,
        #                                               log=True)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.7, log=True)
        # l2_regularizer = CSH.CategoricalHyperparameter('l2_regularizer', [True, False])
        # l2_decay_rate = CSH.UniformFloatHyperparameter('decay_rate', lower=1e-4, upper=1e-1, default_value=1e-2,
        #                                                log=True)

        config_space.add_hyperparameters([lr, optimizer, sgd_momentum, batch_size, dropout_rate,
                                          # l2_regularizer, l2_decay_rate, use_softmax, lr_scheduler, batch_norm,
                                          # non_lin_fn, weight_initializer
                                          ])

        num_glob_conv_blocks = CSH.UniformIntegerHyperparameter('num_glob_conv_blocks', lower=2, upper=5)
        num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=0, upper=4)
        conv_ls_per_block = CSH.UniformIntegerHyperparameter('conv_ls_per_block', lower=1, upper=3, default_value=1)

        # init_fc_neurons = CSH.UniformIntegerHyperparameter('init_fc_neurons', lower=64, upper=512, default_value=256)
        init_fc_neurons = CSH.CategoricalHyperparameter('init_fc_neurons', [32, 64, 128, 256, 512])
        init_conv_filters = CSH.UniformIntegerHyperparameter('init_conv_filters', lower=2, upper=6, default_value=4)

        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=6)
        kernel_stride = CSH.UniformIntegerHyperparameter('kernel_stride', lower=1, upper=2, default_value=1)

        pool_size_glob = CSH.UniformIntegerHyperparameter('pool_size_glob', lower=2, upper=8)
        pool_stride = CSH.UniformIntegerHyperparameter('pool_stride', lower=1, upper=2, default_value=1)

        pool_size_loc = CSH.UniformIntegerHyperparameter('pool_size_loc', lower=2, upper=8)
        num_loc_conv_blocks = CSH.UniformIntegerHyperparameter('num_loc_conv_blocks', lower=1, upper=3)

        config_space.add_hyperparameters([num_glob_conv_blocks,
                                          num_fc_layers,
                                          conv_ls_per_block, kernel_size, kernel_stride,
                                          pool_size_glob, pool_stride,
                                          pool_size_loc, num_loc_conv_blocks,
                                          init_fc_neurons,
                                          init_conv_filters])

        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        config_space.add_condition(cond)

        return config_space

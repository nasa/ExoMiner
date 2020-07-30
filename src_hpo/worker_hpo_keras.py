"""
Custom TensorFlow Keras worker for the hyperparameter optimizer.

TODO: find a way to automate the matplotlib switch backend when running on Pleiades
"""

# 3rd party
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

# local
from src.utils_dataio import InputFn
from src.utils_metrics import get_metrics

import paths

if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


class TransitClassifier(Worker):

    def __init__(self, config_args, worker_id_custom=1, **kwargs):
        """ Initialize the worker.

        :param config_args: configuration parameters
        :param worker_id_custom: int, worker ID
        :param kwargs:  dict, other parameters for the worker
        """

        # call the init of the inherited worker class
        super().__init__(**kwargs)

        self.worker_id_custom = str(worker_id_custom)  # worker id

        # self.models_directory = config_args.models_directory  # directory to save models
        self.results_directory = config_args.results_directory  # directory to save results
        self.tfrec_dir = config_args.tfrec_dir  # tfrecord directory

        # base config that is fixed and it is not part of the evaluated configuration space
        self.base_config = config_args.config

        self.hpo_loss = config_args.hpo_loss  # HPO loss (it has to be one of the metrics/loss being evaluated

        self.ensemble_n = config_args.ensemble_n  # number of models trained per configuration

        # # get cross-entropy class weights (weighted cross-entropy)
        # if config_args.satellite == 'kepler' and not config_args.use_kepler_ce:
        #     self.ce_weights = None
        # else:
        #     self.ce_weights = config_args.ce_weights

        # # get data to filter the datasets on real-time
        # if config_args.filter_data is None:
        #     self.filter_data = {dataset: None for dataset in ['train', 'val', 'test']}
        # else:
        #     self.filter_data = config_args.filter_data

        self.features_set = config_args.features_set

        self.scalar_params_idxs = config_args.scalar_params_idxs

        self.num_gpus = config_args.num_gpus  # number of GPUs per node; each worker (configuration) is assigned one GPU

        self.BaseModel = config_args.BaseModel

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        """

        :param config_id: tuple, configuration ID
        :param config: sampled config
        :param budget: int, budget (number of epochs to train the configuration sampled)
        :param working_directory: str, a name of a directory that is unique to this configuration. Use this to store
        intermediate results on lower budgets that can be reused later for a larger budget (for iterative algorithms,
        for example).
        :param args: tuple, other args
        :param kwargs: dict, other args
        :return:
            dict:
                needs to return a dictionary with two mandatory entries:
                - 'loss': a numerical value that is MINIMIZED
                - 'info': This can be pretty much any build in python type, e.g. a dict with lists as value.
                Due to Pyro4 handling the remote function calls, 3rd party types like numpy arrays are not supported!
        """

        # # set a specific GPU for training the ensemble
        # gpu_id = int(self.worker_id_custom) % self.num_gpus
        # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)

        # merge sampled and fixed configurations
        config.update(self.base_config)

        # initialize callbacks list
        # used to add callbacks inside the scope of this function
        # TODO: couldn't we do any callback outside?
        #       pro is that there are less parameters to the input of the function
        callbacks_list = config['callbacks_list']

        # input function for training on the training set
        train_input_fn = InputFn(file_pattern=self.tfrec_dir + '/train*',
                                 batch_size=config['batch_size'],
                                 mode=tf.estimator.ModeKeys.TRAIN,
                                 label_map=config['label_map'],
                                 data_augmentation=config['data_augmentation'],
                                 # filter_data=self.filter_data['train'],
                                 features_set=self.features_set,
                                 scalar_params_idxs=self.scalar_params_idxs)

        # input functions for evaluating on the validation and test sets
        val_input_fn = InputFn(file_pattern=self.tfrec_dir + '/val*',
                               batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.EVAL,
                               label_map=config['label_map'],
                               # filter_data=filter_data['val'],
                               features_set=self.features_set,
                               scalar_params_idxs=self.scalar_params_idxs)
        test_input_fn = InputFn(file_pattern=self.tfrec_dir + '/test*',
                                batch_size=config['batch_size'],
                                mode=tf.estimator.ModeKeys.EVAL,
                                label_map=config['label_map'],
                                # filter_data=filter_data['test'],
                                features_set=self.features_set,
                                scalar_params_idxs=self.scalar_params_idxs)

        # initialize results variable
        res = {}

        # setup monitoring metrics
        metrics_list = get_metrics()

        for model_i in range(self.ensemble_n):

            printstr = "Training model %i(%i) on worker %s: %d epochs" % (model_i + 1, self.ensemble_n,
                                                                          self.worker_id_custom,
                                                                          int(budget))
            print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')
            sys.stdout.flush()

            # instantiate Keras model
            model = self.BaseModel(config, self.features_set, self.scalar_params_idxs).kerasModel

            # compile model
            if config['optimizer'] == 'Adam':
                model.compile(optimizer=optimizers.Adam(learning_rate=config['lr'],
                                                        beta_1=0.9,
                                                        beta_2=0.999,
                                                        epsilon=1e-8,
                                                        amsgrad=False,
                                                        name='Adam'),  # optimizer
                              # loss function to minimize
                              loss=losses.BinaryCrossentropy(from_logits=False,
                                                             label_smoothing=0,
                                                             name='binary_crossentropy'),
                              # list of metrics to monitor
                              metrics=metrics_list)

            else:
                model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                                       momentum=config['sgd_momentum'],
                                                       nesterov=False,
                                                       name='SGD'),  # optimizer
                              # loss function to minimize
                              loss=losses.BinaryCrossentropy(from_logits=False,
                                                             label_smoothing=0,
                                                             name='binary_crossentropy'),
                              # list of metrics to monitor
                              metrics=metrics_list)

            # fit the model to the training data
            history = model.fit(x=train_input_fn(),
                                y=None,
                                batch_size=None,
                                epochs=int(budget),
                                verbose=0,
                                callbacks=callbacks_list,
                                validation_split=0.,
                                validation_data=val_input_fn(),
                                shuffle=True,  # does the input function shuffle for every epoch?
                                class_weight=None,
                                sample_weight=None,
                                initial_epoch=0,
                                steps_per_epoch=None,
                                validation_steps=None,
                                max_queue_size=10,  # used for generator or keras.utils.Sequence input only
                                workers=1,  # same
                                use_multiprocessing=False  # same
                               )

            # get records of loss and metrics for the training and validation sets
            res_i = history.history

            # evaluate on the test set at the end of the training only
            res_i_eval = model.evaluate(x=test_input_fn(),
                                        y=None,
                                        batch_size=None,
                                        verbose=0,
                                        sample_weight=None,
                                        steps=None,
                                        callbacks=None,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False)

            # add test set metrics and loss to result
            for metric_name_i, metric_name in enumerate(model.metrics_names):
                res_i['test_{}'.format(metric_name)] = res_i_eval[metric_name_i]

            # add results for this model to the results for the ensemble
            if len(res.keys()) == 0:
                for metric_name in res_i:
                    res[metric_name] = [res_i[metric_name]]
            else:
                for metric_name in res:
                    res[metric_name].append(res_i[metric_name])

        # get ensemble average metrics and loss
        for metric in res:
            # res[metric] = {'all scores': res[metric],
            #                         'median': np.median(res[metric], axis=0),
            #                         'mad': np.median(np.abs(res[metric] -
            #                                                 np.median(res[metric], axis=0)), axis=0)}
            res[metric] = {'all scores': res[metric],
                           'central tendency': np.mean(res[metric], axis=0),
                           'deviation': np.std(res[metric], axis=0, ddof=1) / np.sqrt(self.ensemble_n)}

        # save metrics and loss for the ensemble
        np.save(os.path.join(self.results_directory,
                             'config{}_budget{:.0f}_ensemblemetrics.npy'.format(config_id, budget)), res)

        # draw loss and evaluation metric plots for the model on this given budget
        self.draw_plots(res, config_id)

        # report HPO loss and additional metrics and loss
        hpo_loss_val = res['val_{}'.format(self.hpo_loss)]['central tendency'][-1]
        # add test metrics and loss
        info_dict = {metric: [float(res[metric]['central tendency']), float(res[metric]['deviation'])]
                     for metric in res if 'test' in metric and len(res[metric]['central tendency'].shape) == 0}
        # add train and validation metrics and loss
        info_dict.update({metric: [float(res[metric]['central tendency'][-1]), float(res[metric]['deviation'][-1])]
                         for metric in res if 'test' not in metric and len(res[metric]['central tendency'].shape) == 1})
        res_hpo = {'loss': 1 - float(hpo_loss_val),  # HPO loss to be minimized
                   'info': info_dict}

        print('#' * 100)
        print('Finished evaluating configuration {} on worker {} using a budget of {}'.format(config_id,
                                                                                              self.worker_id_custom,
                                                                                              budget))
        for k in res_hpo:
            if k != 'info':
                print('HPO {}: {}'.format(k, res_hpo[k]))
            else:
                for l in res_hpo[k]:
                    # if 'test' in l:
                    print('{}: {} +- {}'.format(l, *res_hpo[k][l]))
                    # else:
                    #     print('{}: {} +- {}'.format(l, res_hpo[k][l][0][-1], res_hpo[k][l][1][-1]))
        print('#' * 100)
        sys.stdout.flush()

        return res_hpo

    def draw_plots(self, res, config_id):
        """ Draw loss and evaluation metric plots.

        :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
        for the test set.
        :param config_id: tuple, configuration ID
        :return:
        """

        # initialize epochs array
        epochs = np.arange(1, len(res['loss']['central tendency']) + 1)

        alpha = 0.1

        # plot loss and optimization metric as function of the epochs
        f, ax = plt.subplots(1, 2)
        for training_loss_i in res['loss']['all scores']:
            ax[0].plot(epochs, training_loss_i, color='b', alpha=alpha)
        ax[0].plot(epochs, res['loss']['central tendency'], label='Training', color='b')
        for validation_loss_i in res['val_loss']['all scores']:
            ax[0].plot(epochs, validation_loss_i, color='r', alpha=alpha)
        ax[0].plot(epochs, res['val_loss']['central tendency'], label='Validation', color='r')
        for test_loss_i in res['test_loss']['all scores']:
            ax[0].scatter(epochs[-1], test_loss_i, c='k', alpha=alpha)
        ax[0].scatter(epochs[-1], res['test_loss']['central tendency'], c='k', label='Test')
        ax[0].set_xlim([0, epochs[-1]])
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Val/Test %.4f/%.4f' % (res['val_loss']['central tendency'][-1],
                                                res['test_loss']['central tendency']))
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        for training_hpoloss_i in res[self.hpo_loss]['all scores']:
            ax[1].plot(epochs, training_hpoloss_i, color='b', alpha=alpha)
        ax[1].plot(epochs, res[self.hpo_loss]['central tendency'], label='Training', color='b')
        for validation_hpoloss_i in res['val_{}'.format(self.hpo_loss)]['all scores']:
            ax[1].plot(epochs, validation_hpoloss_i, color='r', alpha=alpha)
        ax[1].plot(epochs, res['val_{}'.format(self.hpo_loss)]['central tendency'], label='Validation', color='r')
        for test_hpoloss_i in res['test_{}'.format(self.hpo_loss)]['all scores']:
            ax[1].scatter(epochs[-1], test_hpoloss_i, c='k', alpha=alpha)
        ax[1].scatter(epochs[-1], res['test_{}'.format(self.hpo_loss)]['central tendency'], label='Test', c='k')
        ax[1].set_xlim([0, epochs[-1]])
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid(True)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel(self.hpo_loss)
        ax[1].set_title('%s\nVal/Test %.4f/%.4f' % (self.hpo_loss,
                                                    res['val_{}'.format(self.hpo_loss)]['central tendency'][-1],
                                                    res['test_{}'.format(self.hpo_loss)]['central tendency']))
        ax[1].legend(loc="lower right")
        f.suptitle('Config {} | Budget = {:.0f}'.format(config_id, epochs[-1]))
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(os.path.join(self.results_directory,
                               'config_{}_budget{:.0f}_mloss-hpoloss.png'.format(config_id, epochs[-1])))
        plt.close()

        # plot precision, recall, roc auc, pr auc curves for the validation and test sets
        f, ax = plt.subplots()
        ax.plot(epochs, res['val_precision']['central tendency'], label='Val Precision', color='b')
        ax.plot(epochs, res['val_recall']['central tendency'], label='Val Recall', color='r')
        ax.plot(epochs, res['val_auc_roc']['central tendency'], label='Val ROC AUC', color='g')
        ax.plot(epochs, res['val_auc_pr']['central tendency'], label='Val PR AUC', color='k')
        ax.scatter(epochs[-1], res['test_precision']['central tendency'], label='Test Precision', c='b')
        ax.scatter(epochs[-1], res['test_recall']['central tendency'], label='Test Recall', c='r')
        ax.scatter(epochs[-1], res['test_auc_roc']['central tendency'], label='Test ROC AUC', c='g')
        ax.scatter(epochs[-1], res['test_auc_pr']['central tendency'], label='Test PR AUC', c='k')
        ax.grid(True)
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
        ax.set_xlim([0, epochs[-1]])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metric Value')
        ax.set_title('Precision-Recall-ROC AUC-PR AUC\nVal/Test')
        # ax[1].legend(loc="lower right")
        # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(os.path.join(self.results_directory,
                               'config{}_budget{}_prec-rec-auc.png'.format(config_id, epochs[-1])))
        plt.close()

        # plot pr curve
        f, ax = plt.subplots()
        ax.plot(res['val_rec_thr']['central tendency'][-1], res['val_prec_thr']['central tendency'][-1],
                label='Val (AUC={:.3f})'.format(res['val_auc_pr']['central tendency'][-1]), color='r')
        ax.plot(res['test_rec_thr']['central tendency'], res['test_prec_thr']['central tendency'],
                label='Test (AUC={:.3f})'.format(res['test_auc_pr']['central tendency']), color='k')
        ax.plot(res['rec_thr']['central tendency'][-1], res['prec_thr']['central tendency'][-1],
                label='Train (AUC={:.3f})'.format(res['auc_pr']['central tendency'][-1]), color='b')
        ax.scatter(res['val_rec_thr']['central tendency'][-1],
                   res['val_prec_thr']['central tendency'][-1], c='r')
        ax.scatter(res['test_rec_thr']['central tendency'],
                   res['test_prec_thr']['central tendency'], c='k')
        ax.scatter(res['rec_thr']['central tendency'][-1],
                   res['prec_thr']['central tendency'][-1], c='b')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall curve')
        f.savefig(os.path.join(self.results_directory, 'config{}_budget{}_prcurve.png'.format(config_id, epochs[-1])))
        plt.close()

    @staticmethod
    def get_configspace():
        """ Build the hyperparameter configuration space

        :return: ConfigurationsSpace-Object
        """

        config_space = CS.ConfigurationSpace(seed=None)

        # use_softmax = CSH.CategoricalHyperparameter('use_softmax', [True, False])

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, log=True)
        # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp_fast', 'inv_exp_slow'])

        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.001, upper=0.99, log=True)

        # batch_norm = CSH.CategoricalHyperparameter('batch_norm', [True, False])
        # non_lin_fn = CSH.CategoricalHyperparameter('non_lin_fn', ['relu', 'prelu'])
        # weight_initializer = CSH.CategoricalHyperparameter('weight_initializer', ['he', 'glorot'])

        # batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64, 128, 256], default_value=32)

        # previous values: 0.001 to 0.7
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.2, log=True)

        # l2_regularizer = CSH.CategoricalHyperparameter('l2_regularizer', [True, False])
        # l2_decay_rate = CSH.UniformFloatHyperparameter('decay_rate', lower=1e-4, upper=1e-1, default_value=1e-2,
        #                                                log=True)

        config_space.add_hyperparameters([sgd_momentum, dropout_rate, optimizer, lr
                                          # lr, optimizer, batch_size, l2_regularizer, l2_decay_rate, use_softmax, lr_scheduler,
                                          # batch_norm, non_lin_fn, weight_initializer
                                          ])

        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        config_space.add_condition(cond)

        # cond = CS.EqualsCondition(l2_decay_rate, l2_regularizer, True)
        # config_space.add_condition(cond)

        init_conv_filters = CSH.UniformIntegerHyperparameter('init_conv_filters', lower=2, upper=6)
        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=8)
        kernel_stride = CSH.UniformIntegerHyperparameter('kernel_stride', lower=1, upper=2)
        conv_ls_per_block = CSH.UniformIntegerHyperparameter('conv_ls_per_block', lower=1, upper=3)

        pool_size_loc = CSH.UniformIntegerHyperparameter('pool_size_loc', lower=2, upper=8)
        pool_size_glob = CSH.UniformIntegerHyperparameter('pool_size_glob', lower=2, upper=8)
        pool_stride = CSH.UniformIntegerHyperparameter('pool_stride', lower=1, upper=2)

        num_loc_conv_blocks = CSH.UniformIntegerHyperparameter('num_loc_conv_blocks', lower=1, upper=5)
        num_glob_conv_blocks = CSH.UniformIntegerHyperparameter('num_glob_conv_blocks', lower=1, upper=5)

        init_fc_neurons = CSH.CategoricalHyperparameter('init_fc_neurons', [32, 64, 128, 256, 512])
        # init_fc_neurons = CSH.UniformIntegerHyperparameter('init_fc_neurons', lower=64, upper=512, default_value=256)

        num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=1, upper=4)

        config_space.add_hyperparameters([num_glob_conv_blocks,
                                          num_fc_layers,
                                          conv_ls_per_block,
                                          kernel_size,
                                          kernel_stride,
                                          pool_size_glob,
                                          pool_stride,
                                          pool_size_loc,
                                          num_loc_conv_blocks,
                                          init_fc_neurons,
                                          init_conv_filters])

        return config_space

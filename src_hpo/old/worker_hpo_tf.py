"""
Custom TensorFlow Estimator API worker for the hyperparameter optimizer.

TODO: add sess_config as parameter for the worker
      add additional default parameters to the configs file
      find a way to automate the matplotlib switch backend when running on Pleiades
      ******** remember to enable/disable GPU assignment for multi/single GPU systems **********
      add config attribute related with tf.ConfigProto options
"""

# 3rd party
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import os
import sys
import tempfile
import shutil
import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

# local
from src.old.estimator_util import ModelFn, InputFn
from src.old.config import add_default_missing_params
import paths

if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


class TransitClassifier(Worker):

    def __init__(self, config_args, worker_id_custom=1, **kwargs):
        super().__init__(**kwargs)

        self.worker_id_custom = str(worker_id_custom)  # worker id

        self.models_directory = config_args.models_directory  # directory to save models
        self.results_directory = config_args.results_directory  # directory to save results
        self.tfrec_dir = config_args.tfrec_dir  # tfrecord directory

        self.satellite = config_args.satellite  # kepler or tess
        self.label_map = config_args.label_map  # maps between class and integer index
        # self.centr_flag = config_args.centr_flag  # bool, True if centroid data is being used

        self.multi_class = config_args.multi_class  # bool, True for multiclassification
        # self.n_train = config_args.n_train

        self.hpo_loss = config_args.hpo_loss  # HPO loss (it has to be one of the metrics/loss being evaluated

        self.ensemble_n = config_args.ensemble_n  # number of models trained per configuration

        # get cross-entropy class weights (weighted cross-entropy)
        if config_args.satellite == 'kepler' and not config_args.use_kepler_ce:
            self.ce_weights = None
        else:
            self.ce_weights = config_args.ce_weights

        # get data to filter the datasets on real-time
        if config_args.filter_data is None:
            self.filter_data = {dataset: None for dataset in ['train', 'val', 'test']}
        else:
            self.filter_data = config_args.filter_data

        self.features_set = config_args.features_set

        self.scalar_params_idxs = config_args.scalar_params_idxs

        self.num_gpus = config_args.num_gpus  # number of GPUs per node; each worker (configuration) is assigned one GPU

        self.BaseModel = config_args.BaseModel

    @staticmethod
    def get_model_dir(path):
        """Returns a randomly named, non-existing model file folder

        :param path: str, base directory in which the models' folders are created
        :return:
            model_dir_custom: str, path to model directory
        """

        def _gen_dir():
            return os.path.join(path, tempfile.mkdtemp().split('/')[-1])

        model_dir_custom = _gen_dir()

        while os.path.isdir(model_dir_custom):
            model_dir_custom = _gen_dir()

        return model_dir_custom

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):

        # if 'nobackup' in dirname(__file__):  # if running on Pleiades
        # config_sess.gpu_options.force_gpu_compatible = True  # Force pinned memory
        # config_sess.intra_op_parallelism_threads = 1
        # config_sess.gpu_options.visible_device_list = "0"
        # gpu_options = tf.GPUOptions(visible_device_list=str(int(self.worker_id_custom) % 4))
        sess_config = tf.ConfigProto(log_device_placement=False, gpu_options=None)
        if self.num_gpus > 0:
            sess_config.gpu_options.visible_device_list = str(int(self.worker_id_custom) % self.num_gpus)

        config['satellite'] = self.satellite
        config['ce_weights'] = self.ce_weights
        # config['n_train'] = self.n_train
        config['multi_class'] = self.multi_class
        config['label_map'] = self.label_map
        config['worker_id_custom'] = self.worker_id_custom

        config = add_default_missing_params(config=config)

        input_fn_train = InputFn(file_pattern=self.tfrec_dir + '/train*', batch_size=config['batch_size'],
                                 mode=tf.estimator.ModeKeys.TRAIN, label_map=self.label_map,
                                 filter_data=self.filter_data['train'], features_set=self.features_set,
                                 scalar_params_idxs=self.scalar_params_idxs)

        input_fn_traineval = InputFn(file_pattern=self.tfrec_dir + '/train*', batch_size=config['batch_size'],
                                     mode=tf.estimator.ModeKeys.EVAL, label_map=self.label_map,
                                     filter_data=self.filter_data['train'], features_set=self.features_set,
                                     scalar_params_idxs=self.scalar_params_idxs)

        input_fn_val = InputFn(file_pattern=self.tfrec_dir + '/val*', batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.EVAL, label_map=self.label_map,
                               filter_data=self.filter_data['val'], features_set=self.features_set,
                               scalar_params_idxs=self.scalar_params_idxs)

        input_fn_test = InputFn(file_pattern=self.tfrec_dir + '/test*', batch_size=config['batch_size'],
                                mode=tf.estimator.ModeKeys.EVAL, label_map=self.label_map,
                                filter_data=self.filter_data['test'], features_set=self.features_set,
                                scalar_params_idxs=self.scalar_params_idxs)

        metrics_list = ['loss', 'accuracy', 'pr auc', 'precision', 'recall', 'roc auc', 'prec thr', 'rec thr']
        dataset_ids = ['training', 'validation', 'test']
        res = {dataset: {metric: [[] for _ in range(self.ensemble_n)] for metric in metrics_list} for dataset in
               dataset_ids}

        for model_i in range(self.ensemble_n):
            model_dir_custom = self.get_model_dir(self.models_directory)

            classifier = tf.estimator.Estimator(ModelFn(self.BaseModel, config),
                                                config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                              session_config=sess_config,
                                                                              tf_random_seed=None),
                                                model_dir=model_dir_custom)

            for epoch_i in range(int(budget)):  # train model
                printstr = "Training model %i(%i) on worker %s: Epoch %d of %d" % (model_i + 1, self.ensemble_n,
                                                                                   self.worker_id_custom, epoch_i + 1,
                                                                                   int(budget))
                print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')
                sys.stdout.flush()

                _ = classifier.train(input_fn_train)

                # evaluate model on the training, validation and test sets
                res_i = {'training': classifier.evaluate(input_fn_traineval, name='training set'),
                         'validation': classifier.evaluate(input_fn_val, name='validation set'),
                         'test': classifier.evaluate(input_fn_test, name='test set')}

                for dataset in dataset_ids:
                    for metric in metrics_list:
                        res[dataset][metric][model_i].append(res_i[dataset][metric])

            # delete saved model (model will not be read from in future)
            shutil.rmtree(model_dir_custom, ignore_errors=True)

        # get average per epoch between ensembles
        for dataset in dataset_ids:
            for metric in metrics_list:
                # res[dataset][metric] = {'all scores': res[dataset][metric],
                #                         'median': np.median(res[dataset][metric], axis=0),
                #                         'mad': np.median(np.abs(res[dataset][metric] -
                #                                                 np.median(res[dataset][metric], axis=0)), axis=0)}
                res[dataset][metric] = {'all scores': res[dataset][metric],
                                        'central tendency': np.mean(res[dataset][metric], axis=0),
                                        'deviation': np.std(res[dataset][metric], axis=0, ddof=1) /
                                                     np.sqrt(self.ensemble_n)}

        # save metrics
        np.save(os.path.join(self.results_directory, 'ensemblemetrics_{}budget{:.0f}'.format(config_id, budget)), res)

        # draw loss and evaluation metric plots for the model on this given budget
        self.draw_plots(res, config_id)

        # report HPO loss and additional performance metrics and info
        hpo_loss = res['validation'][self.hpo_loss]['central tendency']
        ep = -1  # np.argmax(hpo_loss)  # epoch where the HPO loss was the best
        res_hpo = {'loss': 1 - float(hpo_loss[-1]),
                   'info': {dataset + ' ' + metric: [float(res[dataset][metric]['central tendency'][ep]),
                                                     float(res[dataset][metric]['deviation'][ep])]
                            for dataset in ['validation', 'test'] for metric in metrics_list if metric not in
                            ['prec thr', 'rec thr']}}

        print('#' * 100)
        print('Finished evaluating configuration {} on worker {} using a budget of {}'.format(config_id,
                                                                                              self.worker_id_custom,
                                                                                              budget))
        for k in res_hpo:
            if k != 'info':
                print(k + ': ', res_hpo[k])
            else:
                for l in res_hpo[k]:
                    print(l + ': ', res_hpo[k][l])
        print('#' * 100)
        sys.stdout.flush()

        return res_hpo

    @staticmethod
    def del_savedmodel(savedir):
        """ Delete data pertaining to the model trained and evaluated.

        :param savedir: str, filepath of the folder where the data was stored.
        :return:
        """

        # removes model's folder contents
        for content in os.listdir(savedir):
            fp = os.path.join(savedir, content)
            if os.path.isfile(fp):  # delete file
                os.unlink(fp)
            elif os.path.isdir(fp):  # delete folder
                shutil.rmtree(fp, ignore_errors=True)

    def draw_plots(self, res, config_id):
        """ Draw loss and evaluation metric plots.

        :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
        for the test set.
        :param config_id: tuple, config id
        :return:
        """
        epochs = np.arange(0, len(res['training']['loss']['central tendency']), 1, dtype='int')

        min_val_loss = np.min(res['validation']['loss']['central tendency'])
        loss_ep_idx = np.argmin(res['validation']['loss']['central tendency'])
        max_val_auc = np.max(res['validation']['pr auc']['central tendency'])
        auc_ep_idx = np.argmax(res['validation']['pr auc']['central tendency'])

        alpha = 0.1

        # Loss and evaluation metric plots
        f, ax = plt.subplots(1, 2)
        for training_loss_i in res['training']['loss']['all scores']:
            ax[0].plot(epochs, training_loss_i, color='b', alpha=alpha)
        ax[0].plot(epochs, res['training']['loss']['central tendency'], label='Training', color='b')
        for validation_loss_i in res['validation']['loss']['all scores']:
            ax[0].plot(epochs, validation_loss_i, color='r', alpha=alpha)
        ax[0].plot(epochs, res['validation']['loss']['central tendency'], label='Validation', color='r')
        ax[0].scatter(epochs[loss_ep_idx], min_val_loss, c='r')
        ax[0].scatter(epochs[loss_ep_idx], res['test']['loss']['central tendency'][loss_ep_idx], c='k', label='Test')
        ax[0].set_xlim([0.0, epochs[-1] + 1])
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        ax[0].set_title('Categorical cross-entropy\nVal/Test '
                        '%.4f/%.4f' % (min_val_loss, res['test']['loss']['central tendency'][loss_ep_idx]))

        for training_auc_i in res['training']['pr auc']['all scores']:
            ax[1].plot(epochs, training_auc_i, color='b', alpha=alpha)
        ax[1].plot(epochs, res['training']['pr auc']['central tendency'], label='Training', color='b')
        for validation_auc_i in res['validation']['pr auc']['all scores']:
            ax[1].plot(epochs, validation_auc_i, color='r', alpha=alpha)
        ax[1].plot(epochs, res['validation']['pr auc']['central tendency'], label='Validation', color='r')
        ax[1].scatter(epochs[auc_ep_idx], max_val_auc, c='r')
        ax[1].scatter(epochs[auc_ep_idx], res['test']['pr auc']['central tendency'][auc_ep_idx], label='Test', c='k')
        ax[1].set_xlim([0.0, epochs[-1] + 1])
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('AUC')
        ax[1].legend(loc="lower right")
        ax[1].grid(True)
        ax[1].set_title('Evaluation Metric\nVal/Test %.4f/%.4f' % (max_val_auc,
                                                                   res['test']['pr auc']['central tendency'][auc_ep_idx]))

        f.suptitle('Config {} | Budget = {:.0f} (Best val:{:.0f})'.format(config_id, epochs[-1], epochs[auc_ep_idx]))
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(self.results_directory + '/plotseval_{}budget{:.0f}.png'.format(config_id, epochs[-1] + 1))
        plt.close()

        # Precision and Recall plots
        f, ax = plt.subplots()
        for train_prec_i in res['train']['precision']['all scores']:
            ax.plot(epochs, train_prec_i, color='r', alpha=alpha)
        ax.plot(epochs, res['train']['precision']['central tendency'], color='r', label='Train Prec')

        for val_rec_i in res['validation']['recall']['all scores']:
            ax.plot(epochs, val_rec_i, color='r', alpha=alpha, linestyle='--')
        ax.plot(epochs, res['validation']['recall']['central tendency'], color='r', label='Val Rec', linestyle='--')

        for test_prec_i in res['test']['precision']['all scores']:
            ax.plot(epochs, test_prec_i, color='k', alpha=alpha)
        ax.plot(epochs, res['test']['precision']['central tendency'], color='k', label='Test Prec')

        for test_rec_i in res['test']['recall']['all scores']:
            ax.plot(epochs, test_rec_i, color='k', alpha=alpha, linestyle='--')
        ax.plot(epochs, res['test']['recall']['central tendency'], color='k', label='Test Rec', linestyle='--')

        ax.set_ylabel('Metric value')
        ax.set_xlabel('Epochs')
        ax.set_xlim([0.0, epochs[-1] + 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.set_title('Precision and Recall')
        f.suptitle('Config {} | Budget = {:.0f} (Best val:{:.0f})'.format(config_id, epochs[-1], epochs[auc_ep_idx]))
        f.savefig(self.results_directory + '/prec_rec_{}budget{:.0f}.png'.format(config_id, epochs[-1] + 1))
        plt.close()

        # plot pr curve
        f, ax = plt.subplots()
        for i in range(len(res['validation']['prec thr']['all scores'])):
            ax.plot(res['validation']['rec thr']['all scores'][i][-1],
                    res['validation']['prec thr']['all scores'][i][-1], color='r', alpha=alpha, linestyle='--')
        ax.plot(res['validation']['rec thr']['central tendency'][-1],
                res['validation']['prec thr']['central tendency'][-1], color='k', linestyle='--',
                label='Validation (AUC={:.3f})'.format(res['validation']['pr auc']['central tendency'][-1]))

        for i in range(len(res['test']['prec thr']['all scores'])):
            ax.plot(res['test']['rec thr']['all scores'][i][-1],
                    res['test']['prec thr']['all scores'][i][-1], color='r', alpha=alpha)
        ax.plot(res['test']['rec thr']['central tendency'][-1],
                res['test']['prec thr']['central tendency'][-1], color='k',
                label='Test (AUC={:.3f})'.format(res['test']['pr auc']['central tendency'][-1]))

        # CHANGE THR_VEC ACCORDINGLY TO THE SAMPLED THRESHOLD VALUES
        thr_vec = np.linspace(0, 999, 11, endpoint=True, dtype='int')
        ax.scatter(np.array(res['validation']['rec thr']['central tendency'][-1])[thr_vec],
                   np.array(res['validation']['prec thr']['central tendency'][-1])[thr_vec], c='r')
        ax.scatter(np.array(res['test']['rec thr']['central tendency'][-1])[thr_vec],
                   np.array(res['test']['prec thr']['central tendency'][-1])[thr_vec], c='r')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall curve Val/Test')
        f.suptitle('Config {} | Budget = {:.0f}'.format(config_id, epochs[-1]))
        f.savefig(self.results_directory + '/pr_curve_{}budget{:.0f}.png'.format(config_id, epochs[-1] + 1))
        plt.close()

    @staticmethod
    def get_configspace():
        """
        Build the hyperparameter configuration space
        :return: ConfigurationsSpace-Object
        """
        config_space = CS.ConfigurationSpace()

        # use_softmax = CSH.CategoricalHyperparameter('use_softmax', [True, False])

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp_fast', 'inv_exp_slow'])

        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.001, upper=0.99, default_value=0.9,
                                                      log=True)

        # batch_norm = CSH.CategoricalHyperparameter('batch_norm', [True, False])
        # non_lin_fn = CSH.CategoricalHyperparameter('non_lin_fn', ['relu', 'prelu'])
        # weight_initializer = CSH.CategoricalHyperparameter('weight_initializer', ['he', 'glorot'])

        # batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64, 128, 256], default_value=32)

        # previous values: 0.001 to 0.7
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.2, default_value=0.2,
                                                      log=True)

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

        init_conv_filters = CSH.UniformIntegerHyperparameter('init_conv_filters', lower=2, upper=7, default_value=4)
        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=8, default_value=2)
        kernel_stride = CSH.UniformIntegerHyperparameter('kernel_stride', lower=1, upper=2, default_value=1)
        conv_ls_per_block = CSH.UniformIntegerHyperparameter('conv_ls_per_block', lower=1, upper=3, default_value=1)

        pool_size_loc = CSH.UniformIntegerHyperparameter('pool_size_loc', lower=2, upper=8, default_value=2)
        pool_size_glob = CSH.UniformIntegerHyperparameter('pool_size_glob', lower=2, upper=8, default_value=2)
        pool_stride = CSH.UniformIntegerHyperparameter('pool_stride', lower=1, upper=2, default_value=1)

        num_loc_conv_blocks = CSH.UniformIntegerHyperparameter('num_loc_conv_blocks', lower=1, upper=3, default_value=2)
        num_glob_conv_blocks = CSH.UniformIntegerHyperparameter('num_glob_conv_blocks', lower=2, upper=5,
                                                                default_value=3)

        init_fc_neurons = CSH.CategoricalHyperparameter('init_fc_neurons', [32, 64, 128, 256, 512])
        # init_fc_neurons = CSH.UniformIntegerHyperparameter('init_fc_neurons', lower=64, upper=512, default_value=256)

        num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=0, upper=4, default_value=2)

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

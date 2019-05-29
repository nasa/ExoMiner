"""
Train models using a given configuration obtained on a hyperparameter optimization study.
"""

import os
# import logging
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# logging.getLogger("tensorflow").setLevel(logging.INFO)
# tf.logging.set_verbosity(tf.logging.INFO)
import hpbandster.core.result as hpres

import numpy as np
# import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# if 'nobackup' in os.path.dirname(__file__):
#     from src.estimator_util import InputFn, ModelFn, CNN1dModel
#     from src.eval_results import eval_model
#     from src.config import Config
# else:
from src.estimator_util import InputFn, ModelFn, CNN1dModel
# from src.eval_results import eval_model
from src.config import Config


def draw_plots(res, save_path):
        """ Draw loss and evaluation metric plots.

        :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
        for the test set.
        :param save_path: str, filepath used to save the plots figure.
        :return:
        """

        res['epochs'] = np.array(res['epochs'], dtype='int')

        f, ax = plt.subplots(1, 2)
        ax[0].plot(res['epochs'], res['training loss'], label='Training')
        ax[0].plot(res['epochs'], res['validation loss'], label='Validation', color='r')
        min_val_loss, ep_idx = np.min(res['validation loss']), np.argmin(res['validation loss'])
        ax[0].scatter(res['epochs'][ep_idx], min_val_loss, c='r')
        ax[0].scatter(res['epochs'][ep_idx], res['test loss'][ep_idx], c='k', label='Test')
        ax[0].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[0].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Categorical cross-entropy\nVal/Test %.4f/%.4f' % (min_val_loss, res['test loss'][ep_idx]))
        ax[0].legend(loc="upper right")
        ax[0].grid()
        ax[1].plot(res['epochs'], res['training auc'], label='Training')
        ax[1].plot(res['epochs'], res['validation auc'], label='Validation', color='r')
        max_val_auc, ep_idx = np.max(res['validation auc']), np.argmax(res['validation auc'])
        ax[1].scatter(res['epochs'][ep_idx], max_val_auc, c='r')
        # ax[1].scatter(res['epochs'][-1], res['test auc'], label='Test', c='k')
        ax[1].scatter(res['epochs'][ep_idx], res['test auc'][ep_idx], label='Test', c='k')
        ax[1].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[1].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid()
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('AUC')
        ax[1].set_title('Evaluation Metric\nVal/Test %.4f/%.4f' % (max_val_auc, res['test auc'][ep_idx]))
        ax[1].legend(loc="lower right")
        f.suptitle('Epochs = {:.0f}(Best val:{:.0f})'.format(res['epochs'][-1], res['epochs'][ep_idx]))
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(save_path + '_plotseval_epochs{:.0f}.png'.format(res['epochs'][-1]))

        f, ax = plt.subplots()
        ax.plot(res['epochs'], res['validation precision'], label='Val Precision')
        ax.plot(res['epochs'], res['validation recall'], label='Val Recall')
        ax.plot(res['epochs'], res['validation auc'], label='Val ROC AUC')
        ax.plot(res['epochs'], res['test precision'], label='Test Precision')
        ax.plot(res['epochs'], res['test recall'], label='Test Recall')
        ax.plot(res['epochs'], res['test auc'], label='Test ROC AUC')
        ax.grid()
        # ax.set_xticks(np.arange(0, res['epochs'][-1] + 1, 5))

        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metric Value')
        ax.set_title('Evaluation Metrics\nVal/Test')
        # ax[1].legend(loc="lower right")
        # f.suptitle('Epochs = {:.0f}'.format(res['epochs'][-1]))
        # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(save_path + '_prec_rec_auc.png')

        # min_val_loss, ep = np.min(res['validation loss']), res['epochs'][np.argmin(res['validation loss'])]
        # ax[0].scatter(ep, min_val_loss, c='r')
        # ax[0].scatter(res['epochs'][-1], res['test loss'], c='k', label='Test')
        # ax[0].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[0].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        # # ax[0].set_ylim(bottom=0)
        # ax[0].set_xlabel('Epochs')
        # ax[0].set_ylabel('Loss')
        # ax[0].set_title('Categorical cross-entropy\nVal/Test %.4f/%.4f' % (min_val_loss, res['test loss']))
        # ax[0].legend(loc="upper right")
        # ax[1].plot(res['epochs'], res['training auc'], label='Training')
        # ax[1].plot(res['epochs'], res['validation auc'], label='Validation', color='r')
        # max_val_auc, ep = np.max(res['validation auc']), res['epochs'][np.argmax(res['validation auc'])]
        # ax[1].scatter(ep, max_val_auc, c='r')
        # ax[1].scatter(res['epochs'][-1], res['test auc'], label='Test', c='k')
        # ax[1].set_xlim([0.0, res['epochs'][-1] + 1])
        # if len(res['epochs']) < 16:
        #     ax[1].set_xticks(np.arange(0, res['epochs'][-1] + 1, 1))
        # # ax[1].set_ylim([0.0, 1.05])
        # # ax[1].set_xlabel('Epochs')
        # # ax[1].set_ylabel('AUC')
        # ax[1].set_title('Evaluation Metric\nVal/Test %.4f/%.4f' % (max_val_auc, res['test auc']))
        # # ax[1].legend(loc="lower right")
        # f.suptitle('Epochs = {:.0f}'.format(res['epochs'][-1]))
        # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        # f.savefig(save_path + 'plotseval_epochs{:.0f}.png'.format(res['epochs'][-1]))


def run_main(config, save_path):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: configuration object from the Config class
    :param save_path: str, directory used to save the results
    :return:
    """

    config_sess = tf.ConfigProto(log_device_placement=False)

    classifier = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                        config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                      session_config=config_sess),
                                        model_dir=config.model_dir_custom
                                        )

    train_input_fn = InputFn(file_pattern=config.tfrec_dir + '/train*', batch_size=config.batch_size,
                             mode=tf.estimator.ModeKeys.TRAIN, label_map=config.label_map, centr_flag=config.centr_flag)
    val_input_fn = InputFn(file_pattern=config.tfrec_dir + '/val*', batch_size=config.batch_size,
                           mode=tf.estimator.ModeKeys.EVAL, label_map=config.label_map, centr_flag=config.centr_flag)
    test_input_fn = InputFn(file_pattern=config.tfrec_dir + '/test*', batch_size=config.batch_size,
                            mode=tf.estimator.ModeKeys.EVAL, label_map=config.label_map,
                            centr_flag=config.centr_flag)

    # result = []
    res = {k: [] for k in ['training loss', 'training auc', 'validation loss', 'validation auc', 'epochs',
                           'validation precision', 'validation recall', 'test precision', 'test recall', 'test loss',
                           'test auc']}
    for epoch_i in range(1, config.n_epochs + 1):  # Train and evaluate the model for n_epochs
        print('\n\x1b[0;33;33m' + "Starting epoch %d of %d" % (epoch_i, config.n_epochs) + '\x1b[0m\n')
        # train model
        _ = classifier.train(train_input_fn)

        # evaluate model on given datasets
        print('\n\x1b[0;33;33m' + "Evaluating" + '\x1b[0m\n')

        print('on the training set...')
        # training set
        res_train = classifier.evaluate(train_input_fn)
        res['training loss'].append(res_train['loss'])
        res['training auc'].append(res_train['roc auc'])
        res['epochs'].append(epoch_i)
        print('on the validation set...')
        # validation set
        res_val = classifier.evaluate(val_input_fn)
        res['validation loss'].append(res_val['loss'])
        res['validation auc'].append(res_val['roc auc'])
        res['validation precision'].append(res_val['precision'])
        res['validation recall'].append(res_val['recall'])

        confm_info = {key: value for key, value in res_val.items() if key.startswith('label_')}

        res_i = {'loss': float(res_val['loss']),
                 'val acc': float(res_val['accuracy']),
                 # 'train prec': res_train['precision'],
                 'val prec': float(res_val['precision']),
                 'confmx': confm_info,
                 'epoch': epoch_i}
        if not config.multi_class:
            res_i['roc auc'] = res_val['roc auc']

        # result.append(res_i)
        tf.logging.info('After epoch: {:d}: val acc: {:.6f}, val prec: {:.6f}'.format(epoch_i, res_i['val acc'],
                                                                                      res_i['val prec']))
    # eval_model(config, classifier, result)

        print('on the test set...')
        # evaluate on the test set
        res_test = classifier.evaluate(test_input_fn)
        res['test loss'].append(res_test['loss'])
        res['test auc'].append(res_test['roc auc'])
        res['test precision'].append(res_test['precision'])
        res['test recall'].append(res_test['recall'])

    print('Saving metrics...')
    np.save(save_path + 'res_eval.npy', res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    draw_plots(res, save_path=save_path)


if __name__ == '__main__':

    # results directory
    save_path = '/home/msaragoc/Kepler_planet_finder/results/run_study_4/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + 'models/')

    n_models = 10  # number of models in the ensemble
    n_epochs = 2

    # get best configuration from the HPO study
    res = hpres.logged_results_to_HBS_result('/home/msaragoc/Kepler_planet_finder/configs/study_4')
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    # best_config = id2config[incumbent]['config']
    best_config = id2config[(8, 0, 3)]['config']

    # Shallue's best configuration
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': 1e-4,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    # choose configuration
    config = best_config

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    for item in range(n_models):
        print('Training model %i out of %i on %i' % (item + 1, n_models, n_epochs))
        run_main(Config(n_epochs=n_epochs, model_dir_path=save_path + 'models/', **config),
                 save_path + 'model%i' % (item + 1))


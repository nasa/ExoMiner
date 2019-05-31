"""
Train models using a given configuration obtained on a hyperparameter optimization study.
"""

import os
# import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
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
from src.config import Config


def draw_plots(res, save_path, opt_metric, min_optmetric=False):
    """ Draw loss and evaluation metric plots.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set.
    :param save_path: str, filepath used to save the plots figure.
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective
    epoch. If False, gets the maximum value.
    :return:
    """

    epochs = np.arange(1, len(res['training']['loss']) + 1)
    # min_val_loss, ep_idx = np.min(res['validation loss']), np.argmin(res['validation loss'])
    if min_optmetric:
        ep_idx = np.argmin(res['validation'][opt_metric])
    else:
        ep_idx = np.argmax(res['validation'][opt_metric])

    # loss and optimization metric
    f, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, res['training']['loss'], label='Training')
    ax[0].plot(epochs, res['validation']['loss'], label='Validation', color='r')

    ax[0].scatter(epochs[ep_idx], res['validation']['loss'][ep_idx], c='r')
    ax[0].scatter(epochs[ep_idx], res['test']['loss'][ep_idx], c='k', label='Test')
    ax[0].set_xlim([0, epochs[-1] + 1])
    # ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Categorical cross-entropy\nVal/Test %.4f/%.4f' % (res['validation']['loss'][ep_idx],
                                                                       res['test']['loss'][ep_idx]))
    ax[0].legend(loc="upper right")
    ax[0].grid('on')
    ax[1].plot(epochs, res['training'][opt_metric], label='Training')
    ax[1].plot(epochs, res['validation'][opt_metric], label='Validation', color='r')
    ax[1].scatter(epochs[ep_idx], res['validation'][opt_metric][ep_idx], c='r')
    ax[1].scatter(epochs[ep_idx], res['test'][opt_metric][ep_idx], label='Test', c='k')
    ax[1].set_xlim([0, epochs[-1] + 1])
    # ax[1].set_ylim([0.0, 1.05])
    ax[1].grid('on')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel(opt_metric)
    ax[1].set_title('%s\nVal/Test %.4f/%.4f' % (opt_metric, res['validation'][opt_metric][ep_idx],
                                                               res['test'][opt_metric][ep_idx]))
    ax[1].legend(loc="lower right")
    f.suptitle('Epochs = {:.0f}(Best val:{:.0f})'.format(epochs[-1], epochs[ep_idx]))
    f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.savefig(save_path + '_plotseval_epochs{:.0f}.png'.format(epochs[-1]))
    plt.close()

    # plot precision, recall, roc auc, pr auc curves
    f, ax = plt.subplots()
    ax.plot(epochs, res['validation']['precision'], label='Val Precision')
    ax.plot(epochs, res['validation']['recall'], label='Val Recall')
    ax.plot(epochs, res['validation']['roc auc'], label='Val ROC AUC')
    ax.plot(epochs, res['validation']['pr auc'], label='Val PR AUC')
    ax.plot(epochs, res['test']['precision'], label='Test Precision')
    ax.plot(epochs, res['test']['recall'], label='Test Recall')
    ax.plot(epochs, res['test']['roc auc'], label='Test ROC AUC')
    ax.plot(epochs, res['test']['pr auc'], label='Test PR AUC')
    ax.grid('on')
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
    plt.close()


def run_main(config, save_path, opt_metric, min_optmetric):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: configuration object from the Config class
    :param save_path: str, directory used to save the results
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective epoch.
    If False, gets the maximum value.
    :return:
    """

    config_sess = None  # tf.ConfigProto(log_device_placement=False)

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

    metrics_list = ['loss', 'accuracy', 'pr auc', 'precision', 'recall', 'roc auc']
    dataset_ids = ['training', 'validation', 'test']
    res = {dataset: {metric: [] for metric in metrics_list} for dataset in
           dataset_ids}
    for epoch_i in range(1, config.n_epochs + 1):  # Train and evaluate the model for n_epochs

        print('\n\x1b[0;33;33m' + "Starting epoch %d of %d for %s" %
              (epoch_i, config.n_epochs, save_path.split('/')[-1]) + '\x1b[0m\n')
        # train model
        _ = classifier.train(train_input_fn)

        # evaluate model on given datasets
        print('\n\x1b[0;33;33m' + "Evaluating" + '\x1b[0m\n')
        res_i = {'training': classifier.evaluate(train_input_fn),  # evaluate model on the training set
                 'validation': classifier.evaluate(val_input_fn),  # evaluate model on the validation set
                 'test': classifier.evaluate(test_input_fn)}  # evaluate model on the test set

        for dataset in dataset_ids:
            for metric in metrics_list:
                res[dataset][metric].append(res_i[dataset][metric])

        # confm_info = {key: value for key, value in res_val.items() if key.startswith('label_')}

        # tf.logging.info('After epoch: {:d}: val acc: {:.6f}, val prec: {:.6f}'.format(epoch_i, res_i['val acc'],
        #                                                                               res_i['val prec']))

    print('Saving metrics...')
    np.save(save_path + 'res_eval.npy', res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    draw_plots(res, save_path, opt_metric, min_optmetric)

    print('#' * 100)
    for dataset in dataset_ids:
        print(dataset)
        for metric in metrics_list:
            print('{}: {}'.format(metric, res[dataset][metric]))
    print('#' * 100)


if __name__ == '__main__':

    # results directory
    save_path = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/study_9/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + 'models/')

    n_models = 10  # number of models in the ensemble
    n_epochs = 50

    opt_metric = 'roc auc'
    min_optmetric = False

    # get best configuration from the HPO study
    res = hpres.logged_results_to_HBS_result('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                                             'hpo_configs/study_9')
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best_config = id2config[incumbent]['config']
    # best_config = id2config[(8, 0, 3)]['config']

    # Shallue's best configuration
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': 1e-4,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    # choose configuration
    config = best_config

    for item in range(n_models):
        print('Training model %i out of %i on %i' % (item + 1, n_models, n_epochs))
        run_main(Config(n_epochs=n_epochs, model_dir_path=save_path + 'models/', **config),
                 save_path + 'model%i' % (item + 1), opt_metric, min_optmetric)

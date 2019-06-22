"""
Train models using a given configuration obtained on a hyperparameter optimization study.
"""

# 3rd party
import os
# import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.INFO)
# tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
# import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# local
# if 'nobackup' in os.path.dirname(__file__):
#     from src.estimator_util import InputFn, ModelFn, CNN1dModel
#     from src.eval_results import eval_model
#     from src.config import Config
# else:
from src.estimator_util import InputFn, ModelFn, CNN1dModel, get_model_dir, get_data_from_tfrecord
import src.config
import paths
from src_hpo import utils_hpo


def draw_plots(res, save_path, opt_metric, output_cl, min_optmetric=False):
    """ Draw loss and evaluation metric plots.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set.
    :param save_path: str, filepath used to save the plots figure.
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param output_cl: dict, predicted outputs per class in each dataset
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective
    epoch. If False, gets the maximum value.
    :return:
    """

    epochs = np.arange(1, len(res['training']['loss']) + 1)
    # min_val_loss, ep_idx = np.min(res['validation loss']), np.argmin(res['validation loss'])
    # choose epoch associated with the best value for the metric
    if min_optmetric:
        ep_idx = np.argmin(res['validation'][opt_metric])
    else:
        ep_idx = np.argmax(res['validation'][opt_metric])

    # plot loss and optimization metric as function of the epochs
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
    ax[1].grid(True)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel(opt_metric)
    ax[1].set_title('%s\nVal/Test %.4f/%.4f' % (opt_metric, res['validation'][opt_metric][ep_idx],
                                                               res['test'][opt_metric][ep_idx]))
    ax[1].legend(loc="lower right")
    f.suptitle('Epochs = {:.0f}(Best val:{:.0f})'.format(epochs[-1], epochs[ep_idx]))
    f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.savefig(save_path + '_plotseval_epochs{:.0f}.png'.format(epochs[-1]))
    plt.close()

    # plot precision, recall, roc auc, pr auc curves for the validation and test sets
    f, ax = plt.subplots()
    ax.plot(epochs, res['validation']['precision'], label='Val Precision')
    ax.plot(epochs, res['validation']['recall'], label='Val Recall')
    ax.plot(epochs, res['validation']['roc auc'], label='Val ROC AUC')
    ax.plot(epochs, res['validation']['pr auc'], label='Val PR AUC')
    ax.plot(epochs, res['test']['precision'], label='Test Precision')
    ax.plot(epochs, res['test']['recall'], label='Test Recall')
    ax.plot(epochs, res['test']['roc auc'], label='Test ROC AUC')
    ax.plot(epochs, res['test']['pr auc'], label='Test PR AUC')
    ax.grid(True)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    ax.set_xlim([0, epochs[-1] + 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.set_title('Evaluation Metrics\nVal/Test')
    # ax[1].legend(loc="lower right")
    # f.suptitle('Epochs = {:.0f}'.format(res['epochs'][-1]))
    # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.savefig(save_path + '_prec_rec_auc.png')
    plt.close()

    # plot pr curve
    f, ax = plt.subplots()
    ax.plot(res['validation']['rec thr'][-1], res['validation']['prec thr'][-1],
            label='Val (AUC={:.3f})'.format(res['validation']['pr auc'][-1]), color='r')
    ax.plot(res['test']['rec thr'][-1], res['test']['prec thr'][-1],
            label='Test (AUC={:.3f})'.format(res['test']['pr auc'][-1]), color='b')
    ax.plot(res['training']['rec thr'][-1], res['training']['prec thr'][-1],
            label='Train (AUC={:.3f})'.format(res['training']['pr auc'][-1]), color='k')
    # CHANGE THR_VEC ACCORDINGLY TO THE SAMPLED THRESHOLD VALUES
    thr_vec = np.linspace(0, 999, 11, endpoint=True, dtype='int')
    ax.scatter(np.array(res['validation']['rec thr'][-1])[thr_vec],
               np.array(res['validation']['prec thr'][-1])[thr_vec], c='r')
    ax.scatter(np.array(res['test']['rec thr'][-1])[thr_vec],
               np.array(res['test']['prec thr'][-1])[thr_vec], c='b')
    ax.scatter(np.array(res['training']['rec thr'][-1])[thr_vec],
               np.array(res['training']['prec thr'][-1])[thr_vec], c='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.grid(True)
    ax.legend(loc='bottom left')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Recall curve')
    f.savefig(save_path + '_prec_rec.png')
    plt.close()

    # plot histogram of the class distribution as a function of the predicted output
    bins = np.linspace(0, 1, 11, True)
    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set'}
    for dataset in output_cl:

        hist, bin_edges = {}, {}
        for class_label in output_cl[dataset]:
            counts_cl = list(np.histogram(output_cl[dataset][class_label], bins, density=False, range=(0, 1)))
            counts_cl[0] = counts_cl[0] / len(output_cl[dataset][class_label])
            hist[class_label] = counts_cl[0]
            bin_edges[class_label] = counts_cl[1]

        bins_multicl = np.linspace(0, 1, len(output_cl[dataset]) * 10 + 1, True)
        bin_width = bins_multicl[1] - bins_multicl[0]
        bins_cl = {}
        for i, class_label in enumerate(output_cl[dataset]):
            bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                    for idx in range(i, len(bins_multicl) - 1, len(output_cl[dataset]))]

        f, ax = plt.subplots()
        for class_label in output_cl[dataset]:
            ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
        ax.set_ylabel('Class fraction')
        ax.set_xlabel('Predicted output')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Output distribution')
        ax.set_xticks(np.linspace(0, 1, 11, True))
        ax.legend()
        ax.set_title(dataset_names[dataset])
        plt.savefig(save_path + 'class_predoutput_distribution_{}.png'.format(dataset))
        plt.close()


def run_main(config, n_epochs, data_dir, model_dir, res_dir, opt_metric, min_optmetric):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: configuration object from the Config class
    :param n_epochs: int, number of epochs to train the models
    :param data_dir: str, path to directory with the tfrecord files
    :param model_dir: str, path to directory in which to save the models
    :param res_dir: str, path to directory to save results
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective epoch.
    If False, gets the maximum value.
    :return:
    """

    # get labels for each dataset
    labels = {dataset: [] for dataset in ['train', 'val', 'test']}
    for tfrec_file in os.listdir(data_dir):
        if 'test' in tfrec_file:
            labels['test'] += get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), ['labels'],
                                                      config['label_map'])['labels']
        elif 'val' in tfrec_file:
            labels['val'] += get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), ['labels'],
                                                     config['label_map'])['labels']
        elif 'train' in tfrec_file:
            labels['train'] += get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), ['labels'],
                                                       config['label_map'])['labels']
    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in ['train', 'val', 'test']}

    config_sess = None  # tf.ConfigProto(log_device_placement=False)

    classifier = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                        config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                      session_config=config_sess),
                                        model_dir=get_model_dir(model_dir)
                                        )

    train_input_fn = InputFn(file_pattern=data_dir + '/train*', batch_size=config['batch_size'],
                             mode=tf.estimator.ModeKeys.TRAIN, label_map=config['label_map'],
                             centr_flag=config['centr_flag'])
    val_input_fn = InputFn(file_pattern=data_dir + '/val*', batch_size=config['batch_size'],
                           mode=tf.estimator.ModeKeys.EVAL, label_map=config['label_map'],
                           centr_flag=config['centr_flag'])
    test_input_fn = InputFn(file_pattern=data_dir + '/test*', batch_size=config['batch_size'],
                            mode=tf.estimator.ModeKeys.EVAL, label_map=config['label_map'],
                            centr_flag=config['centr_flag'])

    # METRIC LIST DEPENDS ON THE METRICS COMPUTED FOR THE ESTIMATOR
    metrics_list = ['loss', 'accuracy', 'pr auc', 'precision', 'recall', 'roc auc', 'prec thr', 'rec thr']
    dataset_ids = ['training', 'validation', 'test']
    res = {dataset: {metric: [] for metric in metrics_list} for dataset in
           dataset_ids}
    for epoch_i in range(1, n_epochs + 1):  # Train and evaluate the model for n_epochs

        print('\n\x1b[0;33;33m' + "Starting epoch %d of %d for %s" %
              (epoch_i, n_epochs, res_dir.split('/')[-1]) + '\x1b[0m\n')
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

    # predict on given datasets
    predictions_dataset = {dataset: [] for dataset in ['train', 'val', 'test']}
    for dataset in predictions_dataset:
        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*', batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT, label_map=config['label_map'],
                                   centr_flag=config['centr_flag'])

        for predictions in classifier.predict(predict_input_fn):
            predictions_dataset[dataset].append(predictions[0])
        predictions_dataset[dataset] = np.array(predictions_dataset[dataset], dtype='float')

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in ['train', 'val', 'test']}
    for dataset in output_cl:
        # map_labels
        for class_label in config['label_map']:

            if class_label == 'AFP':
                continue
            elif class_label == 'NTP':
                output_cl[dataset]['NTP+AFP'] = \
                    predictions_dataset[dataset][np.where(labels[dataset] == config['label_map'][class_label])]
            else:
                output_cl[dataset][class_label] = \
                    predictions_dataset[dataset][np.where(labels[dataset] == config['label_map'][class_label])]

    # save results in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(res_dir + 'res_eval.npy', res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    draw_plots(res, res_dir, opt_metric, output_cl, min_optmetric=min_optmetric)

    print('Saving metrics to a txt file...')
    # write results to a txt file
    with open(res_dir + "res_eval.txt", "a") as res_file:
        res_file.write('{} {} - Epoch {} {}\n'.format('#' * 10, res_dir.split('/')[-1], n_epochs, '#' * 10))
        for dataset in dataset_ids:
            res_file.write('Dataset: {}\n'.format(dataset))
            for metric in metrics_list:
                if metric not in ['prec thr', 'rec thr']:
                    res_file.write('{}: {}\n'.format(metric, res[dataset][metric][-1]))
            res_file.write('\n')
        res_file.write('{}'.format('-' * 100))
        res_file.write('\n')

    print('#' * 100)
    print('Performance on last epoch ({})'.format(n_epochs))
    for dataset in dataset_ids:
        print(dataset)
        for metric in metrics_list:
            if metric not in ['prec thr', 'rec thr']:
                print('{}: {}'.format(metric, res[dataset][metric][-1]))
    print('#' * 100)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Shallue's best configuration
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': None,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}
    ######### SCRIPT PARAMETERS #############################################

    study = 'study_bohb_dr25_tcert_spline2'
    # set configuration manually
    config = None

    # tfrecord files directory
    # tfrec_dir = paths.tfrec_dir_DR25_TCERT  # paths.tfrec_dir
    tfrec_dir = paths.tfrec_dir['DR25']['spline']['TCERT']

    n_models = 10  # number of models in the ensemble
    n_epochs = 300
    multi_class = False
    use_kepler_ce = False
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'
    opt_metric = 'pr auc'  # choose which metric to plot side by side with the loss
    min_optmetric = False  # if lower value is better set to True

    # set the configuration from a HPO study
    if config is None:
        res = utils_hpo.logged_results_to_HBS_result(paths.path_hpoconfigs + study,
                                                     '_' + study
                                                     )
        # res = utils_hpo.logged_results_to_HBS_result(paths.path_hpoconfigs + study, '_' + study)
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        config = id2config[incumbent]['config']
        # best_config = id2config[(8, 0, 3)]['config']

    print('Selected configuration: ', config)

    ######### SCRIPT PARAMETERS #############################################

    # results directory
    save_path = paths.pathtrainedmodels + study
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + '/models/')

    # add dataset parameters
    config = src.config.add_dataset_params(tfrec_dir, satellite, multi_class, config)

    # add missing parameters in hpo with default values
    config = src.config.add_default_missing_params(config=config)
    print('Configuration used: ', config)

    for item in range(5, n_models):
        print('Training model %i out of %i on %i' % (item + 1, n_models, n_epochs))
        run_main(config=config,
                 n_epochs=n_epochs,
                 data_dir=tfrec_dir,
                 model_dir=save_path + '/models/',
                 res_dir=save_path + '/model%i' % (item + 1),
                 opt_metric=opt_metric,
                 min_optmetric=min_optmetric)

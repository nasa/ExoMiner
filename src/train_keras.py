"""
Train models using a given configuration obtained on a hyperparameter optimization study.

TODO: allocate several models to the same GPU
      figure out the logging
      add argument to choose model used
      make draw_plots function compatible with TESS
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.INFO)
# tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
# from mpi4py import MPI
import time
from tensorflow import keras
from tensorflow.keras import callbacks, losses, optimizers
import argparse

# local
import paths
if 'home6' in paths.path_hpoconfigs:
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from src.utils_dataio import InputFn, get_data_from_tfrecord, get_data_from_tfrecord_kepler
from src.models_keras import CNN1dPlanetFinderv1
import src.config_keras
from src.utils_metrics import get_metrics
from src_hpo import utils_hpo


def draw_plots(res, save_path, opt_metric, output_cl, model_id, min_optmetric, earlystopping):
    """ Draw loss and evaluation metric plots.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set.
    :param save_path: str, filepath used to save the plots figure.
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param output_cl: dict, predicted outputs per class in each dataset
    :param model_id: int, identifies the model being used
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective
    epoch. If False, gets the maximum value.
    :param earlystopping:
    :return:
    """

    epochs = np.arange(1, len(res['loss']) + 1)

    # min_val_loss, ep_idx = np.min(res['validation loss']), np.argmin(res['validation loss'])
    # choose epoch associated with the best value for the metric
    if earlystopping is not None:
        if min_optmetric:
            ep_idx = np.argmin(res['val_{}'.format(opt_metric)])
        else:
            ep_idx = np.argmax(res['val_{}'.format(opt_metric)])
    else:
        ep_idx = -1

    # plot loss and optimization metric as function of the epochs
    f, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, res['loss'], label='Training', color='b')
    ax[0].plot(epochs, res['val_loss'], label='Validation', color='r')

    # ax[0].scatter(epochs[ep_idx], res['validation']['loss'][ep_idx], c='r')
    ax[0].scatter(epochs[ep_idx], res['test_loss'], c='k', label='Test')
    ax[0].set_xlim([0, epochs[-1] + 1])
    # ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Categorical cross-entropy\nVal/Test %.4f/%.4f' % (res['val_loss'][ep_idx],
                                                                       res['test_loss']))
    ax[0].legend(loc="upper right")
    ax[0].grid(True)
    ax[1].plot(epochs, res[opt_metric], label='Training')
    ax[1].plot(epochs, res['val_{}'.format(opt_metric)], label='Validation', color='r')
    ax[1].scatter(epochs[ep_idx], res['val_{}'.format(opt_metric)][ep_idx], c='r')
    ax[1].scatter(epochs[ep_idx], res['test_{}'.format(opt_metric)], label='Test', c='k')
    ax[1].set_xlim([0, epochs[-1] + 1])
    # ax[1].set_ylim([0.0, 1.05])
    ax[1].grid(True)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel(opt_metric)
    ax[1].set_title('%s\nVal/Test %.4f/%.4f' % (opt_metric, res['val_{}'.format(opt_metric)][ep_idx],
                                                res['test_{}'.format(opt_metric)]))
    ax[1].legend(loc="lower right")
    f.suptitle('Epochs = {:.0f}(Best val:{:.0f})'.format(epochs[-1], epochs[ep_idx]))
    f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.savefig(os.path.join(save_path, 'model{}_plotseval_epochs{:.0f}.png'.format(model_id, epochs[-1])))
    plt.close()

    # plot precision, recall, roc auc, pr auc curves for the validation and test sets
    f, ax = plt.subplots()
    ax.plot(epochs, res['val_precision'], label='Val Precision')
    ax.plot(epochs, res['val_recall'], label='Val Recall')
    ax.plot(epochs, res['val_auc_roc'], label='Val ROC AUC')
    ax.plot(epochs, res['val_auc_pr'], label='Val PR AUC')
    ax.scatter(epochs[ep_idx], res['test_precision'], label='Test Precision')
    ax.scatter(epochs[ep_idx], res['test_recall'], label='Test Recall')
    ax.scatter(epochs[ep_idx], res['test_auc_roc'], label='Test ROC AUC')
    ax.scatter(epochs[ep_idx], res['test_auc_pr'], label='Test PR AUC')
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
    f.savefig(os.path.join(save_path, 'model{}_prec_rec_auc.png'.format(model_id)))
    plt.close()

    # plot pr curve
    f, ax = plt.subplots()
    ax.plot(res['val_rec_thr'][ep_idx], res['val_prec_thr'][ep_idx],
            label='Val (AUC={:.3f})'.format(res['val_auc_pr'][ep_idx]), color='r')
    ax.plot(res['test_rec_thr'], res['test_prec_thr'],
            label='Test (AUC={:.3f})'.format(res['test_auc_pr']), color='k')
    ax.plot(res['rec_thr'][ep_idx], res['prec_thr'][ep_idx],
            label='Train (AUC={:.3f})'.format(res['auc_pr'][ep_idx]), color='b')
    # # CHANGE THR_VEC ACCORDINGLY TO THE SAMPLED THRESHOLD VALUES
    # threshold_range = np.linspace(0, num_thresholds - 1, 11, endpoint=True, dtype='int')
    ax.scatter(res['val_rec_thr'][ep_idx],
               res['val_prec_thr'][ep_idx], c='r')
    ax.scatter(res['test_rec_thr'],
               res['test_prec_thr'], c='k')
    ax.scatter(res['rec_thr'][ep_idx],
               res['prec_thr'][ep_idx], c='b')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.grid(True)
    ax.legend(loc='lower left')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Recall curve')
    f.savefig(os.path.join(save_path, 'model{}_prec_rec.png'.format(model_id)))
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
        ax.set_xticks(np.linspace(0, 1, 11, True))
        ax.legend()
        ax.set_title('Output distribution - {}'.format(dataset_names[dataset]))
        plt.savefig(os.path.join(save_path, 'model{}_class_predoutput_distribution_{}.png'.format(model_id, dataset)))
        plt.close()


def run_main(config, n_epochs, data_dir, base_model, model_dir, res_dir, model_id, opt_metric, min_optmetric,
             callbacks_list, features_set, data_augmentation=False, scalar_params_idxs=None, filter_data=None,
             mpi_rank=None, ngpus_per_node=1):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: configuration object from the Config class
    :param n_epochs: int, number of epochs to train the models
    :param data_dir: str, path to directory with the tfrecord files
    :param model_dir: str, path to root directory in which to save the models
    :param res_dir: str, path to directory to save results
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective epoch
    If False, gets the maximum value
    :param callbacks_list: list, callbacks
    :param features_set: dict, each key-value pair is feature_name: {'dim': feature_dim, 'dtype': feature_dtype}
    :param data_augmentation: bool, whether to use or not data augmentation
    # :param filter_data: dict, containing as keys the names of the datasets. Each value is a dict containing as keys the
    # elements of data_fields or a subset, which are used to filter the examples. For 'label', 'kepid' and 'tce_n' the
    # values should be a list; for the other data_fields, it should be a two element list that defines the interval of
    # acceptable values
    :param mpi_rank: int, rank of the mpi process used to distribute the models for the GPUs; set to None when not
    training multiple models in multiple GPUs in parallel
    :param ngpus_per_node: int, number of GPUs per node when training/evaluating multiple models in parallel
    :return:
    """

    if mpi_rank is None or mpi_rank == 0:
        print('Configuration used: ', config)

    # create directory for the model
    model_dir_sub = os.path.join(model_dir, 'model{}'.format(model_id))
    os.makedirs(model_dir_sub, exist_ok=True)

    # datasets - same name convention as used for the TFRecords
    datasets = ['train', 'val', 'test']

    if filter_data is None:
        filter_data = {dataset: None for dataset in datasets}

    # get labels for each dataset
    labels = {dataset: [] for dataset in datasets}

    tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # find which dataset the TFRecord is from
        dataset = tfrec_file.split('-')[0]

        # labels[dataset] += get_data_from_tfrecord_kepler(os.path.join(data_dir, tfrec_file), ['label'],
        #                                                  config['label_map'], filt=filter_data[dataset])['label']
        labels[dataset] += get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), ['label'],
                                                         config['label_map'], filt=filter_data[dataset])['label']

    # convert from list to numpy array
    # TODO: should make this a numpy array from the beginning
    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in datasets}

    # set visible GPU as function of the MPI rank and number of GPUs per node
    if mpi_rank is not None:
        gpu_id = mpi_rank % ngpus_per_node
    else:
        gpu_id = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)

    # instantiate Keras model
    model = base_model(config, features_set, scalar_params_idxs).kerasModel

    # print model summary
    if mpi_rank is None or mpi_rank == 0:
        model.summary()

    # setup metrics to be monitored
    metrics_list = get_metrics()

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
                                               momentum=config['momentum'],
                                               nesterov=False,
                                               name='SGD'),  # optimizer
                      # loss function to minimize
                      loss=losses.BinaryCrossentropy(from_logits=False,
                                                     label_smoothing=0,
                                                     name='binary_crossentropy'),
                      # list of metrics to monitor
                      metrics=metrics_list)

    # input function for training on the training set
    train_input_fn = InputFn(file_pattern=data_dir + '/train*',
                             batch_size=config['batch_size'],
                             mode=tf.estimator.ModeKeys.TRAIN,
                             label_map=config['label_map'],
                             data_augmentation=data_augmentation,
                             filter_data=filter_data['train'],
                             features_set=features_set,
                             scalar_params_idxs=scalar_params_idxs)

    # # input functions for evaluation on the training, validation and test sets
    # traineval_input_fn = InputFn(file_pattern=data_dir + '/train*',
    #                              batch_size=config['batch_size'],
    #                              mode=tf.estimator.ModeKeys.EVAL,
    #                              label_map=config['label_map'],
    #                              filter_data=filter_data['train'],
    #                              features_set=features_set,
    #                              scalar_params_idxs=scalar_params_idxs)
    val_input_fn = InputFn(file_pattern=data_dir + '/val*',
                           batch_size=config['batch_size'],
                           mode=tf.estimator.ModeKeys.EVAL,
                           label_map=config['label_map'],
                           filter_data=filter_data['val'],
                           features_set=features_set,
                           scalar_params_idxs=scalar_params_idxs)
    test_input_fn = InputFn(file_pattern=data_dir + '/test*',
                            batch_size=config['batch_size'],
                            mode=tf.estimator.ModeKeys.EVAL,
                            label_map=config['label_map'],
                            filter_data=filter_data['test'],
                            features_set=features_set,
                            scalar_params_idxs=scalar_params_idxs)

    # fit the model to the training data
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_split=0.,
                        validation_data=val_input_fn(),
                        shuffle=True,  # does the input function shuffle for every epoch?
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None,
                        max_queue_size=10,  # does not matter when using input function with tf.data API
                        workers=1,  # same
                        use_multiprocessing=False  # same
                        )

    res = history.history

    res_eval = model.evaluate(x=test_input_fn(),
                              y=None,
                              batch_size=None,
                              verbose=1,
                              sample_weight=None,
                              steps=None,
                              callbacks=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False)

    # add test set metrics to result
    for metric_name_i, metric_name in enumerate(model.metrics_names):
        res['test_{}'.format(metric_name)] = res_eval[metric_name_i]

    # if mpi_rank is not None:
    #     sys.stdout.flush()

    # predict on given datasets - needed for computing the output distribution
    predictions_dataset = {dataset: [] for dataset in datasets}
    for dataset in predictions_dataset:

        print('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*',
                                   batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   label_map=config['label_map'],
                                   filter_data=filter_data[dataset],
                                   features_set=features_set,
                                   scalar_params_idxs=scalar_params_idxs)

        predictions_dataset[dataset] = model.predict(predict_input_fn(),
                                                     batch_size=None,
                                                     verbose=1,
                                                     steps=None,
                                                     callbacks=None,
                                                     max_queue_size=10,
                                                     workers=1,
                                                     use_multiprocessing=False)

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in datasets}
    for dataset in output_cl:
        # map_labels
        for class_label in config['label_map']:
            output_cl[dataset][class_label] = predictions_dataset[dataset][
                np.where(labels[dataset] == config['label_map'][class_label])]

    # save results in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(os.path.join(model_dir_sub, 'results.npy'), res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    draw_plots(res, res_dir, opt_metric, output_cl, model_id, min_optmetric, earlystopping=None)

    print('Saving metrics to a txt file...')
    # write results to a txt file
    with open(os.path.join(model_dir_sub, 'results.txt'), 'w') as res_file:
        res_file.write('Performance metrics at epoch {} \n'.format(len(res['loss'])))
        for dataset in datasets:
            res_file.write('Dataset: {}\n'.format(dataset))
            for metric in model.metrics_names:
                if 'prec_thr' not in metric and 'rec_thr' not in metric:
                    if dataset == 'train':
                        res_file.write('{}: {}\n'.format(metric, res['{}'.format(metric)][-1]))
                    elif dataset == 'test':
                        res_file.write('{}: {}\n'.format(metric, res['test_{}'.format(metric)]))
                    else:
                        res_file.write('{}: {}\n'.format(metric, res['val_{}'.format(metric)][-1]))
            res_file.write('\n')
        res_file.write('{}'.format('-' * 100))
        res_file.write('\n')

    print('#' * 100)
    print('Model {}'.format(model_id))
    print('Performance on epoch ({})'.format(len(res['loss'])))
    for dataset in datasets:
        print(dataset)
        for metric in model.metrics_names:
            if 'prec_thr' not in metric and 'rec_thr' not in metric:
                if dataset == 'train':
                    print('{}: {}\n'.format(metric, res['{}'.format(metric)][-1]))
                elif dataset == 'test':
                    print('{}: {}\n'.format(metric, res['test_{}'.format(metric)]))
                else:
                    print('{}: {}\n'.format(metric, res['val_{}'.format(metric)][-1]))
    print('#' * 100)

    # save model, features and config used for training this model
    model.save(os.path.join(model_dir_sub, 'model{}.h5'.format(model_id)))
    np.save(os.path.join(model_dir_sub, 'features_set'), features_set)
    np.save(os.path.join(model_dir_sub, 'config'), config)
    # plot model and save the figure created
    # keras.utils.plot_model(model,
    #                        to_file=os.path.join(model_dir_sub, 'model.png'),
    #                        show_shapes=False,
    #                        show_layer_names=True,
    #                        rankdir='TB',
    #                        expand_nested=False,
    #                        dpi=96)


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    ngpus_per_node = 1  # number of GPUs per node

    # # uncomment for MPI multiprocessing
    # rank = MPI.COMM_WORLD.rank
    # rank = ngpus_per_node * args.job_idx + rank
    # size = MPI.COMM_WORLD.size
    # print('Rank={}/{}'.format(rank, size - 1))
    # sys.stdout.flush()
    # if rank != 0:
    #     time.sleep(2)

    # tf.logging.set_verbosity(tf.logging.ERROR)
    # tf.logging.set_verbosity(tf.logging.INFO)

    # SCRIPT PARAMETERS #############################################

    # name of the study
    study = 'keras_test'

    # results directory
    save_path = os.path.join(paths.pathtrainedmodels, study)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'models'))

    # name of the HPO study from which to get a configuration; config needs to be set to None
    hpo_study = 'bohb_dr25tcert_spline_gapped_glflux'

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = CNN1dPlanetFinderv1

    ngpus_per_node = 1  # number of GPUs per node

    # set configuration manually. Set to None to use a configuration from a HPO study
    config = None

    # example of configuration
    # config = {'batch_size': 32,
    #           'conv_ls_per_block': 2,
    #           'dropout_rate': 0.0053145468133186415,
    #           'init_conv_filters': 3,
    #           'init_fc_neurons': 128,
    #           'kernel_size': 8,
    #           'kernel_stride': 2,
    #           'lr': 0.015878640426114688,
    #           'num_fc_layers': 3,
    #           'num_glob_conv_blocks': 2,
    #           'num_loc_conv_blocks': 2,
    #           'optimizer': 'SGD',
    #           'pool_size_glob': 2,
    #           'pool_size_loc': 2,
    #           'pool_stride': 1,
    #           'sgd_momentum': 0.024701642898564722}

    # tfrecord files directory
    # tfrec_dir = os.path.join(paths.path_tfrecs,
    #                          'Kepler/tfrecordkeplerdr25_flux-centroid_selfnormalized-oddeven'
    #                          '_nonwhitened_gapped_2001-201_updtKOIs')
    tfrec_dir = os.path.join(paths.path_tfrecs,
                             'Kepler',
                             'DR25',
                             'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven_'
                             'updtkois_shuffled_wks_norm')

    # features to be extracted from the dataset(s)
    features_names = ['global_view', 'local_view', 'global_view_centr', 'local_view_centr', 'local_view_odd',
                      'local_view_even']
    features_dim = {feature_name: (2001, 1) if 'global' in feature_name else (201, 1)
                    for feature_name in features_names}
    features_names.append('scalar_params')  # use scalar parameters as input features
    features_dim['scalar_params'] = (12,)  # dimension of the scalar parameter array in the TFRecords
    # choose indexes of scalar parameters to be extracted as features; None to get all of them in the TFRecords
    scalar_params_idxs = [1, 2]
    features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    features_set = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
                    for feature_name in features_names}

    # example of feature set
    # features_set = {'global_view': {'dim': (2001,), 'dtype': tf.float32},
    #                 'local_view': {'dim': (201,), 'dtype': tf.float32}}

    data_augmentation = False  # if True, uses data augmentation in the training set

    # args_inputfn = {'features_set': features_set, 'data_augmentation': data_augmentation,
    #                 'scalar_params_idxs': scalar_params_idxs}

    n_models = 10  # number of models in the ensemble
    n_epochs = 300  # number of epochs used to train each model
    multi_class = False  # multiclass classification
    ce_weights_args = {'tfrec_dir': tfrec_dir, 'datasets': ['train'], 'label_fieldname': 'label',
                       'verbose': False}
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    opt_metric = 'auc_pr'  # choose which metric to plot side by side with the loss
    min_optmetric = False  # if lower value is better set to True

    # callbacks list
    callbacks_list = []

    # early stopping callback
    callbacks_list.append(callbacks.EarlyStopping(monitor='val_{}'.format(opt_metric), min_delta=0,
                                                  patience=20,
                                                  verbose=1,
                                                  mode='max',
                                                  baseline=None,
                                                  restore_best_weights=True))

    # TensorBoard callback
    callbacks_list.append(callbacks.TensorBoard(log_dir=os.path.join(save_path, 'models'),
                                                histogram_freq=0,
                                                write_graph=False,
                                                write_images=True,
                                                update_freq='epoch',
                                                profile_batch=2,
                                                ))

    # set the configuration from a HPO study
    if config is None:
        res = utils_hpo.logged_results_to_HBS_result(os.path.join(paths.path_hpoconfigs, hpo_study)
                                                     , '_{}'.format(hpo_study))
        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config = id2config[incumbent]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(8, 0, 3)]['config']

    # SCRIPT PARAMETERS #############################################

    # add dataset parameters
    config = src.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

    # add missing parameters in hpo with default values
    config = src.config_keras.add_default_missing_params(config=config)

    # comment for multiprocessing using MPI
    for model_i in range(n_models):
        print('Training model %i out of %i on %i epochs...' % (model_i + 1, n_models, n_epochs))
        run_main(config=config,
                 n_epochs=n_epochs,
                 data_dir=tfrec_dir,
                 base_model=BaseModel,
                 model_dir=os.path.join(save_path, 'models'),
                 res_dir=save_path,
                 model_id=model_i + 1,
                 opt_metric=opt_metric,
                 min_optmetric=min_optmetric,
                 callbacks_list=callbacks_list,
                 features_set=features_set,
                 scalar_params_idxs=scalar_params_idxs,
                 data_augmentation=data_augmentation)

    # # uncomment for multiprocessing using MPI
    # if rank < n_models:
    #     print('Training model %i out of %i on %i' % (rank + 1, n_models, n_epochs))
    #     sys.stdout.flush()
    #     run_main(config=config,
    #              n_epochs=n_epochs,
    #              data_dir=tfrec_dir,
    #              base_model=BaseModel,
    #              model_dir=os.path.join(save_path, 'models'),
    #              res_dir=save_path,
    #              model_id = rank + 1,
    #              opt_metric=opt_metric,
    #              min_optmetric=min_optmetric,
    #              callbacks_list=callbacks_list,
    #              features_set=features_set,
    #              scalar_params_idxs=scalar_params_idxs,
    #              mpi_rank=rank,
    #              ngpus_per_node=ngpus_per_node)

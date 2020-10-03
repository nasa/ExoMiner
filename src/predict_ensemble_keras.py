"""
Perform inference using an ensemble of Tensorflow Keras models.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import load_model
import pandas as pd


# local
import paths
if 'home6' in paths.path_hpoconfigs:
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from src.utils_dataio import InputFn, get_data_from_tfrecord
from src.models_keras import create_ensemble
import src.config_keras
from src_hpo import utils_hpo
from src.utils_metrics import get_metrics


def print_metrics(res, datasets, metrics_names, prec_at_top):
    """

    :param res:
    :param datasets:
    :param metrics_names:
    :param prec_at_top:
    :return:
    """

    print('#' * 100)
    print('Performance metrics for the ensemble ({} models)\n'.format(len(models_filepaths)))
    for dataset in datasets:
        if dataset != 'predict':
            print(dataset)
            for metric in metrics_names:
                if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                    print('{}: {}\n'.format(metric, res['{}_{}'.format(dataset, metric)]))

            for k in prec_at_top[dataset]:
                print('{}: {}\n'.format('{}_precision_at_{}'.format(dataset, k),
                                        res['{}_precision_at_{}'.format(dataset, k)]))
    print('#' * 100)


def save_metrics_to_file(save_path, res, datasets, metrics_names, prec_at_top):
    """ Write results to a txt file.

    :param model_dir_sub:
    :param res:
    :param datasets:
    :param metrics_names:
    :param prec_at_top:
    :return:
    """

    # write results to a txt file
    with open(os.path.join(save_path, 'results_ensemble.txt'), 'w') as res_file:

        res_file.write('Performance metrics for the ensemble ({} models)\n'.format(len(models_filepaths)))

        for dataset in datasets:

            if dataset != 'predict':
                res_file.write('Dataset: {}\n'.format(dataset))
                for metric in metrics_names:
                    if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                        res_file.write('{}: {}\n'.format(metric, res['{}_{}'.format(dataset, metric)]))

                for k in prec_at_top[dataset]:
                    res_file.write('{}: {}\n'.format('{}_precision_at_{}'.format(dataset, k),
                                                     res['{}_precision_at_{}'.format(dataset, k)]))

            res_file.write('\n')

        res_file.write('{}'.format('-' * 100))

        res_file.write('\nModels used to create the ensemble:\n')

        for model_filepath in models_filepaths:
            res_file.write('{}\n'.format(model_filepath))

        res_file.write('\n')


def draw_plots_prcurve_roc(res, save_path, dataset):
    """ Plot ROC and PR curves.

    :param res: dict, each key is a specific dataset ('train', 'val', ...) and the values are dicts that contain
    metrics for each dataset
    :param save_path: str, path to save directory
    :param dataset: str, dataset for which the plot is generated
    :return:
    """

    # count number of samples per class to compute TPR and FPR
    num_samples_per_class = {'positive': 0, 'negative': 0}
    num_samples_per_class['positive'] = res['{}_tp'.format(dataset)][0] + res['{}_fn'.format(dataset)][0]
    num_samples_per_class['negative'] = res['{}_fp'.format(dataset)][0] + res['{}_tn'.format(dataset)][0]

    # ROC and PR curves
    f = plt.figure(figsize=(9, 6))
    lw = 2
    ax = f.add_subplot(111, label='PR ROC')
    ax.plot(res['{}_rec_thr'.format(dataset)], res['{}_prec_thr'.format(dataset)], color='darkorange', lw=lw,
            label='PR ROC curve (area = %0.3f)' % res['{}_auc_pr'.format(dataset)])
    # ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax2 = f.add_subplot(111, label='AUC ROC', frame_on=False)
    ax2.plot(res['{}_fp'.format(dataset)] / num_samples_per_class['negative'],
             res['{}_tp'.format(dataset)] / num_samples_per_class['positive'],
             color='darkorange', lw=lw, linestyle='--',
             label='AUC ROC curve (area = %0.3f)' % res['{}_auc_roc'.format(dataset)])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xticks(np.arange(0, 1.05, 0.05))
    ax2.set_yticks(np.arange(0, 1.05, 0.05))
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.set_label_position('top')
    ax2.legend(loc="lower left")

    f.suptitle('PR/ROC Curves - {}'.format(dataset))
    f.savefig(os.path.join(save_path, 'ensemble_pr-roc_curves_{}.png'.format(dataset)))
    plt.close()


def draw_plots_scores_distribution(output_cl, save_path):
    """ Plot distribution of scores.

    :param output_cl: dict, each key is a specific dataset ('train', 'val', ...) and the values are dicts that contain
    the scores for each class ('PC', 'non-PC', for example)
    :param save_path: str, path to save directory
    :return:
    """

    # plot histogram of the class distribution as a function of the predicted output
    bins = np.linspace(0, 1, 11, True)
    for dataset in output_cl:

        hist, bin_edges = {}, {}
        for class_label in output_cl[dataset]:
            counts_cl = list(np.histogram(output_cl[dataset][class_label], bins, density=False, range=(0, 1)))
            counts_cl[0] = counts_cl[0] / max(len(output_cl[dataset][class_label]), 1e-7)
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
        if dataset == 'predict':
            ax.set_ylabel('Dataset fraction')
        else:
            ax.set_ylabel('Class fraction')
        ax.set_xlabel('Predicted output')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Output distribution - {}'.format(dataset))
        ax.set_xticks(np.linspace(0, 1, 11, True))
        if dataset != 'predict':
            ax.legend()
        plt.savefig(os.path.join(save_path, 'ensemble_class_scoredistribution_{}.png'.format(dataset)))
        plt.close()


def plot_precision_at_k(labels_ord, k_curve_arr, dataset, save_path):
    """ Plot precision-at-k and misclassified-at-k curves.

    :param labels_ord:
    :param k_curve_arr:
    :param save_path:
    :return:
    """

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr}
    for k_i in range(len(k_curve_arr)):
        if len(labels_ord) < k_curve_arr[k_i]:
            precision_at_k[k_curve_arr[k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[k_i]] = \
                np.sum(labels_ord[-k_curve_arr[k_i]:]) / k_curve_arr[k_i]

    # precision at k curve
    f, ax = plt.subplots()
    ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
    ax.set_ylim(top=1)
    f.savefig(os.path.join(save_path, 'precisionatk_{}.svg'.format(dataset)))
    plt.close()

    # misclassified examples at k curve
    f, ax = plt.subplots()
    kvalues = np.array(list(precision_at_k.keys()))
    precvalues = np.array(list(precision_at_k.values()))
    num_misclf_examples = kvalues - kvalues * precvalues
    ax.plot(kvalues, num_misclf_examples)
    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
    f.savefig(os.path.join(save_path, 'ensemble_misclassified_at_k_{}.svg'.format(dataset)))
    plt.close()


def run_main(config, features_set, clf_thr, data_dir, res_dir, models_filepaths, datasets, fields, generate_csv_pred,
             scalar_params_idxs=None):
    """ Evaluate model on a given configuration in the specified datasets and also predict on them.

    :param config: configuration object from the Config class
    :param features_set: dict, each key-value pair is feature_name: {'dim': feature_dim, 'dtype': feature_dtype}
    :param clf_thr: float, classification threshold
    :param data_dir: str, path to directory with the tfrecord files
    :param res_dir: str, path to directory to save results
    :param models_filepaths: list, filepaths to models to be integrated in the ensemble
    :param datasets: list, datasets in which the ensemble is evaluated/predicted on
    :param fields: list, fields to extract from the TFRecords datasets
    :param generate_csv_pred: bool, if True also generates a prediction ranking for the specified datasets
    :param scalar_params_idxs: list, indexes of scalar parameters in the TFRecords to be used as features
    # :param mpi_rank: int, rank of the mpi process used to distribute the models for the GPUs; set to None when not
    # training multiple models in multiple GPUs in parallel
    # :param ngpus_per_node: int, number of GPUs per node when training/evaluating multiple models in parallel
    :return:
    """

    verbose = False if 'home6' in paths.path_hpoconfigs else True

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in fields} for dataset in datasets}

    tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # get dataset of the TFRecord
        dataset = tfrec_file.split('-')[0]

        if dataset == 'predict':
            fields_aux = list(fields)
            if 'label' in fields:
                fields_aux.remove('label')
            if 'original_label' in fields:
                fields_aux.remove('original_label')
        else:
            fields_aux = fields

        data_aux = get_data_from_tfrecord(os.path.join(tfrec_dir, tfrec_file), fields_aux, config['label_map'])

        for field in data_aux:
            data[dataset][field].extend(data_aux[field])

    # convert from list to numpy array
    # TODO: should make this a numpy array from the beginning
    for dataset in datasets:
        data[dataset]['label'] = np.array(data[dataset]['label'])
        data[dataset]['original_label'] = np.array(data[dataset]['original_label'])

    # create ensemble
    model_list = []
    for model_i, model_filepath in enumerate(models_filepaths):

        model = load_model(filepath=model_filepath, compile=False)
        model._name = 'model{}'.format(model_i)

        model_list.append(model)

    ensemble_model = create_ensemble(features=features_set, scalar_params_idxs=scalar_params_idxs, models=model_list)

    ensemble_model.summary()

    # set up metrics to be monitored
    metrics_list = get_metrics(clf_threshold=clf_thr)

    # compile model - set optimizer, loss and metrics
    if config['optimizer'] == 'Adam':
        ensemble_model.compile(optimizer=optimizers.Adam(learning_rate=config['lr'],
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
        ensemble_model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                                        momentum=config['sgd_momentum'],
                                                        nesterov=False,
                                                        name='SGD'),  # optimizer
                               # loss function to minimize
                               loss=losses.BinaryCrossentropy(from_logits=False,
                                                              label_smoothing=0,
                                                              name='binary_crossentropy'),
                               # list of metrics to monitor
                               metrics=metrics_list)

    # initialize results dictionary for the evaluated datasets
    res = {}
    for dataset in datasets:

        if dataset == 'predict':
            pass

        print('Evaluating on dataset {}'.format(dataset))

        # input function for evaluating on each dataset
        eval_input_fn = InputFn(file_pattern=data_dir + '/{}*'.format(dataset),
                                batch_size=config['batch_size'],
                                mode=tf.estimator.ModeKeys.EVAL,
                                label_map=config['label_map'],
                                features_set=features_set,
                                scalar_params_idxs=scalar_params_idxs)

        # evaluate model in the given dataset
        res_eval = ensemble_model.evaluate(x=eval_input_fn(),
                                           y=None,
                                           batch_size=None,
                                           verbose=verbose,
                                           sample_weight=None,
                                           steps=None,
                                           callbacks=None,
                                           max_queue_size=10,
                                           workers=1,
                                           use_multiprocessing=False)

        # add evaluated dataset metrics to result dictionary
        for metric_name_i, metric_name in enumerate(ensemble_model.metrics_names):
            res['{}_{}'.format(dataset, metric_name)] = res_eval[metric_name_i]

    # predict on given datasets - needed for computing the output distribution and produce a ranking
    scores = {dataset: [] for dataset in datasets}
    for dataset in scores:

        print('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*',
                                   batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   label_map=config['label_map'],
                                   features_set=features_set,
                                   scalar_params_idxs=scalar_params_idxs)

        scores[dataset] = ensemble_model.predict(predict_input_fn(),
                                                 batch_size=None,
                                                 verbose=verbose,
                                                 steps=None,
                                                 callbacks=None,
                                                 max_queue_size=10,
                                                 workers=1,
                                                 use_multiprocessing=False)

    # initialize dictionary to save the classification scores for each dataset evaluated
    scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8') for dataset in datasets}
    for dataset in datasets:
        # threshold for classification
        scores_classification[dataset][scores[dataset] >= clf_thr] = 1

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in datasets}
    for dataset in output_cl:
        for original_label in config['label_map']:
            # get predictions for each original class individually to compute histogram
            output_cl[dataset][original_label] = scores[dataset][np.where(data[dataset]['original_label'] ==
                                                                          original_label)]

    # compute precision at top-k
    k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
    k_curve_arr = {
        'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
        'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
        'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    }
    labels_sorted = {}
    for dataset in datasets:
        sorted_idxs = np.argsort(scores[dataset], axis=0).squeeze()
        labels_sorted[dataset] = data[dataset]['label'][sorted_idxs].squeeze()

        for k_i in range(len(k_arr[dataset])):
            if len(sorted_idxs) < k_arr[dataset][k_i]:
                res['{}_precision_at_{}'.format(dataset, k_arr[dataset][k_i])] = np.nan
            else:
                res['{}_precision_at_{}'.format(dataset, k_arr[dataset][k_i])] = \
                    np.sum(labels_sorted[dataset][-k_arr[dataset][k_i]:]) / k_arr[dataset][k_i]

    # save evaluation metrics in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(os.path.join(save_path, 'results_ensemble.npy'), res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    for dataset in datasets:
        draw_plots_scores_distribution(output_cl, res_dir)
        if dataset != 'predict':
            draw_plots_prcurve_roc(res, res_dir, dataset)
            plot_precision_at_k(labels_sorted[dataset], k_curve_arr[dataset], dataset, res_dir)

    print('Saving metrics to a txt file...')
    save_metrics_to_file(save_path, res, datasets, ensemble_model.metrics_names, k_arr)

    print_metrics(res, datasets, ensemble_model.metrics_names, k_arr)

    # generate rankings for each evaluated dataset
    if generate_csv_pred:

        print('Generating csv file(s) with ranking(s)...')

        # add predictions to the data dict
        for dataset in datasets:
            data[dataset]['score'] = scores[dataset].ravel()
            data[dataset]['predicted class'] = scores_classification[dataset].ravel()

        # write results to a txt file
        for dataset in datasets:
            print('Saving ranked predictions in dataset {}'
                  ' to {}...'.format(dataset, res_dir + "/ranked_predictions_{}".format(dataset)))
            data_df = pd.DataFrame(data[dataset])

            # sort in descending order of output
            data_df.sort_values(by='score', ascending=False, inplace=True)
            data_df.to_csv(os.path.join(res_dir, "ensemble_ranked_predictions_{}set.csv".format(dataset)), index=False)

    # save model, features and config used for training this model
    ensemble_model.save(os.path.join(save_path, 'ensemble_model.h5'))
    np.save(os.path.join(save_path, 'features_set'), features_set)
    np.save(os.path.join(save_path, 'config'), config)
    # plot ensemble model and save the figure
    keras.utils.plot_model(ensemble_model,
                           to_file=os.path.join(save_path, 'ensemble.png'),
                           show_shapes=False,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=False,
                           dpi=96)


if __name__ == '__main__':

    # SCRIPT PARAMETERS #############################################

    # name of the study
    study = 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu'

    # TFRecord files directory
    tfrec_dir = os.path.join(paths.path_tfrecs,
                             'Kepler',
                             'Q1-Q17_DR25',
                             'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-centroid-centroid_fdl-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-centroid-centroid_fdl-6stellar-bfap-ghost-rollingband_starshuffle_experiment-labels-norm_diffimg_kic_oot_coff-mes-wksmaxmes-wksalbedo-wksptemp-deptherr-perioderr-durationerr'
                             )

    print('Selected TFRecord files directory: {}'.format(tfrec_dir))

    # datasets used; choose from 'train', 'val', 'test', 'predict' - needs to follow naming of TFRecord files
    datasets = ['train', 'val', 'test']
    # datasets = ['predict']

    print('Datasets to be evaluated/tested: {}'.format(datasets))

    # fields to be extracted from the TFRecords and that show up in the ranking created for each dataset
    # set to None if not adding other fields
    # fields = ['target_id', 'label', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label']
    fields = ['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
              'mag', 'ra', 'dec', 'tce_max_mult_ev', 'tce_insol', 'tce_eqt', 'tce_sma', 'tce_prad', 'tce_model_snr',
              'tce_ingress', 'tce_impact', 'tce_incl', 'tce_dor', 'tce_ror']

    print('Fields to be extracted from the TFRecords: {}'.format(fields))

    multi_class = False  # multiclass classification
    ce_weights_args = {'tfrec_dir': tfrec_dir, 'datasets': ['train'], 'label_fieldname': 'label',
                       'verbose': False}
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess

    generate_csv_pred = True

    # set configuration manually. Set to None to use a configuration from a HPO study
    config = None
    # # example of configuration
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

    # name of the HPO study from which to get a configuration; config needs to be set to None
    hpo_study = 'ConfigE-bohb_keplerdr25_g2001-l201_spline_gbal_nongapped_starshuffle_norobovetterkois_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband'

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

    print('Selected configuration: ', config)

    # add dataset parameters
    config = src.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

    # add missing parameters in hpo with default values
    config = src.config_keras.add_default_missing_params(config=config)
    print('Configuration used: ', config)

    clf_thr = 0.5

    # features to be extracted from the dataset
    # features_names = [
    #                   'global_flux_view',
    #                   'local_flux_view',
    #                   # 'global_centr_view',
    #                   # 'local_centr_view',
    #                   # 'local_flux_odd_view',
    #                   # 'local_flux_even_view',
    #                   # 'local_weak_secondary_view'
    # ]
    # features_dim = {feature_name: (2001, 1) if 'global' in feature_name else (201, 1)
    #                 for feature_name in features_names}
    # features_names.append('scalar_params')  # use scalar parameters as input features
    # features_dim['scalar_params'] = (13,)  # dimension of the scalar parameter array in the TFRecords
    # choose indexes of scalar parameters to be extracted as features; None to get all of them in the TFRecords
    scalar_params_idxs = None  # [0, 1, 2, 3, 4, 5]  # [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]
    # features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    # features_set = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
    #                 for feature_name in features_names}

    features_set = {
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
    }

    print('Selected features: {}, {}'.format(features_set, scalar_params_idxs))

    # args_inputfn = {'features_set': features_set, 'scalar_params_idxs': scalar_params_idxs}

    # get models for the ensemble
    models_dir = os.path.join(paths.pathtrainedmodels,
                              study,
                              'models')
    models_filepaths = [os.path.join(models_dir, model_dir, '{}.h5'.format(model_dir))
                        for model_dir in os.listdir(models_dir)
                        if 'model' in model_dir]

    # results directory
    save_path = os.path.join(paths.pathresultsensemble, study)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # SCRIPT PARAMETERS #############################################

    run_main(config=config,
             features_set=features_set,
             scalar_params_idxs=scalar_params_idxs,
             clf_thr=clf_thr,
             data_dir=tfrec_dir,
             res_dir=save_path,
             models_filepaths=models_filepaths,
             datasets=datasets,
             fields=fields,
             generate_csv_pred=generate_csv_pred)

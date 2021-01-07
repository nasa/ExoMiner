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
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# local
import paths
from src.utils_dataio import get_data_from_tfrecord
from src.utils_dataio import InputFnv2 as InputFn
from src.models_keras import create_ensemble
import src.config_keras
from src_hpo import utils_hpo
from src.utils_metrics import get_metrics
from src.utils_visualization import plot_class_distribution, plot_precision_at_k

if 'home6' in paths.path_hpoconfigs:
    plt.switch_backend('agg')


def print_metrics(res, datasets, metrics_names, prec_at_top):
    """ Print results.

    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, dataset names
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :return:
    """

    print('#' * 100)
    print(f'Performance metrics for the ensemble ({len(models_filepaths)} models)\n')
    for dataset in datasets:
        if dataset != 'predict':
            print(dataset)
            for metric in metrics_names:
                if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                    print(f'{metric}: {res[f"{dataset}_{metric}"]}\n')

            for k in prec_at_top[dataset]:
                print(f'{"{dataset}_precision_at_{k}"}: {res[f"{dataset}_precision_at_{k}"]}\n')
    print('#' * 100)


def save_metrics_to_file(save_path, res, datasets, metrics_names, prec_at_top, print_res=False):
    """ Write results to a txt file.

    :param save_path: Path, saving directory
    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, dataset names
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :param print_res: bool, if True it prints results to std_out
    :return:
    """

    # write results to a txt file
    with open(save_path / 'results_ensemble.txt', 'w') as res_file:

        res_file.write(f'Performance metrics for the ensemble ({len(models_filepaths)} models)\n')

        for dataset in datasets:

            if dataset != 'predict':
                res_file.write(f'Dataset: {dataset}\n')
                for metric in metrics_names:
                    if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                        str_aux = f'{metric}: {res[f"{dataset}_{metric}"]}\n'
                        res_file.write(str_aux)
                        if print_res:
                            print(str_aux)

                for k in prec_at_top[dataset]:
                    str_aux = f'{f"{dataset}_precision_at_{k}"}: {res[f"{dataset}_precision_at_{k}"]}\n'
                    res_file.write(str_aux)
                    if print_res:
                        print(str_aux)

            res_file.write('\n')

        res_file.write(f'{"-" * 100}')

        res_file.write('\nModels used to create the ensemble:\n')

        for model_filepath in models_filepaths:
            res_file.write(f'{model_filepath}\n')

        res_file.write('\n')


# TODO: separate functions for each curve?
def plot_prcurve_roc(res, save_path, dataset):
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


def run_main(config, features_set, data_dir, res_dir, models_filepaths, datasets, fields, generate_csv_pred,
             scalar_params_idxs=None):
    """ Evaluate model on a given configuration in the specified datasets and also predict on them.

    :param config: configuration object from the Config class
    :param features_set: dict, each key-value pair is feature_name: {'dim': feature_dim, 'dtype': feature_dtype}
    :param data_dir: str, path to directory with the tfrecord files
    :param res_dir: str, path to directory to save results
    :param models_filepaths: list, filepaths to models to be integrated in the ensemble
    :param datasets: list, datasets in which the ensemble is evaluated/predicted on
    :param fields: list, fields to extract from the TFRecords datasets
    :param generate_csv_pred: bool, if True also generates a prediction ranking for the specified datasets
    :param scalar_params_idxs: list, indexes of scalar parameters in the TFRecords to be used as features
    :return:
    """

    verbose = False if 'home6' in paths.path_hpoconfigs else True

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in fields} for dataset in datasets}

    tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # get dataset of the TFRecord
        dataset = tfrec_file.split('-')[0]

        # if dataset == 'predict':
        #     fields_aux = list(fields)
        #     if 'label' in fields:
        #         fields_aux.remove('label')
        #     if 'original_label' in fields:
        #         fields_aux.remove('original_label')
        # else:
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
        model._name = f'model{model_i}'

        model_list.append(model)

    ensemble_model = create_ensemble(features=features_set, scalar_params_idxs=scalar_params_idxs, models=model_list)

    ensemble_model.summary()

    # set up metrics to be monitored
    metrics_list = get_metrics(clf_threshold=config['clf_thr'])

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
            continue

        print(f'Evaluating on dataset {dataset}')

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
            res[f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

    # predict on given datasets - needed for computing the output distribution and produce a ranking
    scores = {dataset: [] for dataset in datasets}
    for dataset in scores:

        print(f'Predicting on dataset {dataset}...')

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
        scores_classification[dataset][scores[dataset] >= config['clf_thr']] = 1

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in datasets}
    for dataset in output_cl:
        if dataset != 'predict':
            for original_label in config['label_map']:
                # get predictions for each original class individually to compute histogram
                output_cl[dataset][original_label] = scores[dataset][np.where(data[dataset]['original_label'] ==
                                                                              original_label)]
        else:
            output_cl[dataset]['NTP'] = scores[dataset]

    # compute precision at top-k
    labels_sorted = {}
    for dataset in datasets:
        if dataset == 'predict':
            continue
        sorted_idxs = np.argsort(scores[dataset], axis=0).squeeze()
        labels_sorted[dataset] = data[dataset]['label'][sorted_idxs].squeeze()

        for k_i in range(len(config['k_arr'][dataset])):
            if len(sorted_idxs) < config['k_arr'][dataset][k_i]:
                res[f'{dataset}_precision_at_{config["k_arr"][dataset][k_i]}'] = np.nan
            else:
                res[f'{dataset}_precision_at_{config["k_arr"][dataset][k_i]}'] = \
                    np.sum(labels_sorted[dataset][-config['k_arr'][dataset][k_i]:]) / config['k_arr'][dataset][k_i]

    # save evaluation metrics in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(res_dir / 'results_ensemble.npy', res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    for dataset in datasets:
        plot_class_distribution(output_cl[dataset],
                                res_dir / f'ensemble_class_scoredistribution_{dataset}.png')
        if dataset != 'predict':
            plot_prcurve_roc(res, res_dir, dataset)
            plot_precision_at_k(labels_sorted[dataset],
                                config['k_curve_arr'][dataset],
                                res_dir / f'{dataset}')

    print('Saving metrics to a txt file...')
    save_metrics_to_file(res_dir, res, datasets, ensemble_model.metrics_names, config['k_arr'], print_res=True)
    # print_metrics(res, datasets, ensemble_model.metrics_names, config['k_arr'])

    # generate rankings for each evaluated dataset
    if generate_csv_pred:

        print('Generating csv file(s) with ranking(s)...')

        # add predictions to the data dict
        for dataset in datasets:
            data[dataset]['score'] = scores[dataset].ravel()
            data[dataset]['predicted class'] = scores_classification[dataset].ravel()

        # write results to a txt file
        for dataset in datasets:
            print(f'Saving ranked predictions in dataset {dataset} to {res_dir / f"ranked_predictions_{dataset}"}...')
            data_df = pd.DataFrame(data[dataset])

            # sort in descending order of output
            data_df.sort_values(by='score', ascending=False, inplace=True)
            data_df.to_csv(res_dir / f'ensemble_ranked_predictions_{dataset}set.csv', index=False)

    # save model, features and config used for training this model
    ensemble_model.save(res_dir / 'ensemble_model.h5')
    np.save(res_dir / 'features_set', features_set)
    np.save(res_dir / 'config', config)
    # plot ensemble model and save the figure
    keras.utils.plot_model(ensemble_model,
                           to_file=res_dir / 'ensemble.png',
                           show_shapes=False,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=False,
                           dpi=96)


if __name__ == '__main__':

    # name of the study
    study = 'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_tces1'

    # results directory
    save_path = Path(paths.pathresultsensemble) / study
    save_path.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='predict_run')
    logger_handler = logging.FileHandler(filename=save_path / f'predict_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {study}...')

    # TFRecord files directory
    tfrec_dir = os.path.join(paths.path_tfrecs,
                             'Kepler',
                             'Q1-Q17_DR25',
                             'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period_tces1'
                             )

    logger.info(f'Using data from {tfrec_dir}')

    # datasets used; choose from 'train', 'val', 'test', 'predict' - needs to follow naming of TFRecord files
    datasets = ['train', 'val', 'test']
    # datasets = ['predict']

    logger.info(f'Datasets to be evaluated/tested: {datasets}')

    # fields to be extracted from the TFRecords and that show up in the ranking created for each dataset
    # set to None if not adding other fields
    # fields = ['target_id', 'label', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label']
    fields = ['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
              'mag', 'ra', 'dec', 'tce_max_mult_ev', 'tce_insol', 'tce_eqt', 'tce_sma', 'tce_prad', 'tce_model_snr',
              'tce_ingress', 'tce_impact', 'tce_incl', 'tce_dor', 'tce_ror']

    # print('Fields to be extracted from the TFRecords: {}'.format(fields))

    multi_class = False  # multiclass classification
    ce_weights_args = {'tfrec_dir': tfrec_dir,
                       'datasets': ['train'],
                       'label_fieldname': 'label',
                       'verbose': False
                       }
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess

    generate_csv_pred = True

    # set configuration manually. Set to None to use a configuration from a HPO study
    config = np.load(Path(paths.pathtrainedmodels) / study/ 'config.npy', allow_pickle=True).item()
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
    hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-glcentr_' \
                'std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'

    # set the configuration from a HPO study
    if hpo_study is not None:
        res = utils_hpo.logged_results_to_HBS_result(Path(paths.path_hpoconfigs) / hpo_study, f'_{hpo_study}')
        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config = id2config[config_id_hpo]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(8, 0, 3)]['config']

        logger.info(f'Using configuration from HPO study {hpo_study}')
        logger.info(f'HPO Config {config_id_hpo}: {config}')

    config.update({
        # 'num_loc_conv_blocks': 2,
        # 'num_glob_conv_blocks': 5,
        # 'init_fc_neurons': 512,
        # 'num_fc_layers': 4,
        # 'pool_size_loc': 7,
        # 'pool_size_glob': 5,
        # 'pool_stride': 2,
        # 'conv_ls_per_block': 2,
        # 'init_conv_filters': 4,
        # 'kernel_size': 5,
        # 'kernel_stride': 1,
        'non_lin_fn': 'prelu',  # 'relu',
        # 'optimizer': 'Adam',
        # 'lr': 1e-5,
        # 'batch_size': 64,
        # 'dropout_rate': 0,
    })

    # add dataset parameters
    config = src.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

    # add missing parameters in hpo with default values
    config = src.config_keras.add_default_missing_params(config=config)
    # print('Configuration used: ', config)

    logger.info(f'Final configuration used: {config}')

    scalar_params_idxs = None  # [0, 1, 2, 3, 4, 5]  # [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]

    features_set = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even flux views
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # secondary flux views
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        # secondary flux related features
        # 'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid views
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # 'global_centr_fdl_view_norm': {'dim': (2001, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (201, 1), 'dtype': tf.float32},
        # centroid related features
        # 'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        # 'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        # 'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }
    logger.info(f'Feature set: {features_set}')

    # get models for the ensemble
    models_dir = Path(paths.pathtrainedmodels) / study / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]

    run_main(config=config,
             features_set=features_set,
             scalar_params_idxs=scalar_params_idxs,
             data_dir=tfrec_dir,
             res_dir=save_path,
             models_filepaths=models_filepaths,
             datasets=datasets,
             fields=fields,
             generate_csv_pred=generate_csv_pred)

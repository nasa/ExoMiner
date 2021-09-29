"""
Perform inference using an ensemble of Tensorflow Keras models.
"""

import os
# 3rd party
import sys

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
from tensorflow.keras import callbacks

# local
import paths
from src.utils_dataio import get_data_from_tfrecord
from src.utils_dataio import InputFnv2 as InputFn
from models.models_keras import create_ensemble
import models.config_keras
from src_hpo import utils_hpo
from src.utils_metrics import get_metrics, compute_precision_at_k
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_predict import save_metrics_to_file, plot_prcurve_roc


def run_main(config, features_set, data_dir, res_dir, models_filepaths, datasets, fields, generate_csv_pred,
             callbacks_dict):
    """ Evaluate model on a given configuration in the specified datasets and also predict on them.

    :param config: configuration object from the Config class
    :param features_set: dict, each key-value pair is feature_name: {'dim': feature_dim, 'dtype': feature_dtype}
    :param data_dir: str, path to directory with the tfrecord files
    :param res_dir: str, path to directory to save results
    :param models_filepaths: list, filepaths to models to be integrated in the ensemble
    :param datasets: list, datasets in which the ensemble is evaluated/predicted on
    :param fields: list, fields to extract from the TFRecords datasets
    :param generate_csv_pred: bool, if True also generates a prediction ranking for the specified datasets
    :param callbacks_dict: dict, callbacks
    :return:
    """

    # save features and config
    np.save(res_dir / 'features_set', features_set)
    np.save(res_dir / 'config', config)

    verbose = False if 'home6' in paths.path_hpoconfigs else True

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in fields} for dataset in datasets}

    tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # get dataset of the TFRecord
        dataset = tfrec_file.split('-')[0]

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

    ensemble_model = create_ensemble(features=features_set, models=model_list)

    ensemble_model.summary()

    # save model
    ensemble_model.save(res_dir / 'ensemble_model.h5')
    # plot ensemble model and save the figure
    keras.utils.plot_model(ensemble_model,
                           to_file=res_dir / 'ensemble.png',
                           show_shapes=False,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=False,
                           dpi=96)

    # set up metrics to be monitored
    metrics_list = get_metrics(clf_threshold=config['clf_thr'], num_thresholds=config['num_thr'])

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
                                data_augmentation=False,
                                online_preproc_params=None,
                                filter_data=None,
                                )

        for callback_name in callbacks_dict:
            if 'layer' in callback_name:
                callbacks_dict[callback_name].input_fn = eval_input_fn()

        # evaluate model in the given dataset
        res_eval = ensemble_model.evaluate(x=eval_input_fn(),
                                           y=None,
                                           batch_size=None,
                                           verbose=verbose,
                                           sample_weight=None,
                                           steps=None,
                                           callbacks=list(callbacks_dict.values()) if dataset == 'train' else None,
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
                                   features_set=features_set)

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
        prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['k_arr'][dataset])
        res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
                    for k_val in config['k_arr'][dataset]})

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
    save_metrics_to_file(res_dir, res, datasets, ensemble_model.metrics_names, config['k_arr'], models_filepaths,
                         print_res=True)
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


if __name__ == '__main__':

    # name of the study
    study = 'bohb_keplerq1q17dr25_astronet_7-27-2021'

    # results directory
    paths.pathresultsensemble = '/data5/tess_project/experiments/current_experiments/results_ensemble/'
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
                             'tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps'
                             )

    logger.info(f'Using data from {tfrec_dir}')

    # datasets used; choose from 'train', 'val', 'test', 'predict' - needs to follow naming of TFRecord files
    datasets = ['train', 'val', 'test']
    # datasets = ['predict']

    logger.info(f'Datasets to be evaluated/tested: {datasets}')

    # fields to be extracted from the TFRecords and that show up in the ranking created for each dataset
    # set to None if not adding other fields
    fields = {'target_id': 'int_scalar',
              'tce_plnt_num': 'int_scalar',
              # 'oi': 'float_scalar',
              'label': 'string',
              # 'TESS Disposition': 'string',
              'tce_period': 'float_scalar',
              'tce_duration': 'float_scalar',
              'tce_time0bk': 'float_scalar',
              'original_label': 'string',
              'transit_depth': 'float_scalar',
              # 'tce_max_mult_ev': 'float_scalar',
              # 'tce_prad': 'float_scalar',
              # 'sigma_oot_odd': 'float_scalar',
              # 'sigma_it_odd': 'float_scalar',
              # 'sigma_oot_even': 'float_scalar',
              # 'sigma_it_even': 'float_scalar',
              }
    # fields = ['target_id', 'label', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label']
    # fields = ['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
    #           'mag', 'ra', 'dec', 'tce_max_mult_ev', 'tce_insol', 'tce_eqt', 'tce_sma', 'tce_prad', 'tce_model_snr',
    #           'tce_ingress', 'tce_impact', 'tce_incl', 'tce_dor', 'tce_ror']

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

    # get models for the ensemble
    models_study = study
    paths.pathtrainedmodels = '/data5/tess_project/experiments/current_experiments/trained_models/'
    models_dir = Path(paths.pathtrainedmodels) / models_study / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]
    logger.info(f'Models\' file paths: {models_filepaths}')

    # intializing configuration dictionary; can set configuration manually
    config = {}

    # name of the HPO study from which to get a configuration; config needs to be set to None
    hpo_study = 'bohb_keplerq1q17dr25_astronet_7-27-2021'

    # set the configuration from a HPO study
    if hpo_study is not None:
        paths.path_hpoconfigs = '/data5/tess_project/experiments/hpo_configs'
        hpo_study_fp = Path(paths.path_hpoconfigs) / hpo_study
        res = utils_hpo.logged_results_to_HBS_result(hpo_study_fp, f'_{hpo_study_fp.name}')
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
        # 'non_lin_fn': 'prelu',  # 'relu',
        # 'optimizer': 'Adam',
        # 'lr': 1e-5,
        # 'batch_size': 64,
        # 'dropout_rate': 0,
    })

    # add dataset parameters
    config = models.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

    # add missing parameters in hpo with default values
    config = models.config_keras.add_default_missing_params(config=config)
    # print('Configuration used: ', config)

    logger.info(f'Final configuration used: {config}')

    features_set = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (2001, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (201, 1), 'dtype': tf.float32},
        # 'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even flux features
        # 'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'sigma_oot_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_oot_even': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_even': {'dim': (1,), 'dtype': tf.float32},
        # secondary flux features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid features
        # 'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # 'global_centr_fdl_view_norm': {'dim': (2001, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (201, 1), 'dtype': tf.float32},
        # 'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'mag_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        # 'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_hap_stat_diff_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        # 'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        # 'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }
    logger.info(f'Feature set: {features_set}')

    # callbacks list
    callbacks_dict = {}

    # TensorBoard callback
    callbacks_dict['tensorboard'] = callbacks.TensorBoard(
        log_dir=save_path,
        histogram_freq=1,
        write_graph=False,
        write_images=True,
        update_freq='epoch',  # either 'epoch' or 'batch'
        profile_batch=0,
        embeddings_metadata=None,
        embeddings_freq=0
    )

    # # Custom callbacks
    # file_writer = tf.summary.create_file_writer(logdir=os.path.join(save_path, 'logs'),
    #                                             filename_suffix='input_to_fcblock')
    # callbacks_dict['layer_conbranch_concat'] = LayerOutputCallback(
    #     input_fn=None,
    #     batch_size=config['batch_size'],
    #     layer_name='convbranch_concat',
    #     summary_writer=file_writer,
    #     buckets=30,
    #     description='Output convolutional branches',
    #     ensemble=True,
    #     log_dir=os.path.join(save_path, 'logs'),
    #     num_batches=None
    # )
    # callbacks_dict['layer_stellar_dv_scalars'] = LayerOutputCallback(
    #     input_fn=None,
    #     batch_size=config['batch_size'],
    #     layer_name='stellar_dv_scalar_input',
    #     summary_writer=file_writer,
    #     buckets=30,
    #     description='Input scalars',
    #     ensemble=True,
    #     log_dir=os.path.join(save_path, 'logs'),
    #     num_batches=None
    # )

    run_main(
        config=config,
        features_set=features_set,
        data_dir=tfrec_dir,
        res_dir=save_path,
        models_filepaths=models_filepaths,
        datasets=datasets,
        fields=fields,
        generate_csv_pred=generate_csv_pred,
        callbacks_dict=callbacks_dict
    )

""" Run cross-validation experiment. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import pandas as pd
import numpy as np
import logging
# from datetime import datetime
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
from tensorflow.keras.utils import plot_model as plot_model
from tensorflow.keras.models import load_model
# import itertools
import copy
import multiprocessing
from astropy import stats
import shutil
import time
import argparse
import sys
from mpi4py import MPI
import json

# local
from src_preprocessing.tf_util import example_util
import paths
from src.utils_dataio import InputFnCV as InputFn
from src.utils_dataio import get_data_from_tfrecord
from src.models_keras import CNN1dPlanetFinderv1, Astronet, Exonet, CNN1dPlanetFinderv2
import src.config_keras
from src.utils_metrics import get_metrics
from src.models_keras import create_ensemble
import src.utils_predict as utils_predict
import src.utils_train as utils_train
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src_hpo import utils_hpo
from src_preprocessing.preprocess import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc, \
    centering_and_normalization
from utils.utils_dataio import is_jsonable


def train_model(model_id, base_model, n_epochs, config, features_set, data_fps, res_dir, online_preproc_params,
                data_augmentation, callbacks_dict, opt_metric, min_optmetric, logger, verbose=False):

    datasets = list(data_fps.keys())

    model_dir = res_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    model_dir_sub = model_dir / f'model{model_id}'
    model_dir_sub.mkdir(exist_ok=True)

    if 'tensorboard' in callbacks_dict:
        callbacks_dict['tensorboard'].log_dir = model_dir_sub

    # instantiate Keras model
    model = base_model(config, features_set).kerasModel

    # print model summary
    if model_id == 0:
        model.summary()

    # setup metrics to be monitored
    metrics_list = get_metrics(clf_threshold=config['clf_thr'], num_thresholds=config['num_thr'])

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

    # input function for training, validation and test
    train_input_fn = InputFn(filepaths=data_fps['train'],
                             batch_size=config['batch_size'],
                             mode='TRAIN',
                             label_map=config['label_map'],
                             data_augmentation=data_augmentation,
                             online_preproc_params=online_preproc_params,
                             features_set=features_set)
    val_input_fn = InputFn(filepaths=data_fps['val'],
                           batch_size=config['batch_size'],
                           mode='EVAL',
                           label_map=config['label_map'],
                           features_set=features_set)

    # fit the model to the training data
    logger.info(f'[model_{model_id}] Training model...')
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=n_epochs,
                        verbose=verbose,
                        callbacks=list(callbacks_dict.values()),
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

    # save results in a numpy file
    res_fp = model_dir_sub / 'results.npy'
    print(f'[model_{model_id}] Saving metrics to {res_fp}...')
    np.save(res_fp, res)

    print(f'[model_{model_id}] Plotting evaluation results...')
    epochs = np.arange(1, len(res['loss']) + 1)
    # choose epoch associated with the best value for the metric
    if 'early_stopping' in callbacks_dict:
        if min_optmetric:
            ep_idx = np.argmin(res[f'val_{opt_metric}'])
        else:
            ep_idx = np.argmax(res[f'val_{opt_metric}'])
    else:
        ep_idx = -1
    # plot evaluation loss and metric curves
    utils_train.plot_loss_metric(res,
                                 epochs,
                                 ep_idx,
                                 opt_metric,
                                 res_dir / f'model{model_id}_plotseval_epochs{epochs[-1]:.0f}.svg')

    print(f'[model_{model_id}] Saving metrics to a txt file...')
    utils_train.save_metrics_to_file(model_dir_sub, res, datasets, ep_idx, model.metrics_names)

    # save model, features and config used for training this model
    model.save(model_dir_sub / f'model{model_id}.h5')
    if model_id == 1:
        # # save feature set used
        # np.save(model_dir / 'features_set', features_set)
        # # save configuration used
        # np.save(model_dir / 'config', config)
        # save plot of model
        plot_model(model,
                   to_file=res_dir / 'model.svg',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=48)


def predict_ensemble(models_filepaths, config, features_set, data_fps, data_fields, generate_csv_pred, res_dir,
                     logger, verbose=False):

    datasets = list(data_fps.keys())

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in data_fields} for dataset in datasets}
    for dataset in datasets:
        for data_fp in data_fps[dataset]:
            data_aux = get_data_from_tfrecord(str(data_fp), data_fields, config['label_map'])

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

        logger.info(f'[ensemble] Evaluating on dataset {dataset}')
        # input function for evaluating on each dataset
        eval_input_fn = InputFn(filepaths=data_fps[dataset],
                                batch_size=config['batch_size'],
                                mode="EVAL",
                                label_map=config['label_map'],
                                features_set=features_set)

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
        print(f'[ensemble] Predicting on dataset {dataset}...')

        predict_input_fn = InputFn(filepaths=data_fps[dataset],
                                   batch_size=config['batch_size'],
                                   mode='PREDICT',
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
    output_cl = {dataset: {} for dataset in data_fps}
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
    print('[ensemble] Saving metrics to a numpy file...')
    np.save(res_dir / 'results_ensemble.npy', res)

    print('[ensemble] Plotting evaluation results...')
    # draw evaluation plots
    for dataset in datasets:
        plot_class_distribution(output_cl[dataset],
                                res_dir / f'ensemble_class_scoredistribution_{dataset}.png')
        if dataset != 'predict':
            utils_predict.plot_prcurve_roc(res, res_dir, dataset)
            plot_precision_at_k(labels_sorted[dataset],
                                config['k_curve_arr'][dataset],
                                res_dir / f'{dataset}')

    print('[ensemble] Saving metrics to a txt file...')
    utils_predict.save_metrics_to_file(res_dir, res, datasets, ensemble_model.metrics_names, config['k_arr'],
                                       models_filepaths, print_res=True)

    # generate rankings for each evaluated dataset
    if generate_csv_pred:

        print('[ensemble] Generating csv file(s) with ranking(s)...')

        # add predictions to the data dict
        for dataset in data_fps:
            data[dataset]['score'] = scores[dataset].ravel()
            data[dataset]['predicted class'] = scores_classification[dataset].ravel()

        # write results to a txt file
        for dataset in data_fps:
            print(f'[ensemble] Saving ranked predictions in dataset {dataset} to '
                  f'{res_dir / f"ranked_predictions_{dataset}"}...')
            data_df = pd.DataFrame(data[dataset])

            # sort in descending order of output
            data_df.sort_values(by='score', ascending=False, inplace=True)
            data_df.to_csv(res_dir / f'ensemble_ranked_predictions_{dataset}set.csv', index=False)

    # save model, features and config used for training this model
    ensemble_model.save(res_dir / 'ensemble_model.h5')
    # np.save(res_dir / 'features_set', features_set)
    # np.save(res_dir / 'config', config)
    # plot ensemble model and save the figure
    plot_model(ensemble_model,
               to_file=res_dir / 'ensemble.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96)


def compute_normalization_stats(scalar_params, train_data_fps, aux_params, norm_dir, verbose=None):

    scalar_params_data = {scalar_param: [] for scalar_param in scalar_params}

    # FDL centroid time series normalization statistics parameters
    ts_centr_fdl = ['global_centr_fdl_view', 'local_centr_fdl_view']
    ts_centr_fdl_data = {ts: [] for ts in ts_centr_fdl}

    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(aux_params['num_bins_loc'],
                                                              aux_params['nr_transit_durations'])  # same for all TCEs

    # our centroid time series normalization statistics parameters
    ts_centr = ['global_centr_view', 'local_centr_view']
    ts_centr_data = {ts: [] for ts in ts_centr}

    # get data out of the training set TFRecords
    for train_data_i, train_data_fp in enumerate(train_data_fps):

        if verbose:
            print(f'Getting data from {train_data_fp.name} ({train_data_i / len(train_data_fps) * 100} %)')

        # iterate through the shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(train_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # get scalar parameters data
            for scalar_param in scalar_params:
                if scalar_params[scalar_param]['dtype'] == 'int':
                    scalar_params_data[scalar_param].append(example.features.feature[scalar_param].int64_list.value[0])
                elif scalar_params[scalar_param]['dtype'] == 'float':
                    scalar_params_data[scalar_param].append(example.features.feature[scalar_param].float_list.value[0])

            # get FDL centroid time series data
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(aux_params['num_bins_glob'],
                                                                        transitDuration,
                                                                        orbitalPeriod)
            for ts in ts_centr_fdl:
                ts_tce = np.array(example.features.feature[ts].float_list.value)
                if 'glob' in ts:
                    ts_centr_fdl_data[ts].extend(ts_tce[idxs_nontransitcadences_glob])
                else:
                    ts_centr_fdl_data[ts].extend(ts_tce[idxs_nontransitcadences_loc])

            # get centroid time series data
            for ts in ts_centr:
                ts_tce = np.array(example.features.feature[ts].float_list.value)
                if 'glob' in ts:
                    ts_centr_data[ts].extend(ts_tce[idxs_nontransitcadences_glob])
                else:
                    ts_centr_data[ts].extend(ts_tce[idxs_nontransitcadences_loc])

    # save normalization statistics for the scalar parameters (median and robust estimator of std)
    # do not use missing values to compute the normalization statistics for bootstrap FA probability
    scalar_params_data = {scalar_param: np.array(scalar_params_vals) for scalar_param, scalar_params_vals in
                        scalar_params_data.items()}
    scalar_norm_stats = {scalar_param: {'median': np.nan, 'mad_std': np.nan, 'info': scalar_params[scalar_param]}
                       for scalar_param in scalar_params}
    for scalar_param in scalar_params:

        scalar_param_vals = scalar_params_data[scalar_param]

        # remove missing values so that they do not contribute to the normalization statistics
        if not np.isnan(scalar_params[scalar_param]['missing_value']):
            if scalar_param == 'wst_depth':
                scalar_param_vals = scalar_param_vals[
                    np.where(scalar_param_vals > scalar_params[scalar_param]['missing_value'])]
            else:
                scalar_param_vals = scalar_param_vals[
                    np.where(scalar_param_vals != scalar_params[scalar_param]['missing_value'])]

        # log transform the data
        if scalar_params[scalar_param]['log_transform']:

            # add constant value
            if not np.isnan(scalar_params[scalar_param]['log_transform_eps']):
                scalar_param_vals += scalar_params[scalar_param]['log_transform_eps']

            scalar_param_vals = np.log10(scalar_param_vals)

        # compute median as robust estimate of central tendency
        scalar_norm_stats[scalar_param]['median'] = np.median(scalar_param_vals)
        # compute MAD std as robust estimate of deviation from central tendency
        scalar_norm_stats[scalar_param]['mad_std'] = stats.mad_std(scalar_param_vals)

    # save normalization statistics into a numpy file
    np.save(norm_dir / 'train_scalarparam_norm_stats.npy', scalar_norm_stats)

    # create additional csv file with normalization statistics
    scalar_norm_stats_fordf = {}
    for scalar_param in scalar_params:
        scalar_norm_stats_fordf[f'{scalar_param}_median'] = [scalar_norm_stats[scalar_param]['median']]
        scalar_norm_stats_fordf[f'{scalar_param}_mad_std'] = [scalar_norm_stats[scalar_param]['mad_std']]
    scalar_norm_stats_df = pd.DataFrame(data=scalar_norm_stats_fordf)
    scalar_norm_stats_df.to_csv(norm_dir / 'train_scalarparam_norm_stats.csv', index=False)

    # save normalization statistics for FDL centroid time series
    centr_fdl_norm_stats = {ts: {
        'oot_median': np.median(ts_centr_fdl_data[ts]),
        'oot_std': stats.mad_std(ts_centr_fdl_data[ts])
    }
        for ts in ts_centr_fdl}
    np.save(norm_dir / 'train_fdlcentroid_norm_stats.npy', centr_fdl_norm_stats)

    # create additional csv file with normalization statistics
    centr_fdl_norm_stats_fordf = {}
    for ts in ts_centr_fdl:
        centr_fdl_norm_stats_fordf[f'{ts}_oot_median'] = [centr_fdl_norm_stats[ts]['oot_median']]
        centr_fdl_norm_stats_fordf[f'{ts}_oot_std'] = [centr_fdl_norm_stats[ts]['oot_std']]
    centr_fdl_norm_stats_df = pd.DataFrame(data=centr_fdl_norm_stats_fordf)
    centr_fdl_norm_stats_df.to_csv(norm_dir / 'train_fdlcentroid_norm_stats.csv', index=False)

    # save normalization statistics for centroid time series
    centr_norm_stats = {ts: {
        'median': np.median(ts_centr_data[ts]),
        'std': stats.mad_std(ts_centr_data[ts]),
        'clip_value': 30
        # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
        #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
    }
        for ts in ts_centr}
    for ts in ts_centr:
        ts_centr_clipped = np.clip(ts_centr_data[ts], a_max=30, a_min=None)
        clipStats = {
            'median_clip': np.median(ts_centr_clipped),
            'std_clip': stats.mad_std(ts_centr_clipped)
        }
        centr_norm_stats[ts].update(clipStats)
    np.save(norm_dir / 'train_centroid_norm_stats.npy', centr_norm_stats)

    # create additional csv file with normalization statistics
    centr_norm_stats_fordf = {}
    for ts in ts_centr:
        centr_norm_stats_fordf[f'{ts}_median'] = [centr_norm_stats[ts]['median']]
        centr_norm_stats_fordf[f'{ts}_std'] = [centr_norm_stats[ts]['std']]
        centr_norm_stats_fordf[f'{ts}_clip_value'] = [centr_norm_stats[ts]['clip_value']]
        centr_norm_stats_fordf[f'{ts}_median_clip'] = [centr_norm_stats[ts]['median_clip']]
        centr_norm_stats_fordf[f'{ts}_std_clip'] = [centr_norm_stats[ts]['std_clip']]
    centr_norm_stats_df = pd.DataFrame(data=centr_norm_stats_fordf)
    centr_norm_stats_df.to_csv(norm_dir / 'train_centroid_norm_stats.csv', index=False)


def normalize_data(src_data_fp, norm_stats, aux_params, norm_data_dir):

    # get out-of-transit indices for the local views
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(aux_params['num_bins_loc'],
                                                              aux_params['nr_transit_durations'])  # same for all TCEs

    with tf.io.TFRecordWriter(str(norm_data_dir / src_data_fp.name)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # normalize scalar parameters
            norm_features = {}
            for scalar_param in norm_stats['scalar_params']:
                # get the scalar value from the example
                if norm_stats['scalar_params'][scalar_param]['info']['dtype'] == 'int':
                    scalar_param_val = np.array(example.features.feature[scalar_param].int64_list.value)
                elif norm_stats['scalar_params'][scalar_param]['info']['dtype'] == 'float':
                    scalar_param_val = np.array(example.features.feature[scalar_param].float_list.value)

                # remove missing values so that they do not contribute to the normalization statistics
                # for wst_depth, replace values smaller than missing_value by the replace value
                if scalar_param_val < norm_stats['scalar_params'][scalar_param]['info']['missing_value'] and \
                        scalar_param == 'wst_depth':
                    scalar_param_val = norm_stats['scalar_params'][scalar_param]['info']['replace_value']
                # replace missing value by the median
                elif scalar_param_val == norm_stats['scalar_params'][scalar_param]['info']['missing_value']:
                    scalar_param_val = norm_stats['scalar_params'][scalar_param]['median']

                else:  # in the case that the value is not a missing value
                    # log transform the data
                    if norm_stats['scalar_params'][scalar_param]['info']['log_transform']:
                        # add constant value
                        if not np.isnan(norm_stats['scalar_params'][scalar_param]['info']['log_transform_eps']):
                            scalar_param_val += norm_stats['scalar_params'][scalar_param]['info']['log_transform_eps']

                        scalar_param_val = np.log10(scalar_param_val)

                    # clipping the data
                    # if not np.isnan(normStats['scalar_params']['info'][scalarParam]['clip_factor']):
                    #     scalarParamVal = np.clip([scalarParamVal],
                    #                              normStats['scalar_params']['median'][scalarParam] -
                    #                              normStats['scalar_params']['info'][scalarParam]['clip_factor'] *
                    #                              normStats['scalar_params']['mad_std'][scalarParam],
                    #                              normStats['scalar_params']['median'][scalarParam] +
                    #                              normStats['scalar_params']['info'][scalarParam]['clip_factor'] *
                    #                              normStats['scalar_params']['mad_std'][scalarParam]
                    #                              )[0]
                    if not np.isnan(norm_stats['scalar_params'][scalar_param]['info']['clip_factor']):
                        scalar_param_val = np.clip([scalar_param_val],
                                                 norm_stats['scalar_params'][scalar_param]['median'] -
                                                 norm_stats['scalar_params'][scalar_param]['info']['clip_factor'] *
                                                 norm_stats['scalar_params'][scalar_param]['mad_std'],
                                                 norm_stats['scalar_params'][scalar_param]['median'] +
                                                 norm_stats['scalar_params'][scalar_param]['info']['clip_factor'] *
                                                 norm_stats['scalar_params'][scalar_param]['mad_std']
                                                 )[0]

                # standardization
                scalar_param_val = (scalar_param_val - norm_stats['scalar_params'][scalar_param]['median']) / \
                                 norm_stats['scalar_params'][scalar_param]['mad_std']

                norm_features[f'{scalar_param}_norm'] = [scalar_param_val]

            # normalize FDL centroid time series
            # get out-of-transit indices for the global views
            transit_duration = example.features.feature['tce_duration'].float_list.value[0]
            orbital_period = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(aux_params['num_bins_glob'],
                                                                        transit_duration,
                                                                        orbital_period)
            # compute oot global and local flux views std
            glob_flux_view_std = \
                np.std(
                    np.array(example.features.feature['global_'
                                                      'flux_view_'
                                                      'fluxnorm'].float_list.value)[idxs_nontransitcadences_glob],
                    ddof=1)
            loc_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['local_'
                                                 'flux_view_'
                                                 'fluxnorm'].float_list.value)[idxs_nontransitcadences_loc], ddof=1)
            # center and normalize FDL centroid time series
            glob_centr_fdl_view = np.array(example.features.feature['global_centr_fdl_view'].float_list.value)
            glob_centr_fdl_view_norm = \
                centering_and_normalization(glob_centr_fdl_view,
                                            norm_stats['fdl_centroid']['global_centr_fdl_view']['oot_median'],
                                            norm_stats['fdl_centroid']['global_centr_fdl_view']['oot_std']
                                            )
            glob_centr_fdl_view_norm *= glob_flux_view_std / \
                                        np.std(glob_centr_fdl_view_norm[idxs_nontransitcadences_glob], ddof=1)
            loc_centr_fdl_view = np.array(example.features.feature['local_centr_fdl_view'].float_list.value)
            loc_centr_fdl_view_norm = \
                centering_and_normalization(loc_centr_fdl_view,
                                            norm_stats['fdl_centroid']['local_centr_fdl_view']['oot_median'],
                                            norm_stats['fdl_centroid']['local_centr_fdl_view']['oot_std']
                                            )
            loc_centr_fdl_view_norm *= loc_flux_view_std / np.std(loc_centr_fdl_view_norm[idxs_nontransitcadences_loc],
                                                                  ddof=1)

            # normalize centroid time series
            glob_centr_view = np.array(example.features.feature['global_centr_view'].float_list.value)
            loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)

            # 1) clipping to physically meaningful distance in arcsec
            glob_centr_view_std_clip = np.clip(glob_centr_view,
                                               a_max=norm_stats['centroid']['global_centr_view']['clip_value'],
                                               a_min=None)
            glob_centr_view_std_clip = centering_and_normalization(glob_centr_view_std_clip,
                                                                   # normStats['centroid']['global_centr_view']['median_'
                                                                   #                                            'clip'],
                                                                   np.median(glob_centr_view_std_clip),
                                                                   norm_stats['centroid']['global_centr_view']['std_'
                                                                                                              'clip'])

            loc_centr_view_std_clip = np.clip(loc_centr_view, a_max=norm_stats['centroid']['local_centr_view']['clip_'
                                                                                                              'value'],
                                              a_min=None)
            loc_centr_view_std_clip = centering_and_normalization(loc_centr_view_std_clip,
                                                                  # normStats['centroid']['local_centr_view']['median_'
                                                                  #                                           'clip'],
                                                                  np.median(loc_centr_view_std_clip),
                                                                  norm_stats['centroid']['local_centr_view']['std_'
                                                                                                            'clip'])

            # 2) no clipping
            glob_centr_view_std_noclip = centering_and_normalization(glob_centr_view,
                                                                     # normStats['centroid']['global_centr_'
                                                                     #                       'view']['median'],
                                                                     np.median(glob_centr_view),
                                                                     norm_stats['centroid']['global_centr_view']['std'])
            loc_centr_view_std_noclip = centering_and_normalization(loc_centr_view,
                                                                    # normStats['centroid']['local_centr_view']['median'],
                                                                    np.median(loc_centr_view),
                                                                    norm_stats['centroid']['local_centr_view']['std'])

            # 3) center each centroid individually using their median and divide by the standard deviation of the
            # training set
            glob_centr_view_medind_std = glob_centr_view - np.median(glob_centr_view)
            glob_centr_view_medind_std /= norm_stats['centroid']['global_centr_view']['std']
            loc_centr_view_medind_std = loc_centr_view - np.median(loc_centr_view)
            loc_centr_view_medind_std /= norm_stats['centroid']['local_centr_view']['std']

            # add features to the example in the TFRecord
            norm_features.update({
                'local_centr_fdl_view_norm': loc_centr_fdl_view_norm,
                'global_centr_fdl_view_norm': glob_centr_fdl_view_norm,
                'global_centr_view_std_clip': glob_centr_view_std_clip,
                'local_centr_view_std_clip': loc_centr_view_std_clip,
                'global_centr_view_std_noclip': glob_centr_view_std_noclip,
                'local_centr_view_std_noclip': loc_centr_view_std_noclip,
                'global_centr_view_medind_std': glob_centr_view_medind_std,
                'local_centr_view_medind_std': loc_centr_view_medind_std
            })

            for norm_feature_name, norm_feature in norm_features.items():
                example_util.set_float_feature(example, norm_feature_name, norm_feature, allow_overwrite=True)

            writer.write(example.SerializeToString())


def processing_data_run(data_shards_fps, run_params, cv_run_dir, logger=None):

    norm_stats_dir = cv_run_dir / 'norm_stats'
    norm_stats_dir.mkdir(exist_ok=True)

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Computing normalization statistics')
    # with tf.device('/gpu:0'):
    #     compute_normalization_stats(run_params['scalar_params'], data_shards_fps['train'], run_params['norm'],
    #                                 norm_stats_dir, False)
    # with tf.device('/device:CPU:0'):
    p = multiprocessing.Process(target=compute_normalization_stats,
                                args=(run_params['scalar_params'],
                                      data_shards_fps['train'],
                                      run_params['norm'],
                                      norm_stats_dir,
                                      False))
    p.start()
    p.join()

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data')
    norm_data_dir = cv_run_dir / 'norm_data'
    norm_data_dir.mkdir(exist_ok=True)

    # load normalization statistics
    norm_stats = {
        'scalar_params': np.load(norm_stats_dir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item(),
        'fdl_centroid': np.load(norm_stats_dir / 'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item(),
        'centroid': np.load(norm_stats_dir / 'train_centroid_norm_stats.npy', allow_pickle=True).item()
    }

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    jobs = [(file, norm_stats, run_params['norm'], norm_data_dir)
            for file in np.concatenate(list(data_shards_fps.values()))]
    async_results = [pool.apply_async(normalize_data, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    data_shards_fps_norm = {dataset: [norm_data_dir / data_fp.name for data_fp in data_fps]
                            for dataset, data_fps in data_shards_fps.items()}

    return data_shards_fps_norm


def cv_run(cv_dir, data_shards_fps, run_params, logger=None):

    cv_run_dir = cv_dir / f'cv_iter_{run_params["cv_id"]}'
    cv_run_dir.mkdir(exist_ok=True)

    # split training folds into training and validation sets by randomly selecting one of the folds as the validation
    # set
    data_shards_fps_eval = copy.deepcopy(data_shards_fps)
    data_shards_fps_eval['val'] = run_params['rng'].choice(data_shards_fps['train'], 1, replace=False)
    data_shards_fps_eval['train'] = np.setdiff1d(data_shards_fps['train'], data_shards_fps_eval['val'])

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Split for CV iteration: {data_shards_fps_eval}')

    # np.save(cv_run_dir / 'fold_split.npy', data_shards_fps_eval)
    with open(cv_run_dir / 'fold_split.json', 'w') as cv_run_file:
        json.dump({dataset: [str(fp) for fp in data_shards_fps_eval[dataset]] for dataset in data_shards_fps_eval},
                  cv_run_file)

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Normalizing data for CV iteration')
    data_shards_fps_eval = processing_data_run(data_shards_fps_eval, run_params, cv_run_dir, logger)

    # TODO: implement parallel training?
    # pool = multiprocessing.Pool(processes=run_params['n_processes_train'])
    # jobs = [(
    #     model_id,
    #     run_params['base_model'],
    #     run_params['n_epochs'],
    #     run_params['config'],
    #     run_params['features_set'],
    #     data_shards_fps_eval,
    #     cv_run_dir,
    #     run_params['online_preproc_params'],
    #     run_params['data_augmentation'],
    #     run_params['callbacks_dict'],
    #     run_params['opt_metric'],
    #     run_params['min_optmetric'],
    #     run_params['verbose']
    #          )
    #         for model_id in range(run_params['n_models'])]
    # async_results = [pool.apply_async(train_model, job) for job in jobs]
    # pool.close()
    # for async_result in async_results:
    #     async_result.get()

    # sequential training
    for model_id in range(run_params['n_models']): # train 10 models
        logger.info(f'[cv_iter_{run_params["cv_id"]}] Training model {model_id + 1} out of {run_params["n_models"]} on '
                    f'{run_params["n_epochs"]} epochs...')

        # p = multiprocessing.Process(target=train_model,
        #                             args=(model_id,
        #                                   run_params['base_model'],
        #                                   run_params['n_epochs'],
        #                                   run_params['config'],
        #                                   run_params['features_set'],
        #                                   {dataset: fps for dataset, fps in data_shards_fps_eval.items() if
        #                                    dataset != 'test'},
        #                                   cv_run_dir,
        #                                   run_params['online_preproc_params'],
        #                                   run_params['data_augmentation'],
        #                                   run_params['callbacks_dict'],
        #                                   run_params['opt_metric'],
        #                                   run_params['min_optmetric'],
        #                                   logger,
        #                                   run_params['verbose'],
        #                                   ))
        # p.start()
        # p.join()

        with tf.device(run_params['dev_train']):
            train_model(model_id,
                        run_params['base_model'],
                        run_params['n_epochs'],
                        run_params['config'],
                        run_params['features_set'],
                        {dataset: fps for dataset, fps in data_shards_fps_eval.items() if
                         dataset != 'test'},
                        cv_run_dir,
                        run_params['online_preproc_params'],
                        run_params['data_augmentation'],
                        run_params['callbacks_dict'],
                        run_params['opt_metric'],
                        run_params['min_optmetric'],
                        logger,
                        run_params['verbose'],
                        )

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Evaluating ensemble')
    # get the filepaths for the trained models
    models_dir = cv_run_dir / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]

    # evaluate using the ensemble of trained models
    # p = multiprocessing.Process(target=predict_ensemble,
    #                             args=(
    #                                 models_filepaths,
    #                                 run_params['config'],
    #                                 run_params['features_set'],
    #                                 data_shards_fps_eval,
    #                                 run_params['data_fields'],
    #                                 run_params['generate_csv_pred'],
    #                                 cv_run_dir,
    #                                 logger,
    #                                 run_params['verbose'],
    #                             ))
    # p.start()
    # p.join()
    with tf.device(run_params['dev_predict']):
        predict_ensemble(models_filepaths,
                         run_params['config'],
                         run_params['features_set'],
                         data_shards_fps_eval,
                         run_params['data_fields'],
                         run_params['generate_csv_pred'],
                         cv_run_dir,
                         logger,
                         run_params['verbose'],
                         )

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # remove preprocessed data for this run
    shutil.rmtree(cv_run_dir / 'norm_data')
    # TODO: delete the models as well?


def cv():

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    ngpus_per_node = 1  # number of GPUs per node

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    print(f'Rank = {rank}/{size - 1}')

    rank = ngpus_per_node * args.job_idx + rank
    print(f'Rank* = {rank}')
    sys.stdout.flush()
    if rank != 0:
        time.sleep(2)

    # get list of physical GPU devices available to TF in this process
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(f'[rank_{rank}] List of physical GPU devices available: {physical_devices}')

    # select GPU to be used
    # gpu_id = rank % ngpus_per_node
    # tf.config.set_visible_devices(physical_devices[:0], 'GPU')

    # get list of logical GPU devices available to TF in this process
    # logical_devices = tf.config.list_logical_devices('GPU')
    # print(f'List of logical GPU devices available: {logical_devices}')

    tf.debugging.set_log_device_placement(True)

    run_params = {}

    # name of the experiment
    run_params['cv_experiment'] = 'test_cv_kepler'

    # cv root directory
    run_params['cv_dir'] = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33')

    # data directory
    run_params['data_dir'] = run_params['cv_dir'] / 'tfrecords'

    # cv experiment directory
    run_params['cv_experiment_dir'] = Path(paths.path_cv_experiments) / run_params['cv_experiment']
    run_params['cv_experiment_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    run_params['data_shards_fns'] = np.load(run_params['cv_dir'] / 'cv_folds_runs.npy', allow_pickle=True)
    run_params['data_shards_fps'] = [{dataset: [run_params['data_dir'] / fold for fold in cv_iter[dataset]]
                                      for dataset in cv_iter} for cv_iter in run_params['data_shards_fns']]

    if rank >= len(run_params['data_shards_fps']):
        return

    # set up logger
    logger = logging.getLogger(name=f'cv_run_rank_{rank}')
    logger_handler = logging.FileHandler(filename=run_params['cv_experiment_dir'] / f'cv_run_{rank}.log', mode='w')
    # logger_handler_stream = logging.StreamHandler(sys.stdout)
    # logger_handler_stream.setLevel(logging.INFO)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    # logger_handler_stream.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    # logger.addHandler(logger_handler_stream)
    logger.info(f'Starting run {run_params["cv_experiment"]}...')

    run_params['ngpus_per_node'] = ngpus_per_node
    run_params['gpu_id'] = rank % run_params['ngpus_per_node']
    run_params['dev_train'] = f'/gpu:{run_params["gpu_id"]}'
    run_params['dev_predict'] = f'/gpu:{run_params["gpu_id"]}'

    run_params['n_processes_norm_data'] = 15
    run_params['verbose'] = True

    run_params['rnd_seed'] = 2
    run_params['rng'] = np.random.default_rng(seed=run_params['rnd_seed'])

    # normalization information for the scalar parameters
    run_params['scalar_params'] = {
        # stellar parameters
        'tce_steff': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': np.nan, 'dtype': 'int'},
        'tce_slogg': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': np.nan, 'dtype': 'float'},
        'tce_smet': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': np.nan, 'dtype': 'float'},
        'tce_sradius': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                        'clip_factor': np.nan, 'dtype': 'float'},
        'tce_smass': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': np.nan, 'dtype': 'float'},
        'tce_sdens': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': np.nan, 'dtype': 'float'},
        # secondary
        'tce_maxmes': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': np.nan, 'dtype': 'float'},
        'wst_depth': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': 0},
        # 'tce_albedo': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #                'clip_factor': np.nan, 'dtype': 'float'},
        'tce_albedo_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                            'dtype': 'float'},
        # 'tce_ptemp': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #               'clip_factor': np.nan, 'dtype': 'float'},
        'tce_ptemp_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                           'dtype': 'float'},
        # 'wst_robstat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #                 'clip_factor': np.nan, 'dtype': 'float'},
        # odd-even
        # 'tce_bin_oedp_stat': {'missing_value': np.nan, 'log_transform': True, 'log_transform_eps': 1e-32,
        #                       'clip_factor': np.nan, 'dtype': 'float'},
        # other diagnostics
        'boot_fap': {'missing_value': -1, 'log_transform': True, 'log_transform_eps': 1e-32,
                     'clip_factor': np.nan, 'dtype': 'float'},
        'tce_cap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float'},
        'tce_hap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float'},
        'tce_rb_tcount0': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'int'},
        # centroid
        'tce_fwm_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                         'dtype': 'float'},
        'tce_dikco_msky': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'float'},
        'tce_dicco_msky': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'float'},
        'tce_dikco_msky_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': np.nan, 'dtype': 'float'},
        'tce_dicco_msky_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': np.nan, 'dtype': 'float'},
        # flux
        'transit_depth': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                          'clip_factor': 20, 'dtype': 'float'},
        # 'tce_depth_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #                   'clip_factor': np.nan, 'dtype': 'float'},
        'tce_duration': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': np.nan, 'dtype': 'float'},
        # 'tce_duration_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #                      'clip_factor': np.nan, 'dtype': 'float'},
        'tce_period': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': np.nan, 'dtype': 'float'},
        # 'tce_period_err': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
        #                    'clip_factor': np.nan, 'dtype': 'float'},
        'tce_max_mult_ev': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                            'clip_factor': np.nan, 'dtype': 'float'},
        'tce_prad': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float'}
    }

    run_params['norm'] = {
        'nr_transit_durations': 6,
        # 2 * 2.5 + 1,  # 2 * 4 + 1,  # number of transit durations (2*n+1, n on each side of the transit)
        'num_bins_loc': 31,  # 31, 201
        'num_bins_glob': 301  # 301, 2001
    }

    # set configuration manually. Set to None to use a configuration from an HPO study
    config = {}

    # name of the HPO study from which to get a configuration
    hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-glcentr_' \
                'std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'
    # set the configuration from an HPO study
    if hpo_study is not None:
        hpo_path = Path(paths.path_hpoconfigs) / hpo_study
        res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_study}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config = id2config[config_id_hpo]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        logger.info(f'Using configuration from HPO study {hpo_study}')
        logger.info(f'HPO Config {config_id_hpo}: {config}')

    # base model used - check estimator_util.py to see which models are implemented
    run_params['base_model'] = CNN1dPlanetFinderv2
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
    # select convolutional branches
    config['branches'] = [
        'global_flux_view_fluxnorm',
        'local_flux_view_fluxnorm',
        # 'global_centr_fdl_view_norm',
        # 'local_centr_fdl_view_norm',
        'local_flux_oddeven_views',
        'global_centr_view_std_noclip',
        'local_centr_view_std_noclip',
        # 'local_weak_secondary_view_fluxnorm',
        # 'local_weak_secondary_view_selfnorm',
        'local_weak_secondary_view_max_flux-wks_norm'
    ]

    # choose features set
    run_params['features_set'] = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even views
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # centroid views
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # secondary related features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid related features
        'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }

    logger.info(f'Feature set: {run_params["features_set"]}')

    run_params['data_augmentation'] = False  # if True, uses online data augmentation in the training set
    run_params['online_preproc_params'] = {'num_bins_global': 301, 'num_bins_local': 31, 'num_transit_dur': 6}

    run_params['n_models'] = 2  # number of models in the ensemble
    run_params['n_epochs'] = 5  # number of epochs used to train each model
    run_params['multi_class'] = False  # multi-class classification
    run_params['ce_weights_args'] = {
        'tfrec_dir': run_params['cv_experiment_dir'] / 'tfrecords',
        'datasets': ['train'],
        'label_fieldname': 'label',
        'verbose': False
    }
    run_params['use_kepler_ce'] = False  # use weighted CE loss based on the class proportions in the training set
    run_params['satellite'] = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    run_params['opt_metric'] = 'auc_pr'  # choose which metric to plot side by side with the loss
    run_params['min_optmetric'] = False  # if lower value is better set to True

    # callbacks list
    run_params['callbacks_dict'] = {}

    # early stopping callback
    run_params['callbacks_dict']['early_stopping'] = callbacks.EarlyStopping(
        monitor=f'val_{run_params["opt_metric"]}',
        min_delta=0,
        patience=20,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )

    # # TensorBoard callback
    # run_params['callbacks_dict']['tensorboard'] = callbacks.TensorBoard(
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=False,
    #     update_freq='epoch',  # either 'epoch' or 'batch'
    #     profile_batch=2,
    #     embeddings_metadata=None,
    #     embeddings_freq=0
    # )

    # add dataset parameters
    config = src.config_keras.add_dataset_params(
        run_params['satellite'],
        run_params['multi_class'],
        run_params['use_kepler_ce'],
        run_params['ce_weights_args'],
        config
    )

    # add missing parameters in hpo with default values
    config = src.config_keras.add_default_missing_params(config=config)

    logger.info(f'Final configuration used: {config}')
    run_params['config'] = config

    run_params['data_fields'] = ['target_id', 'label', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk',
                                 'original_label']
    run_params['generate_csv_pred'] = True

    # save feature set used
    if rank == 0:
        np.save(run_params['cv_experiment_dir'] / 'features_set.npy', run_params['features_set'])
        # save configuration used
        np.save(run_params['cv_experiment_dir'] / 'config.npy', run_params['config'])

        # save the JSON file with training-evaluation parameters that are JSON serializable
        json_dict = {key: val for key, val in run_params.items() if is_jsonable(val)}
        with open(run_params['cv_experiment_dir'] / 'cv_params.json', 'w') as cv_run_file:
            json.dump(json_dict, cv_run_file)

    # # run each CV iteration sequentially
    # for cv_id, cv_iter in enumerate(run_params['data_shards_fps']):
    #     logger.info(f'Running CV iteration {cv_id} (out of {len(run_params["data_shards_fps"])}): '
    #                 f'{cv_iter}')
    #     run_params['cv_id'] = cv_id
    #     cv_run(
    #            run_params['cv_experiment_dir'],
    #            cv_iter,
    #            run_params,
    #            logger
    #            )

    cv_id = rank
    logger.info(f'Running CV iteration {cv_id} (out of {len(run_params["data_shards_fps"])}): '
                f'{run_params["data_shards_fps"][cv_id]}')
    run_params['cv_id'] = cv_id
    cv_run(
        run_params['cv_experiment_dir'],
        run_params['data_shards_fps'][cv_id],
        run_params,
        logger
    )


if __name__ == '__main__':

    cv()

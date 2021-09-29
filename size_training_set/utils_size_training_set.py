""" Utility functions for size training set experiment. """

# 3rd party
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model as plot_model
from astropy import stats

import src.utils_predict as utils_predict
import src.utils_train as utils_train
from models.models_keras import create_ensemble
from src.utils_dataio import InputFnCV as InputFn
from src.utils_dataio import get_data_from_tfrecord
from src.utils_metrics import get_metrics
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src_preprocessing.preprocess import centering_and_normalization
from src_preprocessing.tf_util import example_util
from src_preprocessing.utils_preprocessing import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc


def check_normalized_features(norm_features, norm_data_dir, tceid):
    """ Check if normalized features have any non-finite values.

    :param norm_features: dict, normalized features
    :param norm_data_dir: Path, normalized TFRecord directory
    :param tceid: str, TCE ID
    :return:
    """

    feature_str_arr = []
    for feature_name, feature_val in norm_features.items():
        if np.any(~np.isfinite(feature_val)):
            feature_str_arr.append(f'Feature {feature_name} has non-finite values.')

    if len(feature_str_arr) > 0:
        with open(norm_data_dir / f'check_normalized_features_{tceid}.txt', 'w') as file:
            for feature_str in feature_str_arr:
                file.write(feature_str)


def compute_normalization_stats(scalar_params, train_data_fps, aux_params, norm_dir, verbose=None):
    """ Compute normalization statistics for features.

    :param scalar_params: dict, scalar parameters to be normalized and auxiliary information (e.g., missing values)
    :param train_data_fps: list, Paths for training set TFRecords
    :param aux_params: dict, auxiliary parameters for normalization
    :param norm_dir: Path, directory to save normalization statistics
    :param verbose: bool
    :return:
    """

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
        if scalar_params[scalar_param]['missing_value'] is not None:
            if scalar_param == 'wst_depth':
                scalar_param_vals = scalar_param_vals[
                    np.where(scalar_param_vals > scalar_params[scalar_param]['missing_value'])]
            else:
                scalar_param_vals = scalar_param_vals[
                    np.where(scalar_param_vals != scalar_params[scalar_param]['missing_value'])]

        # remove non-finite values
        scalar_param_vals = scalar_param_vals[np.isfinite(scalar_param_vals)]

        # log transform the data
        if scalar_params[scalar_param]['log_transform']:

            # add constant value
            if not np.isnan(scalar_params[scalar_param]['log_transform_eps']):
                scalar_param_vals += scalar_params[scalar_param]['log_transform_eps']

            scalar_param_vals = np.log10(scalar_param_vals)

        # compute median as robust estimate of central tendency
        scalar_norm_stats[scalar_param]['median'] = np.median(scalar_param_vals)
        # compute MAD std as robust estimate of deviation from central tendency
        scalar_norm_stats[scalar_param]['mad_std'] = stats.mad_std(scalar_param_vals) \
            if scalar_param not in ['tce_rb_tcount0n'] else np.std(scalar_param_vals)

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
        'clip_value': 30  # arcsec, should be out-of-FOV for Kepler data
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
    """ Perform normalization of the data.

    :param src_data_fp: Path, source TFRecord file path
    :param norm_stats: dict, normalization statistics
    :param aux_params: dict, auxiliary parameters for normalization
    :param norm_data_dir: Path, directory to save normalized TFRecords data
    :return:
    """

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

                replace_flag = False
                # check if there is a placeholder for missing value
                if norm_stats['scalar_params'][scalar_param]['info']['missing_value'] is not None:
                    if scalar_param in ['wst_depth']:
                        replace_flag = scalar_param_val < \
                                       norm_stats['scalar_params'][scalar_param]['info']['missing_value']
                    else:
                        replace_flag = scalar_param_val == \
                                       norm_stats['scalar_params'][scalar_param]['info']['missing_value']

                if ~np.isfinite(scalar_param_val):  # always replace if value is non-finite
                    replace_flag = True

                if replace_flag:
                    # replace by a defined value
                    if norm_stats['scalar_params'][scalar_param]['info']['replace_value'] is not None:
                        scalar_param_val = norm_stats['scalar_params'][scalar_param]['info']['replace_value']
                    else:  # replace by the median
                        scalar_param_val = norm_stats['scalar_params'][scalar_param]['median']

                # log transform the data (assumes data is non-negative after adding eps)
                # assumes that the median is already log-transformed, but not any possible replace value
                if norm_stats['scalar_params'][scalar_param]['info']['log_transform'] and \
                        not (replace_flag and scalar_param_val == norm_stats['scalar_params'][scalar_param]['median']):

                    # add constant value
                    if not np.isnan(norm_stats['scalar_params'][scalar_param]['info']['log_transform_eps']):
                        scalar_param_val += norm_stats['scalar_params'][scalar_param]['info']['log_transform_eps']

                    scalar_param_val = np.log10(scalar_param_val)

                # clipping the data
                if not np.isnan(norm_stats['scalar_params'][scalar_param]['info']['clip_factor']):
                    scalar_param_val = np.clip([scalar_param_val],
                                               norm_stats['scalar_params'][scalar_param]['median'] -
                                               norm_stats['scalar_params'][scalar_param]['info']['clip_factor'] *
                                               norm_stats['scalar_params'][scalar_param]['mad_std'],
                                               norm_stats['scalar_params'][scalar_param]['median'] +
                                               norm_stats['scalar_params'][scalar_param]['info']['clip_factor'] *
                                               norm_stats['scalar_params'][scalar_param]['mad_std']
                                               )[0]
                # TODO: replace by clipping after standardization with [-clip_factor, clip_factor]

                # standardization
                scalar_param_val = (scalar_param_val - norm_stats['scalar_params'][scalar_param]['median']) / \
                                   norm_stats['scalar_params'][scalar_param]['mad_std']
                assert not np.isnan(scalar_param_val)

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

            loc_centr_view_std_clip = np.clip(loc_centr_view,
                                              a_max=norm_stats['centroid']['local_centr_view']['clip_'
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

            # check normalized features
            targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
            tceIdentifierTfrec = example.features.feature['tce_plnt_num'].int64_list.value[0]
            check_normalized_features(norm_features, norm_data_dir, f'{targetIdTfrec}-{tceIdentifierTfrec}')

            # write features into the destination TFRecords
            for norm_feature_name, norm_feature in norm_features.items():
                example_util.set_float_feature(example, norm_feature_name, norm_feature, allow_overwrite=True)

            writer.write(example.SerializeToString())


def processing_data_run(data_shards_fps, run_params, run_dir, logger=None):
    logger.info(f'[run_iter_{run_params["run_id"]}] Creating training set...')

    dest_data_dir = run_dir / 'data'
    dest_data_dir.mkdir(exist_ok=True)

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    jobs = [(file, run_params["train_set_frac"], dest_data_dir, run_params['train_shards_tbl'], run_params['rng'])
            for file in data_shards_fps['train']]
    async_results = [pool.apply_async(select_train_set, job) for job in jobs]
    pool.close()
    train_set_tbl = pd.concat([async_result.get() for async_result in async_results])
    train_set_tbl.to_csv(run_dir / 'trainset_tbl.csv', index=False)

    data_shards_fps['train'] = [dest_data_dir / data_fp.name for data_fp in data_shards_fps['train']]

    logger.info(f'[run_iter_{run_params["run_id"]}] Creating validation set...')

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    jobs = [(file, run_params["train_set_frac"], dest_data_dir, run_params['val_shards_tbl'], run_params['rng'])
            for file in data_shards_fps['val']]
    async_results = [pool.apply_async(select_train_set, job) for job in jobs]
    pool.close()
    train_set_tbl = pd.concat([async_result.get() for async_result in async_results])
    train_set_tbl.to_csv(run_dir / 'valset_tbl.csv', index=False)

    data_shards_fps['val'] = [dest_data_dir / data_fp.name for data_fp in data_shards_fps['val']]

    logger.info(f'[run_iter_{run_params["run_id"]}] Computing normalization statistics...')

    norm_stats_dir = dest_data_dir / 'norm_stats'
    norm_stats_dir.mkdir(exist_ok=True)

    p = multiprocessing.Process(target=compute_normalization_stats,
                                args=(run_params['scalar_params'],
                                      data_shards_fps['train'],
                                      run_params['norm'],
                                      norm_stats_dir,
                                      False))
    p.start()
    p.join()

    logger.info(f'[run_iter_{run_params["run_id"]}] Normalizing the data')
    norm_data_dir = run_dir / 'norm_data'
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


def train_model(model_id, base_model, n_epochs, config, features_set, data_fps, res_dir, online_preproc_params,
                data_augmentation, callbacks_dict, opt_metric, min_optmetric, logger, verbose=False):
    model_dir = res_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    model_dir_sub = model_dir / f'model{model_id}'
    model_dir_sub.mkdir(exist_ok=True)

    if 'tensorboard' in callbacks_dict:
        callbacks_dict['tensorboard'].log_dir = model_dir_sub

    # instantiate Keras model
    model = base_model(config, features_set).kerasModel

    if model_id == 0:
        with open(model_dir_sub / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        plot_model(model,
                   to_file=model_dir_sub / 'model.png',
                   show_shapes=False,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=96)

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
    # logger.info(f'[model_{model_id}] Training model...')
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

    # save model, features and config used for training this model
    model.save(model_dir_sub / f'model{model_id}.h5')

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


def eval_ensemble(models_filepaths, config, features_set, data_fps, data_fields, generate_csv_pred, res_dir,
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

    with open(res_dir / 'ensemble_summary.txt', 'w') as f:
        ensemble_model.summary(print_fn=lambda x: f.write(x + '\n'))
    # plot ensemble model and save the figure
    plot_model(ensemble_model,
               to_file=res_dir / 'ensemble.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96)

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

        # logger.info(f'[ensemble] Evaluating on dataset {dataset}')
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

    with open(res_dir / 'ensemble_summary.txt', 'w') as f:
        ensemble_model.summary(print_fn=lambda x: f.write(x + '\n'))
    # plot ensemble model and save the figure
    plot_model(ensemble_model,
               to_file=res_dir / 'ensemble.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96)

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

    print('[ensemble] Plotting evaluation results...')
    # draw evaluation plots
    for dataset in datasets:
        plot_class_distribution(output_cl[dataset],
                                res_dir / f'ensemble_class_scoredistribution_{dataset}.png')

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


def select_train_set(src_data_fp, frac_set, dest_data_dir, train_shards_tbl_fp, rng):
    """ Randomly choose examples to create a training set that is a fraction `frac_set` of the original training set.

    :param src_data_fp: Path, file path to source TFRecord
    :param frac_set: float, fraction of the training set that is selected randomly ([0, 1])
    :param dest_data_dir: Path, destination data directory to store new TFRecord files
    :param train_shards_tbl_fp: Path, training set shard table file path
    :param rng: NumPy rng, random generator
    :return:
        train_set_df: pandas Dataframe, training set table
    """

    data_to_tbl = []

    train_shards_tbl = pd.read_csv(train_shards_tbl_fp)
    # # randomly selected fraction of training set
    # train_shards_tbl = train_shards_tbl.sample(frac=frac_set, random_state=rng.bit_generator,
    #                                            replace=False).reset_index(drop=True)
    # # randomly selected fraction of AFPs in the training set
    # train_shards_tbl = pd.concat([train_shards_tbl.loc[train_shards_tbl['label'] != 'AFP'],
    #                               train_shards_tbl.loc[train_shards_tbl['label'] == 'AFP'].sample(frac=frac_set, random_state=rng.bit_generator, replace=False).reset_index(drop=True)
    #                               ])
    # train_shards_tbl = train_shards_tbl.sample(frac=1).reset_index(drop=True)
    # randomly select fraction of AFPs+matched NTPs to AFPs in the training set
    train_shards_tbl = pd.concat(
        [train_shards_tbl.loc[~((train_shards_tbl['label'] == 'AFP') | (train_shards_tbl['is_secondary_tce']))],
         train_shards_tbl.loc[(train_shards_tbl['label'] == 'AFP') | (train_shards_tbl['is_secondary_tce'])].sample(
             frac=frac_set, random_state=rng.bit_generator, replace=False).reset_index(drop=True)
         ])
    train_shards_tbl = train_shards_tbl.sample(frac=1).reset_index(drop=True)

    with tf.io.TFRecordWriter(str(dest_data_dir / src_data_fp.name)) as writer:
        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # get label and other data for the example
            target_id = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]

            if len(train_shards_tbl.loc[(train_shards_tbl['target_id'] == target_id) &
                                        (train_shards_tbl['tce_plnt_num'] == tce_plnt_num)]) == 1:
                label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                data_to_tbl.append([target_id, tce_plnt_num, label])

                writer.write(example.SerializeToString())

    train_set_df = pd.DataFrame(data=data_to_tbl, columns=['target_id', 'tce_plnt_num', 'label'])

    return train_set_df

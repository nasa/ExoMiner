""" Utility functions for label noise injection. """

# 3rd party
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model as plot_model

# local
from src_preprocessing.tf_util import example_util
import src.utils_predict as utils_predict
import src.utils_train as utils_train
from models.models_keras import create_ensemble
from src.utils_dataio import InputFnCV as InputFn
from src.utils_dataio import get_data_from_tfrecord
from src.utils_metrics import get_metrics
from src.utils_visualization import plot_class_distribution, plot_precision_at_k


def processing_data_run(data_shards_fps, run_params, cv_run_dir, logger=None):
    # norm_stats_dir = cv_run_dir / 'norm_stats'
    # norm_stats_dir.mkdir(exist_ok=True)
    #
    # logger.info(f'[cv_iter_{run_params["cv_id"]}] Computing normalization statistics')
    # # if run_params['cv_id'] == 1:
    # #     aaa
    # # with tf.device('/gpu:0'):
    # #     compute_normalization_stats(run_params['scalar_params'], data_shards_fps['train'], run_params['norm'],
    # #                                 norm_stats_dir, False)
    # # with tf.device('/device:CPU:0'):
    # p = multiprocessing.Process(target=compute_normalization_stats,
    #                             args=(run_params['scalar_params'],
    #                                   data_shards_fps['train'],
    #                                   run_params['norm'],
    #                                   norm_stats_dir,
    #                                   False))
    # p.start()
    # p.join()
    # # compute_normalization_stats(run_params['scalar_params'],
    # #                                   data_shards_fps['train'],
    # #                                   run_params['norm'],
    # #                                   norm_stats_dir,
    # #                                   False)

    # logger.info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data')
    # norm_data_dir = cv_run_dir / 'norm_data'
    # norm_data_dir.mkdir(exist_ok=True)
    #
    # # load normalization statistics
    # norm_stats = {
    #     'scalar_params': np.load(norm_stats_dir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item(),
    #     'fdl_centroid': np.load(norm_stats_dir / 'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item(),
    #     'centroid': np.load(norm_stats_dir / 'train_centroid_norm_stats.npy', allow_pickle=True).item()
    # }
    #
    # pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    # jobs = [(file, norm_stats, run_params['norm'], norm_data_dir)
    #         for file in np.concatenate(list(data_shards_fps.values()))]
    # async_results = [pool.apply_async(normalize_data, job) for job in jobs]
    # pool.close()
    # for async_result in async_results:
    #     async_result.get()
    #
    # data_shards_fps_norm = {dataset: [norm_data_dir / data_fp.name for data_fp in data_fps]
    #                         for dataset, data_fps in data_shards_fps.items()}

    noise_label_data_dir = cv_run_dir / 'data'
    noise_label_data_dir.mkdir(exist_ok=True)

    logger.info(f'[run_iter_{run_params["run_id"]}] Adding label noise to training set...')

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    # flip dispositions based on probability transition matrix for the different categories
    # jobs = [(file, run_params["prob_transition_mat_run"], noise_label_data_dir, run_params['rng'])
    #         for file in data_shards_fps['train']]
    # async_results = [pool.apply_async(switch_labels, job) for job in jobs]
    # flip dispositions based on tau and table
    jobs = [(file, run_params["prob_transition_mat_run"], noise_label_data_dir, run_params['flip_table'])
            for file in data_shards_fps['train']]
    async_results = [pool.apply_async(switch_labels_from_table, job) for job in jobs]
    pool.close()
    flip_label_tbl = pd.concat([async_result.get() for async_result in async_results])
    flip_label_tbl.to_csv(cv_run_dir / 'flip_label_trainset_tbl.csv', index=False)

    data_shards_fps['train'] = [noise_label_data_dir / data_fp.name for data_fp in data_shards_fps['train']]

    logger.info(f'[run_iter_{run_params["run_id"]}] Adding label noise to validation set...')

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    jobs = [(file, run_params["prob_transition_mat_run"], noise_label_data_dir, run_params['rng'])
            for file in data_shards_fps['val']]
    async_results = [pool.apply_async(switch_labels, job) for job in jobs]
    pool.close()
    flip_label_tbl = pd.concat([async_result.get() for async_result in async_results])
    flip_label_tbl.to_csv(cv_run_dir / 'flip_label_valset_tbl.csv', index=False)

    data_shards_fps['val'] = [noise_label_data_dir / data_fp.name for data_fp in data_shards_fps['val']]

    return data_shards_fps


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


def switch_labels(src_data_fp, prob_transition, dest_data_dir, rng):
    """ Switch label for each example with a given probability obtained from the probability transition matrix.

    :param src_data_fp: Path, file path to source TFRecord
    :param prob_transition: dict, probability transition values for each class
    :param dest_data_dir: Path, destination data directory to store new TFRecord files
    :param rng: NumPy rng, random generator
    :return:
        flip_label_df: pandas Dataframe, describes flip label for each example in the TFRecord file
    """

    data_to_tbl = []

    with tf.io.TFRecordWriter(str(dest_data_dir / src_data_fp.name)) as writer:
        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # get label and other data for the example
            target_id = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]
            label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            # flip label using the probability transition matrix
            flip_label = rng.choice(prob_transition['classes'],
                                    p=prob_transition['prob'][prob_transition['classes'].index(label), :])

            flip = flip_label != label

            if flip:
                # overwrite label
                example_util.set_bytes_feature(example, 'label', [flip_label], allow_overwrite=True)

            data_to_tbl.append([target_id, tce_plnt_num, label, flip_label, flip])

            writer.write(example.SerializeToString())

    flip_label_df = pd.DataFrame(data=data_to_tbl, columns=['target_id', 'tce_plnt_num', 'label', 'new_label', 'flip'])

    return flip_label_df


def switch_labels_from_table(src_data_fp, prob_transition, dest_data_dir, flip_tbl):
    """ Switch label for each example with a given probability obtained from the probability transition matrix.

    :param src_data_fp: Path, file path to source TFRecord
    :param prob_transition: dict, probability transition values for each class
    :param dest_data_dir: Path, destination data directory to store new TFRecord files
    :param flip_tbl: pandas DataFrame, table used to flip items
    :return:
        flip_label_df: pandas Dataframe, describes flip label for each example in the TFRecord file
    """

    top_num_items_to_flip = int(prob_transition['tau'] * len(flip_tbl))
    flip_tbl = flip_tbl.iloc[:top_num_items_to_flip]

    data_to_tbl = []

    with tf.io.TFRecordWriter(str(dest_data_dir / src_data_fp.name)) as writer:
        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_data_fp))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            # get label and other data for the example
            target_id = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]
            label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            tce_found = flip_tbl.loc[(flip_tbl['target_id'] == target_id) & (flip_tbl['tce_plnt_num'] == tce_plnt_num)]

            flip = len(tce_found) == 1

            if flip:
                if label == 'PC':
                    flip_label = 'AFP'
                else:
                    flip_label = 'PC'

                # overwrite label
                example_util.set_bytes_feature(example, 'label', [flip_label], allow_overwrite=True)
            else:
                flip_label = label

            data_to_tbl.append([target_id, tce_plnt_num, label, flip_label, flip])

            writer.write(example.SerializeToString())

    flip_label_df = pd.DataFrame(data=data_to_tbl, columns=['target_id', 'tce_plnt_num', 'label', 'new_label', 'flip'])

    return flip_label_df

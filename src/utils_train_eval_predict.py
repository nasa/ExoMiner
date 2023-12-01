""" Utility functions for training, evaluating and running inference with a Keras model. """

# 3rd party
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

# local
from src.utils_dataio import InputFnv2 as InputFn
from src.utils_metrics import get_metrics, get_metrics_multiclass
from models.utils_models import create_ensemble, compile_model
from models.models_keras import Time2Vec


def train(model, config, model_dir_sub, model_id=1, logger=None):
    """ Train a model with a given configuration. Support for model selection using a validation set and early stopping
    callback.

    :param model: model, TF Keras model
    :param config: dict, configuration for model hyperparameters
    :param model_dir_sub: Path, model directory
    :param model_id: int, model id
    :param logger: logger
    :return:
        res, dict with training results
    """

    # save model, features and config used for training this model
    if model_id == 0 and config['plot_model']:
        # save plot of model
        plot_model(model,
                   to_file=model_dir_sub / 'model_train.png',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=48)

    # get model summary
    if config['rank'] is None or config['rank'] == 0:
        # if logger is not None:
        #     model.summary(print_fn=lambda x: logger.info(x + '\n'))
        # else:
        with open(model_dir_sub / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # setup metrics to be monitored
    if not config['config']['multi_class']:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])
    else:
        metrics_list = get_metrics_multiclass(label_map=config['label_map'])

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)

    # input function for training, validation and test
    train_input_fn = InputFn(
        file_paths=config['datasets_fps']['train'],
        batch_size=config['training']['batch_size'],
        mode='TRAIN',
        label_map=config['label_map'],
        data_augmentation=config['training']['data_augmentation'],
        online_preproc_params=config['training']['online_preprocessing_params'],
        features_set=config['features_set'],
        category_weights=config['training']['category_weights'],
        multiclass=config['config']['multi_class'],
        use_transformer=config['config']['use_transformer'],
        feature_map=config['feature_map'],
        shuffle_buffer_size=config['training']['shuffle_buffer_size'],
        label_field_name=config['label_field_name'],
    )
    if 'val' in config['datasets']:
        val_input_fn = InputFn(
            file_paths=config['datasets_fps']['val'],
            batch_size=config['training']['batch_size'],
            mode='EVAL',
            label_map=config['label_map'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            use_transformer=config['config']['use_transformer'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )
    else:
        val_input_fn = None

    # fit the model to the training data
    if logger is None:
        print('Training model...')
    else:
        logger.info('Training model...')
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=config['training']['n_epochs'],
                        verbose=config['verbose_model'],
                        callbacks=config['callbacks_list']['train'],
                        validation_split=0.,
                        validation_data=val_input_fn() if val_input_fn is not None else None,
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

    if logger is None:
        print('Saving model...')
    else:
        logger.info('Saving model...')

    # save model
    model.save(model_dir_sub / f'model{model_id}.keras')

    res = history.history

    np.save(model_dir_sub / 'res_eval.npy', res)

    return res


def train_model(base_model, config, model_dir_sub, model_id=1, logger=None):
    """ Train a model with a given configuration. Support for model selection using a validation set and early stopping
    callback.

    :param base_model: model fn, core model to be used
    :param config: dict, configuration for model hyperparameters
    :param model_dir_sub: Path, model directory
    :param model_id: int, model id
    :param logger: logger
    :return:
        res, dict with training results
    """

    # instantiate model
    model = base_model(config, config['features_set']).kerasModel

    # train model
    res = train(model, config, model_dir_sub, model_id=model_id, logger=logger)

    return res


def evaluate_model(config, save_dir, logger=None):
    """ Evaluate a model on a given configuration. Support for creating an average score ensemble based on a set of
    models (`config['paths']['models_filepaths']`).

    :param config: dict, configuration parameters
    :param save_dir: Path, save directory
    :param logger: logger
    :return:
        res, dict with evaluation results
    """

    # load model(s)
    if logger is None:
        print('Loading model...')
    else:
        logger.info('Loading model...')
    model_list = []
    custom_objects = {"Time2Vec": Time2Vec}
    with keras.utils.custom_object_scope(custom_objects):
        for model_i, model_filepath in enumerate(config['paths']['models_filepaths']):
            model = load_model(filepath=model_filepath, compile=False)
            model._name = f'model{model_i}'
            model_list.append(model)

    if len(model_list) == 1:
        model = model_list[0]
    else:  # create average score ensemble
        model = create_ensemble(features=config['features_set'],
                                models=model_list,
                                feature_map=config['feature_map'])
        # save ensemble
        model.save(save_dir / 'ensemble_model.keras')

    if config['write_model_summary']:
        with open(save_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot ensemble model and save the figure
    if config['plot_model']:
        keras.utils.plot_model(model,
                               to_file=save_dir / 'model_eval.png',
                               show_shapes=False,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=False,
                               dpi=96)

    # set up metrics to be monitored
    if not config['config']['multi_class']:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])
    else:  # metrics for multiclass setting
        metrics_list = get_metrics_multiclass(config['label_map'])

    # compile model - loss and metrics
    model = compile_model(model, config, metrics_list, train=False)

    # initialize results dictionary for the evaluated datasets
    res = {}
    for dataset in config['datasets']:

        if dataset == 'predict':
            continue

        if logger is None:
            print(f'Evaluating on dataset {dataset}')
        else:
            logger.info(f'Evaluating on dataset {dataset}')

        # input function for evaluating on each dataset
        eval_input_fn = InputFn(file_paths=config['datasets_fps'][dataset],
                                batch_size=config['evaluation']['batch_size'],
                                mode='EVAL',
                                label_map=config['label_map'],
                                features_set=config['features_set'],
                                online_preproc_params=None,
                                multiclass=config['config']['multi_class'],
                                use_transformer=config['config']['use_transformer'],
                                feature_map=config['feature_map'],
                                label_field_name=config['label_field_name'],
                                )

        callbacks_list = []
        # for callback_name in callbacks_dict:
        #     if 'layer' in callback_name:
        #         callbacks_dict[callback_name].input_fn = eval_input_fn()

        # evaluate model in the given dataset
        res_eval = model.evaluate(x=eval_input_fn(),
                                  y=None,
                                  batch_size=None,
                                  verbose=config['verbose_model'],
                                  sample_weight=None,
                                  steps=None,
                                  callbacks=callbacks_list if dataset == 'train' else None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False)

        # add evaluated dataset metrics to result dictionary
        for metric_name_i, metric_name in enumerate(model.metrics_names):
            res[f'{dataset}_{metric_name}'] = res_eval[metric_name_i]
            # res[metric_name] = res_eval[metric_name_i]

    return res


def predict_model(config, save_dir, logger=None):
    """ Run inference using a model. Support for creating an average score ensemble based on a set of
    models (`config['paths']['models_filepaths']`).

    :param config: dict, configuration parameters
    :param save_dir: Path, save directory
    :param logger: logger
    :return:
        scores, dict with predicted scores for each data set
    """

    # load model(s)
    if logger is None:
        print('Loading model...')
    else:
        logger.info('Loading model...')
    model_list = []
    custom_objects = {"Time2Vec": Time2Vec}
    with keras.utils.custom_object_scope(custom_objects):
        for model_i, model_filepath in enumerate(config['paths']['models_filepaths']):
            model = load_model(filepath=model_filepath, compile=False)
            model._name = f'model{model_i}'
            model_list.append(model)

    if len(model_list) == 1:
        model = model_list[0]
    else:  # create average score ensemble
        model = create_ensemble(features=config['features_set'],
                                models=model_list,
                                feature_map=config['feature_map'])
        # save ensemble
        model.save(save_dir / 'ensemble_model.keras')

    if config['write_model_summary']:
        with open(save_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot model and save the figure
    if config['plot_model']:
        keras.utils.plot_model(model,
                               to_file=save_dir / 'model_predict.png',
                               show_shapes=False,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=False,
                               dpi=96)

    # predict on given datasets
    scores = {dataset: [] for dataset in config['datasets']}
    for dataset in scores:

        if logger is None:
            print(f'Predicting on dataset {dataset}...')
        else:
            logger.info(f'Predicting on dataset {dataset}...')

        predict_input_fn = InputFn(
            file_paths=config['datasets_fps'][dataset],  # str(config['paths']['tfrec_dir']) + '/' + dataset + '*',
            batch_size=config['inference']['batch_size'],
            mode='PREDICT',
            label_map=config['label_map'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            use_transformer=config['config']['use_transformer'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )

        scores[dataset] = model.predict(
            predict_input_fn(),
            batch_size=None,
            verbose=config['verbose_model'],
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )

    return scores


def write_performance_metrics_to_txt_file(save_dir, datasets, res_eval):
    """ Write performance metrics in dictionary `res_eval` based on a model's evaluation.

    Args:
        save_dir: Path, save directory
        datasets: list, data sets for which to save metrics
        res_eval: dict, performance metrics for each data set (should include data sets in `datasets`)

    Returns:

    """

    # write results to a txt file
    with open(save_dir / 'loss_and_performance_metrics.txt', 'w') as res_file:

        str_aux = f'Performance metrics for the model\n'
        res_file.write(str_aux)

        for dataset in datasets:  # iterate over data sets
            if dataset != 'predict':  # no metrics for unlabeled data set

                # grab metrics names for data set
                res_eval_dataset_metrics_names = [metric_name for metric_name in res_eval.keys()
                                                  if dataset in metric_name]

                str_aux = f'Dataset: {dataset}\n'
                res_file.write(str_aux)

                for metric in res_eval_dataset_metrics_names:
                    if isinstance(res_eval[metric], float):  # only write metrics that are scalars
                        str_aux = f'{metric}: {res_eval[f"{metric}"]}\n'
                        res_file.write(str_aux)

            res_file.write('\n')


def set_tf_data_type_for_features(features_set):
    """ Set TF data types for features in the feature set.

    Args:
        features_set: dict, each key is the name of a feature that maps to a dictionary with keys 'dim' and 'dtype'.
        'dim' is a list that describes the dimensionality of the feature and 'dtype' the data type of the feature.
        'dtype' should be a string (either 'float' - mapped to tf.float32; or 'int' - mapped to tf.int64).

    Returns:
        features_set: the data type is now a TensorFlow data type

    """

    # choose features set
    for feature_name, feature in features_set.items():
        if feature['dtype'] == 'float':
            features_set[feature_name]['dtype'] = tf.float32
        if feature['dtype'] == 'int':
            features_set[feature_name]['dtype'] = tf.int64

    return features_set

""" Utility functions for training, evaluating and running inference with a Keras model. """

# 3rd party
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

    # print model summary
    if config['rank'] is None or config['rank'] == 0:
        if logger is not None:
            model.summary(print_fn=lambda x: logger.info(x + '\n'))
        else:
            with open(model_dir_sub / 'model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        # model.summary()

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
        file_paths=config['datasets_fps']['train'],  # str(config['paths']['tfrec_dir']) + '/train*',
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
    model.save(model_dir_sub / f'model{model_id}.h5')

    res = history.history

    np.save(model_dir_sub / 'res_eval.npy', res)

    return res


def train_model(base_model, config, model_dir_sub, model_id=1, logger=None):
    """ Train a model with a given configuration. Support for model selection using a validation set and early stopping
    callback.

    :param base_model: model fn, core model to be used
    :param config: dict, configuration for model hyper-parameters
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


def evaluate_model(config, logger=None):
    """ Evaluate a model on a given configuration. Support for creating an average score ensemble based on a set of
    models (`config['paths']['models_filepaths']`).

    :param config: dict, configuration parameters
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
        model.save(config['paths']['experiment_dir'] / 'ensemble_model.h5')

    if logger is not None:
        model.summary(print_fn=lambda x: logger.info(x + '\n'))
    else:
        with open(config['paths']['experiment_dir'] / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot ensemble model and save the figure
    if config['plot_model']:
        keras.utils.plot_model(model,
                               to_file=config['paths']['experiment_dir'] / 'model_eval.png',
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

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)

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
        eval_input_fn = InputFn(file_paths=config['datasets_fps'][dataset],  # str(config['paths']['tfrec_dir']) + '/{}*'.format(dataset),
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


def predict_model(config, logger=None):
    """ Run inference using a model. Support for creating an average score ensemble based on a set of
    models (`config['paths']['models_filepaths']`).

    :param config: dict, configuration parameters
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
        model.save(config['paths']['experiment_dir'] / 'ensemble_model.h5')

    if logger is not None:
        model.summary(print_fn=lambda x: logger.info(x + '\n'))
    else:
        with open(config['paths']['experiment_dir'] / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot model and save the figure
    if config['plot_model']:
        keras.utils.plot_model(model,
                               to_file=config['paths']['experiment_dir'] / 'model_predict.png',
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

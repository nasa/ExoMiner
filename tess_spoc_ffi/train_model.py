"""
Train a model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model, custom_object_scope
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import yaml
from pathlib import Path
import logging
# from functools import partial

# local
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features
from src.utils.utils_metrics import get_metrics, get_metrics_multiclass
from models.utils_models import compile_model
from models import models_keras
# from src.train.utils_train import filter_examples_tfrecord_obs_type  # ComputePerformanceOnFFIand2min, filter_examples_tfrecord_obs_type
from models.models_keras import Time2Vec, SplitLayer


def freeze_layers(model, layers_to_train_fp):
    """ Freeze layers in a model whose name is in `layers_to_be_frozen`.

    :param model: TF Keras model, model with layers to be frozen
    :param layers_to_train_fp: Path, file path to yaml with list of names of layers to be trainable

    :return: TF Keras model, model with frozen layers
    """

    with open(layers_to_train_fp, 'r') as file:
        layers_to_train_lst = yaml.unsafe_load(file)

    print(f'Layers to be trained: {layers_to_train_lst}')

    for layer in model.layers:
        # if layer.name not in layers_to_train_lst:
        # TODO: experimenting
        if layer.name not in layers_to_train_lst:
            layer.trainable = False

    return model


def train_model(config, model_dir, logger=None):
    """ Train a model.

    :param config: dict, training run parameters
    :param model_dir: Path, directory used to save the trained model
    :param logger: logger

    :return:
    """

    # set tensorflow data type for features in the feature set
    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    if config['model_fp']:  # load pre-existing model
        print(f'Loading model from {config["model_fp"]}')
        custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
        with custom_object_scope(custom_objects):
            model = load_model(filepath=config['model_fp'], compile=False)
    else:
        base_model = getattr(models_keras, config['model_architecture'])
        model = base_model(config, config['features_set']).kerasModel

    if config['plot_model']:
        # save plot of model
        plot_model(model,
                   to_file=model_dir / 'model.png',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=48)

    # get model summary
    with open(model_dir / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # setup metrics to be monitored
    if config['config']['multi_class']:
        metrics_list = get_metrics_multiclass(label_map=config['label_map'])
    else:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])

    if config['trainable_layers_fp']:  # freeze layers
        print(f'Freezing layers based on file: {config["trainable_layers_fp"]}')
        model = freeze_layers(model, config['trainable_layers_fp'])

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
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
            # filter_fn=partial(filter_examples_tfrecord_obs_type, obs_type=1),
        )
    else:
        val_input_fn = None

    # set train callbacks
    config['callbacks_list'] = {'train': []}
    if config['callbacks']['train'] is not None:
        for callback_name, callback_params in config['callbacks']['train'].items():
            if callback_name == 'early_stopping':  # early stopping callback
                config['callbacks_list']['train'].append(
                    callbacks.EarlyStopping(**config['callbacks']['train']['early_stopping']))
            else:
                if logger is None:
                    print(f'Callback {callback_name} not implemented for training. Skipping this callback.')
                else:
                    logger.info(f'Callback {callback_name} not implemented for training. Skipping this callback.')

    config['callbacks_list']['train'] += [
        # ComputePerformanceAfterFilteringLabel(config, model_dir / 'train_cats_monitoring', filter_examples_tfrecord_label),
        # ComputePerformanceOnFFIand2min(config, model_dir / 'train_obs_type_monitoring',
        #                                filter_examples_tfrecord_obs_type),
    ]

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
                        )

    if logger is None:
        print('Saving model...')
    else:
        logger.info('Saving model...')

    # save model
    model.save(model_dir / f'model.keras')

    res = history.history

    np.save(model_dir / 'res_train.npy', res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--model_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--model_fp', type=str, help='Path to pre-existing model', default=None)
    args = parser.parse_args()

    model_dir_fp = Path(args.model_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as file:
        train_config = yaml.unsafe_load(file)

    # set up logger
    train_config['logger'] = logging.getLogger(name=f'train_model')
    logger_handler = logging.FileHandler(filename=model_dir_fp / 'train_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    train_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    train_config['logger'].addHandler(logger_handler)
    train_config['logger'].info(f'Starting training model in {model_dir_fp}')

    train_config['model_fp'] = args.model_fp

    train_model(train_config, model_dir_fp, logger=train_config['logger'])

"""
Train a model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks
import numpy as np
import argparse
import yaml
from pathlib import Path
import logging
import yaml
import traceback
# from functools import partial

# local
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features
from src.utils.utils_metrics import get_metrics, get_metrics_multiclass
from models.utils_models import compile_model
from models import models_keras
# from src.train.utils_train import filter_examples_tfrecord_obs_type  # ComputePerformanceOnFFIand2min, filter_examples_tfrecord_obs_type

def log_info(message, logger=None, include_traceback=False):
    """Log information either to stdout or Python Logger if `logger` is not `None`.

    :param str message: log message
    :param Python Logger logger: logger. If `None`, message is printed to stdout
    :param bool include_traceback: if True, includes traceback (requires being called under and try/exception block). Defaults to False
    """
    
    if include_traceback:
        message += "\n" + traceback.format_exc()
        
    if logger:
        logger.info(message)
    else:
        print(message)


def validate_config(config):
    """Validates configuration for training run.

    :param dict config: configuration for training run
    :raises ValueError: if configuration is missing required tields
    """
    
    required_keys = ['features_set', 'model_architecture', 'training', 'metrics', 'label_map']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def create_input_fn(config, dataset, mode):
    """Creates input function for dataset `datasert` based on keys from a dictionary "datasets_fps" in `config`.

    :param dict config: training run configuration
    :param str dataset: dataset
    :param str mode: input function mode. Either 'TRAIN', 'EVAL', or 'PREDICT'
    :return input fn: input function for a given dataset
    """
    return InputFn(
        file_paths=config['datasets_fps'][dataset],
        batch_size=config['training']['batch_size'],
        mode=mode,
        label_map=config['label_map'],
        data_augmentation=config['training'].get('data_augmentation', {}),
        online_preproc_params=config['training'].get('online_preprocessing_params', {}),
        features_set=config['features_set'],
        category_weights=config['training'].get('category_weights'),
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        shuffle_buffer_size=config['training'].get('shuffle_buffer_size', 1000),
        label_field_name=config['label_field_name'],
        # filter_fn=partial(filter_examples_tfrecord_obs_type, obs_type='ffi'),
    )
    
    
def create_callbacks_train_model(train_callbacks_config, model_dir, logger=None):
    """Create train callbacks.

    :param dict train_callbacks_config: contains parameters for the arguments of callbacks intended to be used during training
    :param Path model_dir: model directory
    :param Python Logger logger: logger, defaults to None
    :return list: list of train callbacks
    """
    
    callbacks_list = []
    
    if train_callbacks_config is not None:
        for callback_name, callback_params in train_callbacks_config.items():
            if callback_name == 'early_stopping':  # early stopping callback
                callbacks_list.append(callbacks.EarlyStopping(**callback_params))
            elif callback_name == 'tensorboard':
                tensorboard_dir = model_dir / 'tensorboard'
                tensorboard_dir.mkdir(exist_ok=True)
                callbacks_list.append(callbacks.TensorBoard(log_dir=tensorboard_dir, **callback_params))
            else:
                if logger is None:
                    print(f'Callback {callback_name} not implemented for training. Skipping this callback.')
                else:
                    logger.info(f'Callback {callback_name} not implemented for training. Skipping this callback.')

    # additional callbacks that are hardcoded instead of using parameters from configuration
    callbacks_list += [
        # ComputePerformanceAfterFilteringLabel(config, model_dir / 'train_cats_monitoring', filter_examples_tfrecord_label),
        # ComputePerformanceOnFFIand2min(config, model_dir / 'train_obs_type_monitoring',
        #                                filter_examples_tfrecord_obs_type),
    ]
    
    return callbacks_list
    
    
def train_model(config, model_dir, logger=None):
    """ Train a model.

    :param config: dict, training run parameters
    :param model_dir: Path, directory used to save the trained model
    :param logger: logger

    :return:
    """

    validate_config(config)
    
    # set tensorflow data type for features in the feature set
    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

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

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)

    # input function for training and validation    
    train_input_fn = create_input_fn(config, 'train', 'TRAIN')
    val_input_fn = create_input_fn(config, 'val', 'EVAL') if 'val' in config['datasets'] else None

    # set train callbacks
    callbacks_train = create_callbacks_train_model(config['callbacks']['train'], model_dir, logger=None)

    # fit the model to the training data
    log_info("Training model...", logger)
    try:
        history = model.fit(x=train_input_fn(),
                            epochs=config['training']['n_epochs'],
                            verbose=config['verbose_model'],
                            callbacks=callbacks_train,
                            validation_data=val_input_fn(),
                            # initial_epoch=N, # resume training from a model trained up to epoch N
                            )
    except Exception as e:
        log_info(f'Training failed: {e}', logger, include_traceback=True)
        raise

    log_info("Finished training model.", logger)
    
    log_info("Saving model...", logger)
    
    # save model
    model.save(model_dir / f'model.keras')

    res = history.history

    # save results into NumPy file
    np.save(model_dir / 'res_train.npy', res)
    
    # save results into human-readable YAML file
    with open(model_dir / 'res_train.yaml', 'w') as f:
        try:
            yaml.dump(history.history, f, default_flow_style=False)
        except Exception as e:
            log_info(f"Unable to save training results into YAML file {str(model_dir / 'res_train.yaml')}:\n{e}", logger, include_traceback=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default=None)
    parser.add_argument('--model_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    model_dir_fp = Path(args.model_dir)
    config_fp = Path(args.config_fp)

    with open(config_fp, 'r') as file:
        train_config = yaml.unsafe_load(file)

    # set up logger
    train_config['logger'] = logging.getLogger(name=f'train_model')
    logger_handler = logging.FileHandler(filename=model_dir_fp / 'train_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    train_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    train_config['logger'].addHandler(logger_handler)
    train_config['logger'].info(f'Starting training model in {model_dir_fp}')

    train_model(train_config, model_dir_fp, logger=train_config['logger'])

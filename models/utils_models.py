"""
Utility functions associated with building TF/Keras models.
"""

# 3rd party
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers


def create_inputs(features, feature_map=None):
    """ Create input layers for the input features.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :param feature_map: maps features' names to features names expected by the model
    :return:
        inputs: dictionary, each key-value pair is a feature_name: feature
    """

    if feature_map is None:
        feature_map = {}

    inputs = {}
    for feature in features:
        if feature not in feature_map:
            inputs[feature] = tf.keras.Input(shape=features[feature]['dim'],
                                             batch_size=None,
                                             name=feature,
                                             dtype=features[feature]['dtype'],
                                             sparse=False,
                                             tensor=None,
                                             )
        else:
            inputs[feature_map[feature]] = tf.keras.Input(shape=features[feature]['dim'],
                                                          batch_size=None,
                                                          name=feature_map[feature],
                                                          dtype=features[feature]['dtype'],
                                                          sparse=False,
                                                          tensor=None,
                                                          )

    return inputs


def create_ensemble(features, models, feature_map=None):
    """ Create a Keras ensemble.

    :param features: dictionary, each key-value pair is a dictionary {'dim': feature_dim, 'dtype': feature_dtype}
    :param models: list, list of Keras models
    :param feature_map: maps features' names to features' names expected by the model

    :return:
        Keras average ensemble
    """

    inputs = create_inputs(features=features, feature_map=feature_map)

    single_models_outputs = [model(inputs) for model in models]

    if len(single_models_outputs) == 1:
        outputs = single_models_outputs
    else:
        outputs = tf.keras.layers.Average(name='avg_model_outputs')(single_models_outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name='ensemble_avg_model')


def compile_model(model, config, metrics_list, train=True):
    """ Compile model.

    :param model: Keras model
    :param config: dict, configuration parameters
    :param metrics_list: list, monitored metrics
    :param train: bool, if set to True, then model is also compiled with optimizer
    :return:
        compiled model
    """

    # set loss
    if config['config']['multi_class']:  # multiclass
        model_loss = losses.CategoricalCrossentropy(from_logits=False, name='categorical_crossentropy')

    else:
        if config['config']['loss'] == 'focal_crossentropy':
            model_loss = losses.BinaryFocalCrossentropy(
                apply_class_balancing=config['config']['focal_class_balancing'],
                alpha=config['config']['focal_loss_alpha'],
                gamma=config['config']['focal_loss_gamma'],
                from_logits=False,
                label_smoothing=0.0,
                axis=-1,
                reduction='sum_over_batch_size',
                name='binary_focal_crossentropy'
            )
        elif config['config']['loss'] == 'crossentropy':
            model_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='binary_crossentropy')

    # set optimizer
    if train:
        if config['config']['optimizer'] == 'Adam':
            model_optimizer = optimizers.Adam(
                learning_rate=config['config']['lr'],
                                              beta_1=0.9,
                                              beta_2=0.999,
                                              epsilon=1e-8,
                                              amsgrad=False,
                                              name='Adam'
                                              )
        elif config['config']['optimizer'] == 'AdamW':
            model_optimizer = optimizers.AdamW(
                learning_rate=config['config']['lr'],
                weight_decay=0.004,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                jit_compile=True,
                name="AdamW",
            )
        elif config['config']['optimizer'] == 'SGD':
            model_optimizer = optimizers.SGD(
                learning_rate=config['config']['lr'],
                momentum=config['config']['sgd_momentum'],
                nesterov=False,
                name='SGD'
            )

        else:
            raise ValueError(f'Optimizer {config["config"]["optimizer"]} is not supported. Choose among Adam, AdamW, '
                             f'and SGD.')

    # compile model with chosen optimizer, loss and monitored metrics
    if train:
        model.compile(optimizer=model_optimizer, loss=model_loss, metrics=metrics_list)
    else:
        model.compile(loss=model_loss, metrics=metrics_list)

    return model

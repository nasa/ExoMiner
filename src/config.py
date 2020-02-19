"""
Utility script for defining configurations.
"""

# local
from src.estimator_util import get_ce_weights


label_map = {'kepler': {True: {"PC": 1,  # True: multi-class, False: binary classification
                               "NTP": 0,
                               "AFP": 2},
                        False: {'PC': 1,
                                'NTP': 0,
                                'AFP': 0}},
             'tess': {True: {"PC": 1,
                             "NTP": 0,
                             "EB": 2,
                             "BEB": 2},
                      False: {"PC": 1,
                              "NTP": 0,
                              "EB": 0,
                              "BEB": 0}}}


def add_default_missing_params(config):
    """ Adds parameters with default values that were not optimized in the HPO study.

    :param config: dict, model parameters tested in the HPO study
    :return:
        config: dict, with additional parameters that were not optimized in the HPO study
    """

    # check parameters not optimized by the HPO study
    default_parameters = {'non_lin_fn': False,
                          'batch_norm': False,
                          'weight_initializer': None,
                          'force_softmax': False,
                          'use_kepler_ce': False,
                          'decay_rate': None,
                          'batch_size': 32,
                          'optimizer': 'SGD',
                          'lr': 1e-5}

    for default_parameter in default_parameters:
        if default_parameter not in config:
            config[default_parameter] = default_parameters[default_parameter]

    return config


def add_dataset_params(satellite, multi_class, use_kepler_ce, config, ce_weights_args):
    """ Adds parameters related to the dataset used - kepler/tess, binary/multi class., labels' map, centroid data,
    CE weights,...

    :param satellite: str, satellite used. Either 'kepler' or 'tess'
    :param multi_class: bool, True for multiclass, binary classification otherwise (PC vs Non-PC)
    :param use_kepler_ce: bool, if True, uses weighted CE
    :param config: dict, model parameters and hyperparameters
    :return:
        config: dict, now with parameters related to the dataset used
    """

    config['satellite'] = satellite
    config['multi_class'] = multi_class
    config['label_map'] = label_map[satellite][multi_class]
    config['ce_weights'], config['n_train_examples'] = get_ce_weights(label_map=config['label_map'], **ce_weights_args)
    config['use_kepler_ce'] = use_kepler_ce

    return config

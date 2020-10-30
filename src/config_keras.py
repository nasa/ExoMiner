"""
Utility script for defining configurations.
"""

import numpy as np

# local
from src.utils_dataio import get_ce_weights

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
                              "BEB": 0,
                              "KP": 1}}}


def add_default_missing_params(config):
    """ Adds parameters with default values that were not optimized in the HPO study.

    :param config: dict, model parameters tested in the HPO study
    :return:
        config: dict, with additional parameters that were not optimized in the HPO study
    """

    default_parameters = {
        'datasets': ['train', 'val', 'test'],
        'clf_thr': 0.5,
        'num_thr': 1000,
        'non_lin_fn': 'relu',
        'batch_norm': False,
        'weight_initializer': None,
        'force_softmax': False,
        'use_kepler_ce': False,
        'decay_rate': None,
        'batch_size': 32,
        # 'optimizer': 'SGD',
        # 'lr': 1e-5,
        # 'k_arr': {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]},
        'k_arr': {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]},  # no PPs
        'k_curve_arr': {
            # 'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
            # 'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
            # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
            'train': np.linspace(25, 1800, 100, endpoint=True, dtype='int'),  # no PPs
            'val': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
            'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
        }
    }

    for default_parameter in default_parameters:
        if default_parameter not in config:
            config[default_parameter] = default_parameters[default_parameter]

    return config


def add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config=None):
    """ Adds parameters related to the dataset used - kepler/tess, binary/multi class., labels' map, centroid data,
    CE weights,...

    :param satellite: str, satellite used. Either 'kepler' or 'tess'
    :param multi_class: bool, True for multiclass, binary classification otherwise (PC vs Non-PC)
    :param use_kepler_ce: bool, if True, uses weighted CE
    :param ce_weights_args: dict, CE weights parameters
    :param config: dict, model parameters and hyperparameters
    :return:
        config: dict, now with parameters related to the dataset used
    """

    if config is None:
        config = {}

    config['satellite'] = satellite
    config['multi_class'] = multi_class
    config['label_map'] = label_map[satellite][multi_class]
    if use_kepler_ce:
        config['ce_weights'] = get_ce_weights(label_map=config['label_map'], **ce_weights_args)
    else:
        config['ce_weights'] = None
    config['use_kepler_ce'] = use_kepler_ce

    return config

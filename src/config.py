"""
Configuration class used to instantiate models with a config given as a dictionary.
"""

# 3rd party
# import os

# local
# if 'nobackup' in os.path.dirname(__file__):
#     from src.estimator_util import get_ce_weights, get_model_dir
# else:
#     from src.estimator_util import get_ce_weights, get_model_dir
from src.estimator_util import get_ce_weights, get_model_dir
# from estimator_util import get_ce_weights, get_model_dir
# import paths

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

    # check parameters not optimized by the HPO study
    default_parameters = {'non_lin_fn': False,
                          'batch_norm': False,
                          'weight_initializer': None,
                          'force_softmax': False,
                          'use_kepler_ce': False,
                          'decay_rate': None}

    for default_parameter in default_parameters:
        if default_parameter not in config:
            config[default_parameter] = default_parameters[default_parameter]

    return config


<<<<<<< HEAD

        # For self-training
        # tfrecord_dir_tpsrejects = '/data5/tess_project/classifiers_Laurent/Kepler_classifier/Astronet_Models/tfrecords/kepler_rejects'
=======
def add_dataset_params(tfrec_dir, satellite, multi_class, config):
>>>>>>> b630a2ba86a219cb217fee4eef798af648442eef

    config['satellite'] = satellite
    config['multi_class'] = multi_class
    config['label_map'] = label_map[satellite][multi_class]
    config['ce_weights'], config['centr_flag'], config['n_train_examples'] = \
        get_ce_weights(config['label_map'], tfrec_dir)

    return config


# class Config(object):
#
#     def __init__(self, n_epochs, batch_size, conv_ls_per_block, dropout_rate, init_conv_filters, init_fc_neurons,
#                  kernel_size, kernel_stride, lr, num_fc_layers, num_glob_conv_blocks, num_loc_conv_blocks, optimizer,
#                  pool_size_glob, pool_size_loc, pool_stride, decay_rate=None, model_dir_path=None, **kwargs):
#
#         if 'sgd_momentum' in kwargs:
#             self.sgd_momentum = kwargs['sgd_momentum']
#         else:
#             self.sgd_momentum = None
#         self.batch_size = batch_size
#         self.conv_ls_per_block = conv_ls_per_block
#         self.decay_rate = decay_rate
#         self.dropout_rate = dropout_rate
#         self.init_conv_filters = init_conv_filters
#         self.init_fc_neurons = init_fc_neurons
#         self.kernel_size = kernel_size
#         self.kernel_stride = kernel_stride
#         self.lr = lr
#         self.num_fc_layers = num_fc_layers
#         self.num_glob_conv_blocks = num_glob_conv_blocks
#         self.num_loc_conv_blocks = num_loc_conv_blocks
#         self.optimizer = optimizer
#         self.pool_size_glob = pool_size_glob
#         self.pool_size_loc = pool_size_loc
#         self.pool_stride = pool_stride
#
#         self.n_epochs = n_epochs
#
#         self.multi_class = False
#         self.force_softmax = False  # Use softmax in case of binary classification?
#         self.use_kepler_ce = False  # If kepler model, use weighted cross entropy?
#
#         self.label_map = {'kepler': {True: {"PC": 1,  # True: multi-class, False: binary classification
#                                        "NTP": 0,
#                                        "AFP": 2},
#                                      False: {'PC': 1,
#                                         'NTP': 0,
#                                         'AFP': 0}},
#                          'tess': {True: {"PC": 1,
#                                      "NTP": 0,
#                                      "EB": 2,
#                                      "BEB": 2},
#                                   False: {"PC": 1,
#                                       "NTP": 0,
#                                       "EB": 0,
#                                       "BEB": 0}}}
#
#         # if 'nobackup' not in os.path.dirname(__file__):
#         #     self.tfrec_dir = '/home/msaragoc/Kepler_planet_finder/tfrecord_kepler'
#         # else:
#         self.tfrec_dir = paths.tfrec_dir
#
#         # For self-training
#         # tfrecord_dir_tpsrejects = '/data5/tess_project/classifiers_Laurent/Kepler_classifier/Astronet_Models/tfrecords/kepler_rejects'
#
#         ######  End of input section #######
#
#         self.satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'
#         self.label_map = self.label_map[self.satellite][self.multi_class]
#         self.ce_weights, self.centr_flag, self.n_train_examples = get_ce_weights(self.label_map, self.tfrec_dir)
#
#         # self.model_dir_custom = get_model_dir()
#         self.model_dir_path = model_dir_path
#         if self.model_dir_path is not None:
#             self.model_dir_custom = get_model_dir(self.model_dir_path)
#         else:
#             self.model_dir_custom = None

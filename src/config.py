"""
Configuration class used to instantiate models with a config given as a dictionary.
"""

import os

if 'nobackup' in os.path.dirname(__file__):
    from src.estimator_util import get_ce_weights, get_model_dir
else:
    from src.estimator_util import get_ce_weights, get_model_dir

# from estimator_util import get_ce_weights, get_model_dir


class Config(object):

    def __init__(self, n_epochs, batch_size, conv_ls_per_block, dropout_rate, init_conv_filters, init_fc_neurons,
                 kernel_size, kernel_stride, lr, num_fc_layers, num_glob_conv_blocks, num_loc_conv_blocks, optimizer,
                 pool_size_glob, pool_size_loc, pool_stride, decay_rate=None, model_dir_path=None, **kwargs):
        if 'sgd_momentum' in kwargs:
            self.sgd_momentum = kwargs['sgd_momentum']
        else:
            self.sgd_momentum = None
        self.batch_size = batch_size
        self.conv_ls_per_block = conv_ls_per_block
        self.decay_rate = decay_rate
        self.dropout_rate = dropout_rate
        self.init_conv_filters = init_conv_filters
        self.init_fc_neurons = init_fc_neurons
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.lr = lr
        self.num_fc_layers = num_fc_layers
        self.num_glob_conv_blocks = num_glob_conv_blocks
        self.num_loc_conv_blocks = num_loc_conv_blocks
        self.optimizer = optimizer
        self.pool_size_glob = pool_size_glob
        self.pool_size_loc = pool_size_loc
        self.pool_stride = pool_stride

        self.n_epochs = n_epochs

        self.multi_class = False
        self.force_softmax = False  # Use softmax in case of binary classification?
        self.use_kepler_ce = False  # If kepler model, use weighted cross entropy?

        self.label_map = {'kepler': {True: {"PC": 1,  # True: multi-class, False: binary classification
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

        # if 'nobackup' not in os.path.dirname(__file__):
        #     self.tfrec_dir = '/home/msaragoc/Kepler_planet_finder/tfrecord_kepler'
        # else:
        # self.tfrec_dir = '/home6/msaragoc/work_dir/data/tfrecord_kepler'
        # self.tfrec_dir = '/home/msaragoc/Kepler_planet_finder/tfrecord_kepler'
<<<<<<< HEAD
<<<<<<< HEAD
        self.tfrec_dir = '/home/msaragoc/Kepler_planet_finder/Data/tfrecord_kepler'
=======
        self.tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_kepler'
>>>>>>> c2e1bc6b21dba7d177f67a97f4c07a4e7adfb4ab
=======
        self.tfrec_dir = '/home/msaragoc/Kepler_planet_finder/Data/tfrecord_kepler'
>>>>>>> 7d211e769bc5a119170b3ec1e1c8cb67f97cb4d7

        # For self-training
        # tfrecord_dir_tpsrejects = '/data5/tess_project/classifiers_Laurent/Kepler_classifier/Astronet_Models/tfrecords/kepler_rejects'

        ######  End of input section #######

        self.satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'
        self.label_map = self.label_map[self.satellite][self.multi_class]
        self.ce_weights, self.centr_flag, self.n_train_examples = get_ce_weights(self.label_map, self.tfrec_dir)

        # self.model_dir_custom = get_model_dir()
        self.model_dir_path = model_dir_path
        if self.model_dir_path is not None:
            self.model_dir_custom = get_model_dir(self.model_dir_path)
        else:
            self.model_dir_custom = None

    # n_epochs = 50
    #
    # lr = 1e-5
    # optimizer = 'Adam'  # 'SGD'
    # # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp', 'piecew_inv_exp'])
    # # sgd_momentum = 0.9
    # batch_size = 64
    # dropout_rate = 0
    # decay_rate = 1e-2
    #
    # num_glob_conv_blocks = 5
    # num_loc_conv_blocks = 2
    #
    # num_fc_layers = 4
    #
    # conv_ls_per_block = 2
    #
    # init_fc_neurons = 512
    #
    # init_conv_filters = 16
    # kernel_size = 5
    # kernel_stride = 1
    #
    # pool_size_glob = 5
    # pool_size_loc = 7
    # pool_stride = 2

    # multi_class = False
    # force_softmax = False  # Use softmax in case of binary classification?
    # use_kepler_ce = False  # If kepler model, use weighted cross entropy?
    #
    # label_map = {'kepler': {True: {"PC": 1,  # True: multi-class, False: binary classification
    #                                "NTP": 0,
    #                                "AFP": 2},
    #                         False: {'PC': 1,
    #                                 'NTP': 0,
    #                                 'AFP': 0}},
    #              'tess': {True: {"PC": 1,
    #                              "NTP": 0,
    #                              "EB": 2,
    #                              "BEB": 2},
    #                       False: {"PC": 1,
    #                               "NTP": 0,
    #                               "EB": 0,
    #                               "BEB": 0}}}
    #
    # if 'nobackup' not in os.path.dirname(__file__):
    #     tfrec_dir = '/home/plopesge/Desktop/dr24_vs_dr25/tfrecord_dr25keplernonwhitened'
    # else:
    #     tfrec_dir = '/nobackupp2/lswilken/Kepler_planet_finder/tfrecords/tfrecord_manual_2dkeplernonwhitened'
    #
    # # For self-training
    # tfrecord_dir_tpsrejects = '/data5/tess_project/classifiers_Laurent/Kepler_classifier/Astronet_Models/tfrecords/kepler_rejects'
    #
    # ######  End of input section #######
    #
    # satellite = 'kepler' # if 'kepler' in tfrec_dir else 'tess'
    # label_map = label_map[satellite][multi_class]
    # ce_weights, centr_flag, n_train_examples = get_ce_weights(label_map, tfrec_dir)
    #
    # model_dir_custom = get_model_dir()

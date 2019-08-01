"""
Baseline configurations from other papers.

[1] Shallue, Christopher J., and Andrew Vanderburg. "Identifying exoplanets with deep learning: A five-planet resonant
chain around kepler-80 and an eighth planet around kepler-90." The Astronomical Journal 155.2 (2018): 94.

[2] Ansdell, Megan, et al. "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning."
The Astrophysical Journal Letters 869.1 (2018): L7.
"""

# Shallue and Vanderburg best configuration [1]
astronet = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7, 'init_conv_filters': 4,
            'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': None, 'kernel_stride': 1, 'pool_stride': 2,
            'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5, 'optimizer': 'Adam', 'kernel_size': 5,
            'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

# Exonet [2] is similar to Astronet, but add the centroid time series as a second channel and merges the stellar
# parameters with the flatten output of the convolutional columns
exonet = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7, 'init_conv_filters': 4,
            'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': None, 'kernel_stride': 1, 'pool_stride': 2,
            'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5, 'optimizer': 'Adam', 'kernel_size': 5,
            'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

# Exonet-XS [2]
# last max pooling layer is a global max pooling layer
exonet_xs = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 2, 'init_conv_filters': 4,
             'conv_ls_per_block': 1, 'dropout_rate': 0, 'decay_rate': None, 'kernel_stride': 1, 'pool_stride': 2,
             'num_fc_layers': 1, 'batch_size': 64, 'lr': 1e-5, 'optimizer': 'Adam', 'kernel_size': 5,
             'num_glob_conv_blocks': 3, 'pool_size_glob': 2}


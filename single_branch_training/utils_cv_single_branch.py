"""
Utility functions used during training, evaluating and predicting in cv.py.
- Custom callbacks, ...
"""

# 3rd party
import tensorflow as tf
import numpy as np
import pandas as pd


class LayerWeightCallback(tf.keras.callbacks.Callback):

    def __init__(self, layers, log_dir):
        """ Callback that reads and saves weight values for a given layer in a set of single-branch models.

        :param layers: dict, key is single-branch model name and value is layer name in that single-branch model
        :param log_dir: dict, log
        """

        super(LayerWeightCallback, self).__init__()
        self.layers = layers
        self.log_dir = log_dir

    def get_data(self):
        """ Get weights from the last FC layer in each single-branch model queried into a dictionary.

        Returns:
            weights_data, dict, `single-branch` maps to a dictionary of weights` and `biases` that contain
            respectively the flattened array of weights and biases for a given single-branch model
        """

        weights_data = {f'{single_branch_name}_{layer_name}': {'weights': None, 'biases': []}
                        for single_branch_name, layer_name in self.layers.items()}
        for single_branch_name, layer_name in self.layers.items():
            single_branch_model = self.model.get_layer(single_branch_name)  # get single-branch model
            layer = single_branch_model.get_layer(layer_name)  # get FC layer
            weights = layer.get_weights()[0].flatten()  # get weights
            biases = layer.get_weights()[1].flatten()  # get biases
            weights_data[f'{single_branch_name}_{layer_name}'] = {'weights': weights, 'biases': biases}

        return weights_data

    def on_epoch_end(self, epoch, logs=None):
        """ Write to a summary the output of a given layer at the end of each epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        weights_data = self.get_data()

        np.save(self.log_dir / f'epoch_{epoch}.npy', weights_data)

    def on_train_end(self, logs=None):
        """ Save weights in a given layer at the end of training into NumPy and csv files.

        Args:
            logs: dict, logs

        Returns:

        """

        weights_all_epochs = {f'{single_branch_name}_{layer_name}': {'weights': None, 'biases': None}
                              for single_branch_name, layer_name in self.layers.items()}
        for epoch_i in range(self.params['epochs']):
            weights_epoch = np.load(self.log_dir / f'epoch_{epoch_i}.npy', allow_pickle=True).item()
            for layer_name, weights in weights_epoch.items():
                if weights_all_epochs[layer_name]['weights'] is None:
                    weights_all_epochs[layer_name]['weights'] = \
                        np.nan * np.ones((self.params['epochs'], len(weights['weights'])))
                    weights_all_epochs[layer_name]['biases'] = \
                        np.nan * np.ones((self.params['epochs'], len(weights['biases'])))

                weights_all_epochs[layer_name]['weights'][epoch_i] = weights_epoch[layer_name]['weights']
                weights_all_epochs[layer_name]['biases'][epoch_i] = weights_epoch[layer_name]['biases']

        np.save(self.log_dir / 'all_epochs.npy', weights_all_epochs)
        for layer_name, weights in weights_all_epochs.items():
            weights_df = pd.DataFrame(weights_all_epochs[layer_name]['weights'].T,
                                      index=[f'weight_{i}'
                                             for i in range(weights_all_epochs[layer_name]['weights'].T.shape[0])])
            biases_df = pd.DataFrame(weights_all_epochs[layer_name]['biases'].T,
                                     index=[f'bias_{i}'
                                            for i in range(weights_all_epochs[layer_name]['biases'].T.shape[0])])

            weights_df.to_csv(self.log_dir / f'{layer_name}_weights_all_epochs.csv')
            biases_df.to_csv(self.log_dir / f'{layer_name}_biases_all_epochs.csv')

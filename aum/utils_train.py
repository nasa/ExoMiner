"""
Utility functions for training models for AUM experiments.
"""

# 3rd party
from tensorflow import keras
import pandas as pd
import numpy as np

# local
from src.utils_dataio import InputFnv2 as InputFn


class TrackLogitsCallback(keras.callbacks.Callback):
    """ Custom callback for tracking logits at the end of each epoch. """
    def __init__(self, tfrec_filepaths, label_map, batch_size, features_set, log_dir, verbose=False):
        """ Creates callback to track logits.

        :param tfrec_filepaths: list, TFRecord shards from which to extract the logits form
        :param label_map: dict, map between category and label id
        :param batch_size: int, batch size
        :param features_set: dict, features set
        :param log_dir: str, directory where logits files are stored
        :param verbose: bool, verbose
        """

        super(TrackLogitsCallback, self).__init__()

        self.tfrec_filepaths = tfrec_filepaths
        self.label_map = label_map
        self.batch_size = batch_size
        self.features_set = features_set
        self.verbose = verbose
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        """ Runs at the end of each epoch.

        :param epoch: int, epoch
        :param logs: dict, log with values for monitored metrics at given epoch
        :return:
        """

        # print(f'On epoch {epoch} end: {logs}')

        # get logit layer output
        layer_logit_output = self.model.get_layer('logits').output
        # build new model that has as output the logit layer
        logit_model = keras.Model(inputs=self.model.input, outputs=layer_logit_output)

        # run inference on the dataset using this model
        predict_input_fn = InputFn(file_paths=self.tfrec_filepaths,
                                   batch_size=self.batch_size,
                                   mode='PREDICT',
                                   label_map=self.label_map,
                                   filter_data=None,
                                   features_set=self.features_set)

        logits = logit_model.predict(predict_input_fn(),
                                     batch_size=None,
                                     verbose=self.verbose,
                                     steps=None,
                                     callbacks=None,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False,
                                     )

        # save logits to csv file
        logits_df = pd.DataFrame(data=logits, columns=np.unique(list(self.label_map.values())))
        logits_df.to_csv(self.log_dir / f'logits_epoch-{epoch}.csv', index=False)

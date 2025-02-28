"""
Utility functions used during training.
- Custom callbacks, ...
"""

# 3rd party
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
from functools import partial
from tensorflow import keras

# local
from src.utils.utils_dataio import InputFnv2 as InputFn


class LayerOutputCallback(tf.keras.callbacks.Callback):

    def __init__(self, input_fn, batch_size, layer_name, summary_writer, buckets, log_dir, num_batches=None,
                 description='custom', ensemble=False):
        """ Callback that writes to a histogram summary the output of a given layer.

        :param input_fn: input function
        :param batch_size: int, batch size
        :param layer_name: str, name of the layer
        :param summary_writer: summary writer
        :param buckets: int, bucket size
        :param log_dir:
        :param num_batches: int, number of batches to extract
        :param description: str, optional description
        :param ensemble: bool, True if dealing with an ensemble of models.
        """

        super(LayerOutputCallback, self).__init__()
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.layer_name = layer_name
        self.summary_writer = summary_writer
        self.description = description
        self.log_dir = log_dir
        self.buckets = buckets
        self.ensemble = ensemble
        self.num_batches = num_batches if num_batches is not None else np.inf
        self.csv_fp = Path(self.log_dir) / f'{self.layer_name}.csv'

    # def on_batch_end(self, batch, logs=None):
    #
    #     layer = [l for l in self.model.layers if l.name == 'convbranch_wscalar_concat'][0]
    #     get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #     print('Batch {} | Input to FC block: '.format(batch), get_layer_output.outputs)

    # def on_train_batch_end(self, batch, logs=None):
    #
    #     num_batches = int(24000 / self.batch_size)
    #     if batch in np.linspace(0, num_batches, 10, endpoint=False, dtype='uint32'):
    #         # pass
    #         layer = [l for l in self.model.layers if l.name == self.layer_name][0]
    #         get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #         for batch_i, (batch_input, batch_label) in enumerate(self.inputFn):
    #             if batch_i == batch:
    #                 batch_output_layer = get_layer_output(batch_input)
    #                 # print('Batch {} | Input to FC block: '.format(batch), batch_output_layer)
    #                 with self.summaryWriter.as_default():
    #                     tf.summary.histogram(data=tf.convert_to_tensor(batch_output_layer, dtype=tf.float32), name='{}_output'.format(self.layer_name), step=batch, buckets=30, description='aaa')

    def get_data(self):

        if self.ensemble:
            models_in_ensemble = [l for l in self.model.layers if 'model' in l.name]
            data = []
            for model in models_in_ensemble:
                data_model = []
                layer = [l for l in model.layers if l.name == self.layer_name][0]
                get_layer_output = tf.keras.backend.function(inputs=model.input, outputs=layer.output)
                for batch_i, (batch_input, batch_label) in enumerate(self.input_fn):
                    batch_output_layer = get_layer_output(batch_input)
                    data_batch = np.concatenate([batch_output_layer, np.expand_dims(batch_label.numpy(), axis=1)],
                                                axis=1)
                    data_model.append(data_batch)
                    if len(data_model) == self.num_batches:
                        break
                data.append(data_model)
        else:
            layer = [l for l in self.model.layers if l.name == self.layer_name][0]
            get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
            data = []
            for batch_i, (batch_input, batch_label) in enumerate(self.input_fn):
                batch_output_layer = get_layer_output(batch_input)
                data_batch = np.concatenate([batch_output_layer, np.expand_dims(batch_label.numpy(), axis=1)],
                                                axis=1)
                data.append(data_batch)
                if len(data) == self.num_batches:
                    break

        return data

    def write_to_csv(self, data):

        if self.ensemble:
            for model_i, model_data in enumerate(data):
                model_csv_fp = self.csv_fp.parent / f'{self.csv_fp.stem}_model{model_i + 1}.csv'
                for batch_data in model_data:
                    data_df = pd.DataFrame(batch_data)
                    if model_csv_fp.exists():
                        data_df.to_csv(model_csv_fp, index=False, mode='a', header=None)
                    else:
                        data_df.to_csv(model_csv_fp, index=False, mode='w')
        else:
            for batch_data in data:
                data_df = pd.DataFrame(batch_data)
                if self.csv_fp.exists():
                    data_df.to_csv(self.csv_fp, index=False, mode='a', header=None)
                else:
                    data_df.to_csv(self.csv_fp, index=False, mode='w')

    # def on_predict_end(self, logs=None):
    #
    #     data = self.get_data()
    #     with self.summary_writer.as_default():
    #
    #         tf.summary.histogram(data=tf.convert_to_tensor([data], dtype=tf.float32),
    #                              name=self.layer_name,
    #                              step=1,
    #                              buckets=self.buckets,
    #                              description=self.description
    #                              )

    def on_test_end(self, logs=None):

        data = self.get_data()

        self.write_to_csv(data)

        # with self.summary_writer.as_default():
        #     tf.summary.histogram(data=tf.convert_to_tensor([np.ndarray.flatten(np.array(data))], dtype=tf.float32),
        #                          name=self.layer_name,
        #                          step=1,
        #                          buckets=self.buckets,
        #                          description=self.description
        #                          )

    def on_epoch_end(self, epoch, logs=None):
        """ Write to a summary the output of a given layer at the end of each epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        data = self.get_data()

        with self.summary_writer.as_default():

            tf.summary.histogram(data=tf.convert_to_tensor([data], dtype=tf.float32),
                                 name=self.layer_name,
                                 step=epoch,
                                 buckets=self.buckets,
                                 description=self.description
                                 )


class PredictDuringFitCallback(tf.keras.callbacks.Callback):
    """
    Callback for predicting scores at the end of the epoch for different data sets.
    """
    def __init__(self, dataset_fps, root_save_dir, batch_size, label_map, features_set, multi_class, md_features,
                 feature_map, data_to_tbl, verbose=False):
        """ Initialize callback.

        Args:
            dataset_fps: dict, each data set key maps to a list of TFRecord file paths
            root_save_dir: Path, root saving directory for the model
            batch_size: int, batch size for inference
            label_map: dict, map between category and label id
            features_set: dict, features used by the model
            multi_class: bool, if True running multiclassification setting
            md_features: bool, if True features with dimension >= 2 are being used
            feature_map: dict, map between TFRecord feature name and model feature name
            data_to_tbl: dict, each data set key maps to dict with features to be integrated in the score table such as examples' ids
            verbose:
        """

        super(PredictDuringFitCallback, self).__init__()
        self.dataset_fps = dataset_fps
        self.save_dir = root_save_dir / 'scores_per_epoch'
        self.save_dir.mkdir()
        self.batch_size = batch_size
        self.label_map = label_map
        self.features_set = features_set
        self.multi_class = multi_class
        self.md_features = md_features
        self.feature_map = feature_map
        self.data_to_tbl = data_to_tbl
        self.verbose = verbose

    def create_score_table(self, scores, data_to_tbl):
        """ Create score table from scores output by the model.

        Args:
            scores: NumPy array, scores for examples
            data_to_tbl: dict, data to display in the scores table besides scores.

        Returns:

        """

        if not self.multi_class:
            data_to_tbl['score'] = scores.ravel()
            # data['predicted class'] = scores_classification[dataset].ravel()
        else:
            for class_label, label_id in self.label_map.items():
                data_to_tbl[f'score_{class_label}'] = scores[:, label_id]
            # data['predicted class'] = scores_classification[dataset]

        data_df = pd.DataFrame(data_to_tbl)

        # sort in descending order of output
        if not self.multi_class:
            data_df.sort_values(by='score', ascending=False, inplace=True)

        return data_df

    def on_epoch_end(self, epoch, logs=None):
        """ Write scores by running inference with the model at the end of the epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        for dataset, fps in self.dataset_fps.items():
            predict_input_fn = InputFn(
                file_paths=fps,
                batch_size=self.batch_size,
                mode='PREDICT',
                label_map=self.label_map,
                features_set=self.features_set,
                multiclass=self.multi_class,
                use_transformer=self.md_features,
                feature_map=self.feature_map
            )

            scores = self.model.predict(
                predict_input_fn(),
                batch_size=None,
                verbose=self.verbose,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
            )

            tbl = self.create_score_table(scores, self.data_to_tbl[dataset])

            tbl.to_csv(self.save_dir / f'scores_{dataset}set_epoch{epoch}.csv', index=False)


def filter_examples_tfrecord_obs_type(parsed_features, label_id, obs_type):
    """ Filters out examples based on `obs_type`.

    Args:
        parsed_features: tf tensor, parsed features for example
        label_id: tf tensor, label id for example
        obs_type: str, observation type to be filtered

    Returns: tf boolean tensor

    """

    # vocabulary = {'ffi': 0, '2min': 1}
    #
    # # table = tf.lookup.StaticHashTable(
    # #     initializer=tf.lookup.KeyValueTensorInitializer(
    # #         keys=list(vocabulary.keys()),
    # #         values=list(vocabulary.values()),
    # #         key_dtype=tf.string,
    # #         value_dtype=tf.int32)
    # #     )
    # table_initializer = tf.lookup.KeyValueTensorInitializer(keys=list(vocabulary.keys()),
    #                                                         values=list(vocabulary.values()),
    #                                                         key_dtype=tf.string,
    #                                                         value_dtype=tf.int32)
    # obs_type_mapping = tf.lookup.StaticHashTable(table_initializer, default_value=-1)
    # encoding_obs_type = vocabulary[obs_type]  # obs_type_mapping.lookup(obs_type)


    # table_initializer = tf.lookup.KeyValueTensorInitializer(keys=list(self.label_map.keys()),
    #                                                             values=list(self.label_map.values()),
    #                                                             key_dtype=tf.string,
    #                                                             value_dtype=tf.int32)
    # label_to_id = tf.lookup.StaticHashTable(table_initializer, default_value=-1)
    # label_id = label_to_id.lookup(parsed_label[self.label_field_name])
    #

    # encoding_obs_type = table.lookup(vocabulary[obs_type])

    return tf.squeeze(parsed_features['obs_type'] == obs_type)
    # return tf.squeeze(parsed_features['tce_period_norm'] >= 0)
    # return tf.squeeze(tf.equal(parsed_features['obs_type'], encoding_obs_type))
    # return tf.reduce_any(tf.equal(parsed_features['obs_type'], encoding_obs_type))


def filter_examples_tfrecord_label(parsed_features, label_id, label):
    """ Filters out examples whose label is not `label`.

    Args:
        parsed_features: tf tensor, parsed features for example
        label_id: tf tensor, label id for example
        label: str, label to be used as filter

    Returns: tf boolean tensor

    """

    return tf.squeeze(parsed_features['label'] == label)


class ComputePerformanceOnFFIand2min(keras.callbacks.Callback):
    """
    Custom Keras Callback to evaluate model at the end of each epoch separately on 2-min and FFI data.
    """

    def __init__(self, config, save_dir, filter_fn=None):
        """ Callback constructor.

        Args:
            config: dict, configuration parameters for the run.
            save_dir: Path, save directory for the results.
            filter_fn: function, filter function used to filter parsed examples from the TFRecord datasets.
        """

        super().__init__()
        self.config = config
        self.datasets = list(self.config['datasets_fps'].keys())
        self.obs_types = ['ffi', '2min']
        self.filter_fn = filter_fn
        self.save_dir = save_dir

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Evaluate model at the end of the epoch.

        Args:
            epoch: int, epoch number
            logs: dict, logs returned by keras.callbacks.Callback

        Returns:

        """

        res = {obs_type: {} for obs_type in self.obs_types}

        for obs_type in self.obs_types:  # iterate on observation type (i.e., ffi and 2min)
            for dataset in self.datasets:  # iterate on datasets (i.e., train, val, test)
                eval_input_fn = InputFn(file_paths=self.config['datasets_fps'][dataset],
                                        batch_size=self.config['evaluation']['batch_size'],
                                        mode='EVAL',
                                        label_map=self.config['label_map'],
                                        features_set=self.config['features_set'],
                                        multiclass=self.config['config']['multi_class'],
                                        use_transformer=self.config['config']['use_transformer'],
                                        feature_map=self.config['feature_map'],
                                        label_field_name=self.config['label_field_name'],
                                        filter_fn=partial(filter_examples_tfrecord_obs_type, obs_type=obs_type),
                                        )

                # evaluate model in the given dataset
                res_eval = self.model.evaluate(x=eval_input_fn(),
                                               y=None,
                                               batch_size=None,
                                               verbose=0,
                                               sample_weight=None,
                                               steps=None,
                                               callbacks=None
                                               )

                # add evaluated dataset metrics to result dictionary
                if len(res_eval) == 0:  # no examples for that category; set to NaN
                    for metric_name_i, metric_name in enumerate(self.model.metrics_names):
                        res[obs_type][f'{dataset}_{metric_name}'] = np.nan
                else:
                    for metric_name_i, metric_name in enumerate(self.model.metrics_names):
                        res[obs_type][f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

        np.save(self.save_dir / f'res_eval_epoch_{epoch}.npy', res)


class ComputePerformanceAfterFilteringLabel(keras.callbacks.Callback):
    """
    Custom Keras Callback to evaluate model at the end of each epoch separately on each label.
    """

    def __init__(self, config, save_dir, filter_fn=None):
        """ Callback constructor.

        Args:
            config: dict, configuration parameters for the run.
            save_dir: Path, save directory for the results.
            filter_fn: function, filter function used to filter parsed examples from the TFRecord datasets.
        """

        super().__init__()
        self.config = config
        self.datasets = list(self.config['datasets_fps'].keys())
        self.labels = ['KP', 'CP', 'FP', 'EB', 'NEB', 'NPC', 'NTP', 'BD']
        self.filter_fn = filter_fn
        self.save_dir = save_dir

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Evaluate model at the end of the epoch.

        Args:
            epoch: int, epoch number
            logs: dict, logs returned by keras.callbacks.Callback

        Returns:

        """

        res = {label: {} for label in self.labels}

        for label in self.labels:  # iterate on observation type (i.e., ffi and 2min)
            for dataset in self.datasets:  # iterate on datasets (i.e., train, val, test)
                eval_input_fn = InputFn(file_paths=self.config['datasets_fps'][dataset],
                                        batch_size=self.config['evaluation']['batch_size'],
                                        mode='EVAL',
                                        label_map=self.config['label_map'],
                                        features_set=self.config['features_set'],
                                        multiclass=self.config['config']['multi_class'],
                                        feature_map=self.config['feature_map'],
                                        label_field_name=self.config['label_field_name'],
                                        filter_fn=partial(filter_examples_tfrecord_label, label=label),
                                        )

                # evaluate model in the given dataset
                res_eval = self.model.evaluate(x=eval_input_fn(),
                                               y=None,
                                               batch_size=None,
                                               verbose=0,
                                               )

                # add evaluated dataset metrics to result dictionary
                if len(res_eval) == 0:  # no examples for that category; set to NaN
                    for metric_name_i, metric_name in enumerate(self.model.metrics_names):
                        res[label][f'{dataset}_{metric_name}'] = np.nan
                else:
                    for metric_name_i, metric_name in enumerate(self.model.metrics_names):
                        res[label][f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

        np.save(self.save_dir / f'res_eval_epoch_{epoch}.npy', res)

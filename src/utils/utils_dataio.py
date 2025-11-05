""" Utility functions for data I/O for TFRecords and model ingestion. """

# 3rd party
import os
import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
import numpy as np
# import tensorflow_probability as tfp
import traceback
import pandas as pd

# local
from src.train.data_augmentation import phase_shift, phase_inversion  # , add_whitegaussiannoise
from src_preprocessing.tf_util.example_util import get_feature


def set_tf_data_type_for_features(features_set):
    """ Set TF data types for features in the feature set.

    Args:
        features_set: dict, each key is the name of a feature that maps to a dictionary with keys 'dim' and 'dtype'.
        'dim' is a list that describes the dimensionality of the feature and 'dtype' the data type of the feature.
        'dtype' should be a string (either 'float' - mapped to tf.float32; or 'int' - mapped to tf.int64).

    Returns:
        features_set: the data type is now a TensorFlow data type

    """

    # choose features set
    for feature_name, feature in features_set.items():
        if feature['dtype'] == 'float':
            features_set[feature_name]['dtype'] = tf.float32
        if feature['dtype'] == 'int':
            features_set[feature_name]['dtype'] = tf.int64
        if feature['dtype'] == 'string':
            features_set[feature_name]['dtype'] = tf.string

    return features_set


# @tf.function(jit_compile=True)
def prepare_augment_example_online(serialized_example, online_preproc_params):
    """ Prepare parameters for online data augmentation to apply consistently across multiple features of the example.

        Args:
            serialized_example: tf serialized example, example features in TFRecord dataset
            online_preproc_params: dict, parameters for online data augmentation

        Returns: tuple of online augmentation parameters
    """

    # Randomly reverse time series features with probability reverse_time_series_prob.
    should_reverse = tf.less(x=tf.random.uniform([], minval=0, maxval=1), y=0.5, name="should_reverse")

    # bin shifting
    bin_shift = [-5, 5]
    shift = tf.random.uniform(shape=(), minval=bin_shift[0], maxval=bin_shift[1],
                              dtype=tf.dtypes.int32, name='randuniform')

    # get oot indices for Gaussian noise augmentation added to oot indices
    tce_ephem = tf.io.parse_single_example(serialized=serialized_example,
                                           features={'tce_period': tf.io.FixedLenFeature([], tf.float32),
                                                     'tce_duration': tf.io.FixedLenFeature([], tf.float32)})

    # boolean tensor for oot indices for global view
    idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(online_preproc_params['num_bins_global'],
                                                                tce_ephem['tce_duration'],
                                                                tce_ephem['tce_period'])
    # boolean tensor for oot indices for local view
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(online_preproc_params['num_bins_local'],
                                                              online_preproc_params['num_transit_dur'])

    return should_reverse, shift, idxs_nontransitcadences_glob, idxs_nontransitcadences_loc


# @tf.function(jit_compile=True)
def augment_example_online(feature_example, should_reverse, shift):
    """ Perform online augmentation across multiple features of the example.

        Args:
            feature_example: NumPy array, deserialized feature for example in TFRecord dataset
            should_reverse: bool, if True, feature is reversed on x-axis
            shift: int, how many timesteps to shift
            # idxs_nontransitcadences_glob: NumPy array, non-transit timesteps in global view
            # idxs_nontransitcadences_loc: NumPy array, non-transit timesteps in local view

        Returns: feature_example after being transformed for augmentation
    """

    # # add white gaussian noise
    # if 'global' in feature_name:
    #     oot_values = tf.boolean_mask(feature_example, idxs_nontransitcadences_glob)
    # else:
    #     oot_values = tf.boolean_mask(feature_example, idxs_nontransitcadences_loc)
    #
    # # oot_median = tf.math.reduce_mean(oot_values, axis=0, name='oot_mean')
    # oot_median = tfp.stats.percentile(oot_values, 50, axis=0, name='oot_median')
    # # oot_values_sorted = tf.sort(oot_values, axis=0, direction='ASCENDING', name='oot_sorted')
    # # oot_median = tf.slice(oot_values_sorted, oot_values_sorted.shape[0] // 2, (1,),
    # #                       name='oot_median')

    # oot_std = tf.math.reduce_std(oot_values, axis=0, name='oot_std')

    # value = add_whitegaussiannoise(value, oot_median, oot_std)

    # value = add_whitegaussiannoise(value, parsed_features[feature_name + '_meanoot'],
    #                                parsed_features[feature_name + '_stdoot'])

    # invert phase
    feature_example = phase_inversion(feature_example, should_reverse)

    # phase shift some bins
    feature_example = phase_shift(feature_example, shift)

    return feature_example

def get_data_from_tfrecords_for_predictions_table(datasets, data_fields, datasets_fps):
    """ Get data of the `data_fields` in the TFRecord files for different data sets defined in `datasets` and create a
    pandas DataFrame with those data for each data set.

    Args:
        datasets: list, data sets (e.g., 'train', 'val', 'test')
        data_fields: list, fields to extract data for from the TFRecord files
        datasets_fps: dict, each key is a data set (e.g., 'train', 'val', 'test') that maps to a list of TFRecord
        file paths

    Returns:
        - dataset_tbls, dict of pandas DataFrames for each data set in `datasets` containing as columns the
        data in `data_fields` from the TFRecord files.
    """

    # instantiate dictionary to get data from the TFRecords to be displayed in the table with predictions
    dataset_tbls = {dataset: {field: [] for field in data_fields} for dataset in datasets}
    for dataset in datasets:  # iterate over the data sets
        for tfrec_fp in datasets_fps[dataset]:  # iterate over the TFRecord files
            data_aux = get_data_from_tfrecord(tfrec_fp, data_fields)
            for field in data_aux:
                dataset_tbls[dataset][field].extend(data_aux[field])

    dataset_tbls = {dataset: pd.DataFrame(dataset_tbl) for dataset, dataset_tbl in dataset_tbls.items()}

    return dataset_tbls


class InputFnv2(object):
    """Class that acts as a callable input function."""

    def __init__(self, file_paths, batch_size, mode, label_map, features_set, data_augmentation=False,
                 online_preproc_params=None, category_weights=None, sample_weights=False,
                 multiclass=False, shuffle_buffer_size=None, shuffle_seed=24, prefetch_buffer_nsamples=256,
                 feature_map=None, label_field_name='label', filter_fn=None, cache_enabled=False,
                 tfrecord_read_buffer_size=64):
        """Initializes the input function.

        :param file_paths: str, File pattern matching input TFRecord files, e.g. "/tmp/train-?????-of-00100". May also
            be a comma-separated list of file patterns; or list, list of file paths.
        :param batch_size: int, batch size
        :param mode: A tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        :param label_map: dict, map between class name and integer value
        :param features_set: dict of the features to be extracted from the dataset, the key is the feature name and the
            value is a dict with the dimension 'dim' of the feature and its data type 'dtype'
            (can the dimension and data type be inferred from the tensors in the dataset?)
        :param data_augmentation: bool, if True data augmentation is performed
        :param online_preproc_params: dict, contains data used for preprocessing examples online for data augmentation
        :param category_weights: dict, each key/val pair represents the weight for any sample associated with the
            given category as key
        :param sample_weights: bool, if True uses sample weights provided in the TFRecords for the examples in the
            training set.
        :param multiclass: bool, if True maps label ids to  one-hot encoding
        :param shuffle_buffer_size: int, size of the buffer used for shuffling. Buffer size equal or larger than the
            training dataset size guarantees perfect shuffling. If None, it will set buffer size equal to the training
            set size. Beware of out-of-memory issues caused by too large buffer size.
        :param shuffle_seed: int, shuffle seed
        :param prefetch_buffer_nsamples: int, number of samples which when divided by the batch size gives the number of
            batches prefetched
        :feature_map: dict, mapping of feature name to a new name
        :label_field_name: str, name for label stored in the TFRecord files
        :filter_fn: function, used to filter data in the TFRecord files
        :cache_enabled: bool, if True the dataset is cached (dataset needs to fit in memory)
        :tfrecord_read_buffer_size: int, size of read buffer for TFRecord dataset in MB

        :return:
        """

        self.file_paths = file_paths

        self.mode = mode
        self.batch_size = tf.constant(batch_size, dtype=tf.int64)  # batch_size
        self.label_map = label_map
        self.n_classes = len(np.unique(list(label_map.values())))
        self.features_set = features_set

        self.data_augmentation = data_augmentation and self.mode == 'TRAIN'
        self.online_preproc_params = online_preproc_params
        self.label_field_name = label_field_name

        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.prefetch_buffer_size = max(1, int(prefetch_buffer_nsamples / batch_size))

        self.filter_fn = filter_fn

        self.cache_enabled = cache_enabled
        self.tfrecord_read_buffer_size = tfrecord_read_buffer_size

        self.category_weights = category_weights
        self.sample_weights = sample_weights

        self.multiclass = multiclass

        self.feature_map = feature_map

        # build the lookup table once during init
        self.label_to_id = self._build_label_to_id_table(label_map)

        self.label_field = {self.label_field_name: tf.io.FixedLenFeature([], tf.string)}

        if category_weights is not None and self.mode == 'TRAIN':
            self.label_to_weight = self._build_label_to_weight(category_weights)
        else:
            self.label_to_weight = None

        if sample_weights and self.mode == 'TRAIN':
            self.sample_weight_field = {'sample_weight': tf.io.FixedLenFeature([], tf.float32)}
        else:
            self.sample_weight_field = None

        # get features names, shapes and data types to be extracted from the TFRecords
        self.data_fields = self._prepare_data_fields()

        self.include_labels = self.mode in ['TRAIN', 'EVAL']

        if feature_map is not None:
            self.feature_map_table = self._build_feature_map_table()
        else:
            self.feature_map_table = None

    def _build_feature_map_table(self):
        """ Builds a feature map table.

        :return:
            a static hash table that maps feature names to new names
        """

        keys = tf.constant(list(self.feature_map.keys()), dtype=tf.string)
        values = tf.constant(list(self.feature_map.values()), dtype=tf.string)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
        feature_map_table = tf.lookup.StaticHashTable(initializer,
                                                      default_value=tf.constant("", dtype=tf.string))

        return feature_map_table

    def _prepare_data_fields(self):
        """ Builds data fields dictionary based on requested features set dimensions and data types.

            Args:

            Returns: dictionary of data fields with expected dimensions and data types
        """

        data_fields = {}
        for feature_name, feature_info in self.features_set.items():
            if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:  # N-D feature, N > 1
                if 'unfolded' in feature_name:  # tensor features
                    data_fields[feature_name] = tf.io.FixedLenFeature(1, tf.string)
                else:  # N-D features stored as flattened arrays
                    data_fields[feature_name] = tf.io.FixedLenFeature([np.prod(feature_info['dim'])], tf.float32)
            else:
                data_fields[feature_name] = tf.io.FixedLenFeature(feature_info['dim'], feature_info['dtype'])

        return data_fields

    @staticmethod
    def _build_label_to_id_table(label_map):
        """ Builds hash table that maps label to label ID.

            Args:
                label_map: dict, mapping from label to label id
            Returns: TF lookup static hash table for labels to label IDs

        """

        with tf.init_scope():
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=list(label_map.keys()),
                values=list(label_map.values()),
                key_dtype=tf.string,
                value_dtype=tf.float32  # tf.int32
            )
            return tf.lookup.StaticHashTable(initializer, default_value=-1)

    @staticmethod
    def _build_label_to_weight(category_weights):
        """ Builds hash table that maps label to label ID.

            Args:
                category_weights: dict, mapping from label to label weight

            Returns: TF lookup static hash table for labels to weights
        """

        category_weight_table_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(category_weights.keys()),
            values=list(category_weights.values()),
            key_dtype=tf.string,
            value_dtype=tf.float32)

        label_to_weight = tf.lookup.StaticHashTable(category_weight_table_initializer, default_value=1)

        return label_to_weight

    def __call__(self):
        """ Builds the input pipeline.

        :return:
            a tf.data.Dataset with features and labels
        """

        # @tf.function(jit_compile=True)
        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors.

            :param serialized_example: a single tf.Example
            :return:
                tuple, feature and label tensors
            """

            # parse the features
            parsed_features_ex = tf.io.parse_single_example(serialized=serialized_example, features=self.data_fields)
            
            # parse tensors from strings to arrays with the correct shape
            def parse_and_reshape(feature_name):
                
                feature_info = self.features_set[feature_name]
                
                feature_value = parsed_features_ex[feature_name]
                
                if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:
                    if 'unfolded' in feature_name:  # for features stored as Tensors
                        tensor = tf.io.parse_tensor(feature_value[0], out_type=feature_info['dtype'])
                        feature_value = tf.reshape(tensor, feature_info['dim'])
                    else:  # reshape N-D features into their original dimensions from a flattened array
                        feature_value = tf.reshape(feature_value, feature_info['dim'])
                
        
                tf.debugging.check_numerics(feature_value, message=f"NaN or Inf in feature: {feature_name}")

                return feature_value

            feature_names = list(parsed_features_ex.keys())
            parsed_features = {name: parse_and_reshape(name) for name in feature_names}

            # get labels if in TRAIN or EVAL mode
            if self.include_labels:
                parsed_label = tf.io.parse_single_example(serialized=serialized_example, features=self.label_field)

            # set example weight
            if self.category_weights is not None and self.mode == 'TRAIN' and self.include_labels:
                example_weight = self.label_to_weight.lookup(parsed_label[self.label_field_name])
            elif self.sample_weights and self.mode == 'TRAIN':
                example_weight = tf.io.parse_single_example(serialized=serialized_example,
                                                            features=self.sample_weight_field)
            else:
                example_weight = None

            # map label to label id
            if self.include_labels:

                # map label to integer
                label_id = self.label_to_id.lookup(parsed_label[self.label_field_name])
                
                tf.debugging.check_numerics(label_id, message="NaN or Inf in label")

                # tf.debugging.assert_greater_equal(label_id, 0, message="Invalid label")
                tf.debugging.assert_greater_equal(label_id, tf.constant(0.0, dtype=tf.float32), message="Invalid label")

                if self.multiclass:
                    label_id = tf.cast(label_id, tf.int32)
                    label_id = tf.one_hot(label_id, self.n_classes)

            # prepare data augmentation
            if self.data_augmentation:
                should_reverse, shift, idxs_nontransitcadences_glob, idxs_nontransitcadences_loc = (
                    prepare_augment_example_online(serialized_example, self.online_preproc_params))

            # initialize feature output
            output = {}
            for feature_name, feature_value in parsed_features.items():

                feature_name_tensor = tf.constant(feature_name)

                # data augmentation for flux time series features
                if self.data_augmentation:
                    should_augment = tf.strings.regex_full_match(feature_name_tensor, ".*view.*")
                    feature_value = tf.cond(
                        should_augment,
                        lambda: augment_example_online(feature_value, should_reverse, shift),
                        lambda: feature_value
                    )

                # # update feature name
                if self.feature_map is not None:
                    remapped_name = self.feature_map_table.lookup(feature_name_tensor)
                    use_original = tf.equal(remapped_name, "")
                    final_name = tf.cond(use_original, lambda: feature_name_tensor, lambda: remapped_name)
                    output[final_name.numpy().decode()] = feature_value
                else:
                    output[feature_name] = feature_value

            # return output for example
            if self.include_labels:
                if example_weight is not None:
                    return output, label_id, example_weight
                else:
                    return output, label_id
            else:
                return output

        if isinstance(self.file_paths, str):
            file_patterns = self.file_paths.split(",")
            filenames = []
            for p in file_patterns:
                matches = tf.io.gfile.glob(p)
                if not matches:
                    raise ValueError("Found no input files matching {}".format(p))
                filenames.extend(matches)
        else:
            filenames = [str(fp) for fp in self.file_paths]

        # create filename dataset based on the list of tfrecords filepaths
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

        # shuffle the TFRecord files
        if self.mode == 'TRAIN':
            filename_dataset = filename_dataset.shuffle(buffer_size=len(filenames), seed=self.shuffle_seed)

        # map a TFRecordDataset object to each tfrecord filepath
        # dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
        # interleave TFRecord files for parallel loading
        dataset = filename_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=self.tfrecord_read_buffer_size * 1024 * 1024),  # x MB buffer size
            cycle_length=tf.data.AUTOTUNE if self.mode != 'PREDICT' else 1,
            num_parallel_calls=tf.data.AUTOTUNE if self.mode != 'PREDICT' else 1
        )

        # shuffle the examples in the dataset if training
        if self.mode == 'TRAIN':
            # if 'eval_with_2mindata_transferlearning' in filenames[0]:
            #     self.shuffle_buffer_size = 1000
            # else:
            #     if not self.shuffle_buffer_size:
            #         self.shuffle_buffer_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
            #     else:
            #         self.shuffle_buffer_size = dataset.cardinality()
            dataset = dataset.shuffle(self.shuffle_buffer_size, seed=self.shuffle_seed)
            # dataset = dataset.shuffle(30000, seed=self.shuffle_seed)
            # dataset = dataset.shuffle(dataset.cardinality(), seed=self.shuffle_seed)

        # map the example parser across the tfrecords dataset to extract the examples and manipulate them
        # (e.g., real-time data augmentation, shuffling, ...)
        # dataset = dataset.map(_example_parser, num_parallel_calls=4)
        # number of parallel calls is set dynamically based on available CPU; it defines number of parallel calls to
        # process asynchronously
        # dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        if self.mode == 'PREDICT' else False)

        if self.filter_fn:  # use filter function to filter parsed examples in the TFRecord dataset
            dataset = dataset.filter(self.filter_fn)

        # creates batches by combining consecutive elements
        dataset = dataset.batch(self.batch_size,
                                deterministic=True if self.mode == 'PREDICT' else False,
                                num_parallel_calls=tf.data.AUTOTUNE)

        # prefetches batches determined by the buffer size chosen
        # parallelized processing in the CPU with model computations in the GPU
        # dataset = dataset.prefetch(max(1, self.prefetch_buffer_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def get_ce_weights(label_map, tfrec_dir, datasets=['train'], label_fieldname='label', verbose=False):
    """ Compute class cross-entropy weights based on the amount of labels for each class.

    :param label_map: dict, map between class name and integer value
    :param tfrec_dir: str, filepath to directory with the tfrecords
    :param datasets: list, datasets used to compute the CE weights
    :param label_fieldname: str, name of the label field in the TFRecords
    :param verbose: bool
    :return:
        ce_weights: list, weight for each class (class 0, class 1, ...)
        TODO: ce_weights should be a dictionary
    """

    # get labels ids
    label_ids = label_map.values()

    # assumes TFRecords filenames (train, val, test)-xxxxx
    # only gets TFRecords filenames for the given datasets
    filenames = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir) if file.split('-')[0] in datasets]

    # instantiate list of labels
    label_vec = []

    # iterate through the TFRecords files
    for file in filenames:

        tfrecord_dataset = tf.data.TFRecordDataset(file)
        try:
            for string_record in tfrecord_dataset.as_numpy_iterator():  # parse the label

                example = tf.train.Example()
                example.ParseFromString(string_record)
                label = example.features.feature[label_fieldname].bytes_list.value[0].decode("utf-8")
                label_vec.append(label_map[label])

        except tf.errors.DataLossError as err:
            print("Data Loss Error: " + str(err))

    # count instances for each class
    label_counts = [label_vec.count(label_id) for label_id in label_ids]

    # give more weight to classes with fewer instances
    ce_weights = [max(label_counts) / max(count_i, 1e-7) for count_i in label_counts]

    if verbose:
        for dataset in datasets:
            print('Number of examples for dataset {}: {}'.format(dataset, label_counts))
        print('CE weights: {}'.format(ce_weights))

    return ce_weights


def get_data_from_tfrecord(tfrecord, data_fields):
    """ Extract data from a tfrecord file.

    :param tfrecord: str, tfrecord filepath
    :param data_fields: list of data fields to be extracted from the tfrecords.

    :return:
        data: dict, each key value pair is a list of values for a specific data field

    """

    # initialize data dict
    data = {field: [] for field in data_fields}

    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecord))
    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        # extracting data fields
        datum = {}

        for field in data_fields:

            try:
                extracted_feature = get_feature(example, field)[0]

                if isinstance(extracted_feature, bytes):  # convert bytes to strings
                    extracted_feature = extracted_feature.decode("utf-8")
                datum[field] = extracted_feature

            except Exception as e:
                print(traceback.format_exc())
                print(e)
                raise TypeError(f'Field not found: {field}')

        # add example
        for field in data_fields:
            data[field].append(datum[field])

    return data


def get_data_from_tfrecords(tfrecords, data_fields):
    """ Extract data from a set of tfrecord files.

    :param tfrecords: list of tfrecords filepaths.
    :param data_fields: list of data fields to be extracted from the tfrecords.

    :return:
        data: dict, each key value pair is a list of values for a specific data field
    """

    data = {field: [] for field in data_fields}

    for tfrecord in tfrecords:
        data_aux = get_data_from_tfrecord(tfrecord, data_fields)
        for field in data_aux:
            data[field].extend(data_aux[field])

    return data


# @tf.function(jit_compile=False)
def get_out_of_transit_idxs_loc(num_bins_loc, num_transit_durations):
    """ Get boolean mask of out-of-transit cadences for the local views.

    :param num_bins_loc: int, number of bins for the local views
    :param num_transit_durations: int, number of transit durations in the local views
    :return:
        idxs_nontransitcadences_loc: tf 1D boolean tensor with out-of-transit indices set to True and in-transit
        indices set to False
    """

    # get out-of-transit indices for local views
    transit_duration_bins_loc = num_bins_loc / num_transit_durations  # number of bins per transit duration

    # get left and right in-transit indices
    left_lim = tf.cast(tf.math.round((num_bins_loc - transit_duration_bins_loc) / 2), tf.int32)
    right_lim = tf.cast(tf.math.round((num_bins_loc + transit_duration_bins_loc) / 2), tf.int32)

    idx_range = tf.range(0, num_bins_loc)

    idxs_nontransitcadences_loc = tf.math.logical_or(tf.math.less_equal(idx_range, left_lim),
                                                     tf.math.greater_equal(idx_range, right_lim))

    return idxs_nontransitcadences_loc


# @tf.function(jit_compile=False)
def get_out_of_transit_idxs_glob(num_bins_glob, transit_duration, orbital_period):
    """ Get boolean mask of out-of-transit cadences for the global views.

    :param num_bins_glob: int, number of bins for the global views
    :param transit_duration: tf scalar float tensor, transit duration
    :param orbital_period: tf scalar float tensor, orbital period
    :return:
        idxs_nontransitcadences_glob: tf 1D boolean tensor with out-of-transit indices set to True and in-transit
        indices set to False
    """

    # get out-of-transit indices for global views
    frac_durper = transit_duration / orbital_period  # ratio transit duration to orbital period

    # get left and right in-transit indices
    left_lim = tf.cast(tf.math.round(num_bins_glob / 2 * (1 - frac_durper)), tf.int32)
    right_lim = tf.cast(tf.math.round(num_bins_glob / 2 * (1 + frac_durper)), tf.int32)

    idx_range = tf.range(0, num_bins_glob)

    idxs_nontransitcadences_glob = tf.math.logical_or(tf.math.less_equal(idx_range, left_lim),
                                                      tf.math.greater_equal(idx_range, right_lim))

    return idxs_nontransitcadences_glob


if __name__ == '__main__':

    tfrec_fp = '/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159/tfrecords/eval/shard-0001'
    data_fields_lst = ['label']
    tfrec_data = get_data_from_tfrecord(tfrec_fp, data_fields_lst)

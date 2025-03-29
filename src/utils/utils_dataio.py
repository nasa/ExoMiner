""" Utility functions for data I/O for TFRecords and model ingestion. """

# 3rd party
import os
import tensorflow as tf
import numpy as np
# import tensorflow_probability as tfp
import traceback
import pandas as pd

# local
from src.train.data_augmentation import phase_shift, phase_inversion, add_whitegaussiannoise
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
                 feature_map=None, label_field_name='label', filter_fn=None):
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
        :feature_map: dict, mapping of label to label id
        :label_field_name: str, name for label stored in the TFRecord files
        :filter_fn: function, used to filter data in the TFRecord files

        :return:
        """

        self.file_paths = file_paths
        self.mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        self.n_classes = len(np.unique(list(label_map.values())))
        self.features_set = features_set
        self.data_augmentation = data_augmentation and self.mode == 'TRAIN'
        self.online_preproc_params = online_preproc_params
        self.label_field_name = label_field_name

        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.prefetch_buffer_size = int(prefetch_buffer_nsamples / self.batch_size)

        self.filter_fn = filter_fn

        self.category_weights = category_weights
        self.sample_weights = sample_weights
        self.multiclass = multiclass

        self.feature_map = feature_map if feature_map is not None else {}

    def __call__(self):
        """ Builds the input pipeline.

        :return:
            a tf.data.Dataset with features and labels
        """

        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors.

            :param serialized_example: a single tf.Example
            :return:
                tuple, feature and label tensors
            """

            # get features names, shapes and data types to be extracted from the TFRecords
            data_fields = {}
            for feature_name, feature_info in self.features_set.items():
                if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:  # N-D feature, N > 1
                    data_fields[feature_name] = tf.io.FixedLenFeature(1, tf.string)
                else:
                    data_fields[feature_name] = tf.io.FixedLenFeature(feature_info['dim'], feature_info['dtype'])

            # parse the features
            parsed_features = tf.io.parse_single_example(serialized=serialized_example, features=data_fields)

            # get labels if in TRAIN or EVAL mode
            if include_labels:
                label_field = {self.label_field_name: tf.io.FixedLenFeature([], tf.string)}
                parsed_label = tf.io.parse_single_example(serialized=serialized_example, features=label_field)

            if self.category_weights is not None and self.mode == 'TRAIN':
                category_weight_table_initializer = tf.lookup.KeyValueTensorInitializer(
                    keys=list(self.category_weights.keys()),
                    values=list(self.category_weights.values()),
                    key_dtype=tf.string,
                    value_dtype=tf.float32)

                label_to_weight = tf.lookup.StaticHashTable(category_weight_table_initializer, default_value=1)
                example_weight = label_to_weight.lookup(parsed_label[self.label_field_name])
            elif self.sample_weights and self.mode == 'TRAIN':
                sample_weight_field = {'sample_weight': tf.io.FixedLenFeature([], tf.float32)}
                example_weight = tf.io.parse_single_example(serialized=serialized_example, features=sample_weight_field)

            # prepare data augmentation
            if self.data_augmentation:
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
                idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(self.online_preproc_params['num_bins_global'],
                                                                            tce_ephem['tce_duration'],
                                                                            tce_ephem['tce_period'])
                # boolean tensor for oot indices for local view
                idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(self.online_preproc_params['num_bins_local'],
                                                                          self.online_preproc_params['num_transit_dur'])

            # map label to integer value
            label_id = tf.cast(0, dtype=tf.int32, name='cast_label_to_int32')
            if include_labels:
                # map label to integer
                label_id = label_to_id.lookup(parsed_label[self.label_field_name])

                # Ensure that the label_id is non negative to verify a successful hash map lookup.
                assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.cast(0, dtype=tf.int32)),
                                               ["Unknown label string:", parsed_label[self.label_field_name]],
                                               name='assert_non-negativity')

                with tf.control_dependencies([assert_known_label]):
                    label_id = tf.identity(label_id)

            if self.multiclass:
                label_id = tf.one_hot(label_id, self.n_classes)

            # initialize feature output
            output = {}
            for feature_name, value in parsed_features.items():

                feature_info = self.features_set[feature_name]
                if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:  # parse tensors
                    value = tf.io.parse_tensor(serialized=value[0], out_type=self.features_set[feature_name]['dtype'])
                    value = tf.reshape(value, self.features_set[feature_name]['dim'])

                # data augmentation for time series features
                if 'view' in feature_name and self.data_augmentation:

                    # with tf.variable_scope('input_data/data_augmentation'):

                    # add white gaussian noise
                    if 'global' in feature_name:
                        oot_values = tf.boolean_mask(value, idxs_nontransitcadences_glob)
                    else:
                        oot_values = tf.boolean_mask(value, idxs_nontransitcadences_loc)

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
                    value = phase_inversion(value, should_reverse)

                    # phase shift some bins
                    value = phase_shift(value, shift)

                if feature_name in list(self.feature_map.keys()):
                    output[self.feature_map[feature_name]] = value
                else:
                    output[feature_name] = value

            # FIXME: should it return just output when in PREDICT mode? Would have to change predict.py yielding part
            if (self.category_weights is not None or self.sample_weights) and self.mode == 'TRAIN':
                return output, label_id, example_weight
            else:
                return output, label_id

        # Create a HashTable mapping label strings to integer ids.
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=list(self.label_map.keys()),
                                                                values=list(self.label_map.values()),
                                                                key_dtype=tf.string,
                                                                value_dtype=tf.int32)

        label_to_id = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

        include_labels = self.mode in ['TRAIN', 'EVAL']

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
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

        # shuffle the examples in the dataset if training
        if self.mode == 'TRAIN':
            if 'eval_with_2mindata_transferlearning' in filenames[0]:
                self.shuffle_buffer_size = 1000
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
        # dataset = dataset.batch(self.batch_size)
        dataset = dataset.batch(self.batch_size, deterministic=True if self.mode == 'PREDICT' else False,
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

    # iterate throught the TFRecords files
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

    # give more weight to classes with less instances
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


def get_data_from_tfrecords(tfrecords, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a set of tfrecord files.

    :param tfrecords: list of tfrecords filepaths.
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field
    """

    data = {field: [] for field in data_fields}

    if filt is not None:
        data['selected_idxs'] = []

    for tfrecord in tfrecords:
        data_aux = get_data_from_tfrecord(tfrecord, data_fields, label_map=label_map, filt=filt, coupled=coupled)
        for field in data_aux:
            data[field].extend(data_aux[field])

    return data


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
    data_fields = ['label']
    tfrec_data = get_data_from_tfrecord(tfrec_fp, data_fields)

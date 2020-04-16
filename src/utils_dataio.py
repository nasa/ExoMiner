"""
Utilty functions for data I/O.
"""

# 3rd party
import os
import tensorflow as tf
import numpy as np
import tempfile

# local
from src.utils_train import phase_shift, phase_inversion, add_whitegaussiannoise


class InputFn(object):
    """Class that acts as a callable input function for the Estimator."""

    def __init__(self, file_pattern, batch_size, mode, label_map, features_set, data_augmentation=False,
                 filter_data=None, scalar_params_idxs=None):
        """Initializes the input function.

        :param file_pattern: File pattern matching input TFRecord files, e.g. "/tmp/train-?????-of-00100".
        May also be a comma-separated list of file patterns.
        :param batch_size: int, batch size
        :param mode: A tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
        :param label_map: dict, map between class name and integer value
        :param features_set: dict of the features to be extracted from the dataset, the key is the feature name and the
        value is a dict with the dimension 'dim' of the feature and its data type 'dtype'
        (can the dimension and data type be inferred from the tensors in the dataset?)
        :param data_augmentation: bool, if True data augmentation is performed
        :param filter_data:
        :param scalar_params_idxs: list, indexes of features to extract from the scalar features Tensor
        :return:
        """

        self._file_pattern = file_pattern
        self._mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        self.features_set = features_set
        self.data_augmentation = data_augmentation and self._mode == tf.estimator.ModeKeys.TRAIN
        self.scalar_params_idxs = scalar_params_idxs

        self.filter_data = filter_data

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
            data_fields = {feature_name: tf.io.FixedLenFeature(feature_info['dim'], feature_info['dtype'])
                           for feature_name, feature_info in self.features_set.items()}

            # # get auxiliary data from TFRecords required to perform data augmentation
            # if self.data_augmentation:
            #     for feature_name in self.features_set:
            #         if 'view' in feature_name:
            #             data_fields[feature_name + '_rmsoot'] = tf.io.FixedLenFeature([], tf.float32)

            # get labels if in TRAIN or EVAL mode
            # FIXME: change the feature name to 'label' - standardization of TCE feature names across different
            #  TFRecords (Kepler/TESS) sources - remember that for backward compatibility we need to keep
            #  av_training_set
            if include_labels:
                # data_fields['av_training_set'] = tf.io.FixedLenFeature([], tf.string)
                data_fields['label'] = tf.io.FixedLenFeature([], tf.string)

            # # initialize filtering data fields
            # if self.filter_data is not None:
            #     data_fields['kepid'] = tf.io.FixedLenFeature([], tf.int64)
            #     data_fields['tce_plnt_num'] = tf.io.FixedLenFeature([], tf.int64)

            # Parse the features.
            parsed_features = tf.io.parse_single_example(serialized=serialized_example, features=data_fields)

            # prepare data augmentation
            if self.data_augmentation:
                # Randomly reverse time series features with probability reverse_time_series_prob.
                should_reverse = tf.less(x=tf.random.uniform([], minval=0, maxval=1), y=0.5, name="should_reverse")
                bin_shift = [-5, 5]
                shift = tf.random.uniform(shape=(), minval=bin_shift[0], maxval=bin_shift[1],
                                          dtype=tf.dtypes.int32, name='randuniform')

            # initialize feature output
            output = {}
            # if self.filter_data is not None:
            #     output['filt_features'] = {}

            label_id = tf.cast(0, dtype=tf.int32, name='cast_label_to_int32')

            for feature_name, value in parsed_features.items():

                if 'oot' in feature_name:
                    continue
                # label
                # FIXME: change the feature name to 'label' - standardization of TCE feature names across different
                #  TFRecords (Kepler/TESS) sources - remember that for backward compatibility we need to keep
                #  av_training_set
                elif include_labels and feature_name == 'label':  # either 'label' or 'av_training_set'

                    # map label to integer
                    label_id = label_to_id.lookup(value)

                    # Ensure that the label_id is non negative to verify a successful hash map lookup.
                    assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.cast(0, dtype=tf.int32)),
                                                   ["Unknown label string:", value], name='assert_non-negativity')

                    with tf.control_dependencies([assert_known_label]):
                        label_id = tf.identity(label_id)

                # # filtering features
                # elif self.filter_data is not None and feature_name == 'kepid':
                #     output['filt_features']['kepid'] = value
                # elif self.filter_data is not None and feature_name == 'tce_plnt_num':
                #     output['filt_features']['tce_n'] = value

                # scalar features (e.g, stellar, TCE, transit fit parameters)
                elif feature_name == 'scalar_params':
                    if self.scalar_params_idxs is None:
                        output['scalar_params'] = value
                    else:  # choose only some of the scalar_params based on their indexes
                        output['scalar_params'] = tf.gather(value, indices=self.scalar_params_idxs, axis=0)

                # time-series features
                else:  # input_config.features[feature_name].is_time_series:

                    # data augmentation
                    if self.data_augmentation:

                        # with tf.variable_scope('input_data/data_augmentation'):

                        # invert phase
                        value = phase_inversion(value, should_reverse)

                        # phase shift some bins
                        value = phase_shift(value, shift)

                        # add white gaussian noise
                        value = add_whitegaussiannoise(value, parsed_features[feature_name + '_meanoot'],
                                                       parsed_features[feature_name + '_stdoot'])

                    output[feature_name] = value

            # FIXME: should it return just output when in PREDICT mode? Would have to change predict.py yielding part
            return output, label_id

        def filt_func(x, y):
            """ Utility function used to filter examples from the dataset based on their target ID + TCE planet number

            :param x: feature tensor
            :param y: label tensor
            :return:
                boolean tensor, True for valid examples, False otherwise
            """

            z1 = tf.as_string(x['filt_features']['target_id'])
            z_ = tf.convert_to_tensor('_')
            z2 = tf.as_string(x['filt_features']['tce_plnt_num'])
            zj = tf.strings.join([z1, z_, z2])

            return tf.math.reduce_any(tf.math.equal(zj,
                                                    tf.convert_to_tensor(self.filter_data['target_id+tce_plnt_num'])))

        def get_features_and_labels(x, y):
            """ Utility function used to remove the features used to filter the dataset.

            :param x: feature tensor
            :param y: label tensor
            :return:
                tuple, dict with features tensor, and label tensor
            """

            return {'time_series_features': x['time_series_features']}, y

        # with tf.variable_scope('input_data'):

        # Create a HashTable mapping label strings to integer ids.
        table_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(self.label_map.keys()),
            values=list(self.label_map.values()),
            key_dtype=tf.string,
            value_dtype=tf.int32)

        label_to_id = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

        include_labels = self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

        file_patterns = self._file_pattern.split(",")
        filenames = []
        for p in file_patterns:
            matches = tf.io.gfile.glob(p)
            if not matches:
                raise ValueError("Found no input files matching {}".format(p))
            filenames.extend(matches)

        # tf.logging.info("Building input pipeline from %d files matching patterns: %s", len(filenames), file_patterns)

        # create filename dataset based on the list of tfrecords filepaths
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

        # map a TFRecordDataset object to each tfrecord filepath
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

        # shuffle the dataset if training or evaluating
        # FIXME: for perfect sampling, the buffer_size should be larger than the size of the dataset. Can we handle it?
        #        set variables for buffer size and shuffle seed?
        # if self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(27000, seed=None)

        # do not repeat the dataset
        dataset = dataset.repeat(1)

        # map the example parser across the tfrecords dataset to extract the examples and manipulate them
        # (e.g., real-time data augmentation, shuffling, ...)
        # dataset = dataset.map(_example_parser, num_parallel_calls=4)
        # number of parallel calls is set dynamically based on available CPU; it defines number of parallel calls to
        # process asynchronously
        dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # filter the dataset based on the filtering features
        if self.filter_data is not None:
            dataset = dataset.filter(filt_func)

            # remove the filtering features from the dataset
            dataset = dataset.map(get_features_and_labels)

        # creates batches by combining consecutive elements
        dataset = dataset.batch(self.batch_size)

        # prefetches batches determined by the buffer size chosen
        # parallelized processing in the CPU with model computations in the GPU
        dataset = dataset.prefetch(max(1, int(256 / self.batch_size)))

        return dataset


def get_model_dir(path):
    """Returns a randomly named, non-existing model file folder.

    :param path: str, root directory for saved model directories
    :return:
        model_dir_custom: str, model directory
    """

    def _gen_dir():
        return os.path.join(path, tempfile.mkdtemp().split('/')[-1])

    model_dir_custom = _gen_dir()

    while os.path.isdir(model_dir_custom):  # try until there is no current directory with that name
        model_dir_custom = _gen_dir()

    return model_dir_custom


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


def get_num_samples(label_map, tfrec_dir, datasets, label_fieldname='label'):
    """ Compute number of samples in the datasets for each class.

    :param label_map: dict, map between class name and integer value
    :param tfrec_dir: str, filepath to directory with the tfrecords
    :param datasets: list, datasets to be counted. It follows that convention that the tfrecords have in their name
    train, val, or test if they contain examples pertaining to the training, validation or test sets, respectively.
    :return:
        n_samples: int, number of samples in the datasets for each class
    """

    n_samples = {dataset: {label: 0 for label in label_map.values()} for dataset in datasets}

    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir) if
                 any([dataset in file for dataset in datasets])]

    for file in filenames:

        file_dataset = file.split('/')[-1]
        curr_dataset = datasets[np.where([dataset in file_dataset for dataset in datasets])[0][0]]

        tfrecord_dataset = tf.data.TFRecordDataset(file)
        try:
            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)
                try:
                    label = example.features.feature[label_fieldname].bytes_list.value[0].decode("utf-8")
                except ValueError as e:
                    print('No label field found on the example. Ignoring it.')
                    print('Error output:', e)
                    continue

                n_samples[curr_dataset][label_map[label]] += 1

        except tf.errors.DataLossError as err:
            print("{}".format(err))

    return n_samples


def get_data_from_tfrecord(tfrecord, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a tfrecord file.

    :param tfrecord: str, tfrecord filepath
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field

    # TODO: add lookup table
    #       deal with errors
    """

    # valid fields and features in the TFRecords
    # main fields: ['target_id', 'tce_plnt_num', 'label']
    FIELDS = ['tce_period', 'tce_duration', 'tce_time0bk', 'mes', 'ra', 'dec', 'mag', 'mag_uncert', 'tce_time0bk_err',
              'tce_period_err', 'tce_duration_err', 'transit_depth', 'transit_depth_err', 'sectors']
    TIMESERIES = ['global_view', 'local_view', 'local_view_centr', 'global_view_centr', 'global_view_odd',
                  'local_view_odd']

    if filt is not None:
        union_fields = np.union1d(data_fields, list(filt.keys()))  # get fields that are in both
        if 'target_id+tce_plnt_num' in list(filt.keys()):  # add target_id and tce_plnt_num
            union_fields = np.concatenate((union_fields, ['target_id', 'tce_plnt_num']))
    else:
        union_fields = data_fields

    # initialize data dict
    data = {field: [] for field in union_fields}

    if filt is not None:
        data['selected_idxs'] = []

    # record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=tfrecord)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord)
    try:
        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            if filt is not None:
                data['selected_idxs'].append(False)

            # extracting data fields
            datum = {}
            if 'label' in union_fields:
                label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

                if label_map is not None:  # map from label to respective integer
                    datum['label'] = label_map[label]

            if 'original_label' in union_fields:
                datum['original_label'] = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

            if 'target_id' in union_fields:
                datum['target_id'] = example.features.feature['target_id'].int64_list.value[0]

            if 'oi' in union_fields:
                datum['oi'] = example.features.feature['oi'].int64_list.value[0]

            if 'tce_plnt_num' in union_fields:
                datum['tce_plnt_num'] = example.features.feature['tce_plnt_num'].int64_list.value[0]

            if 'sectors' in union_fields:  # for TESS data
                datum['sectors'] = example.features.feature['sectors'].bytes_list.value[0].decode("utf-8")

            # float parameters
            for field in FIELDS:
                if field in union_fields:
                    datum[field] = example.features.feature[field].float_list.value[0]

            # time-series features
            for timeseries in TIMESERIES:
                if timeseries in union_fields:
                    datum[timeseries] = example.features.feature[timeseries].float_list.value

            # filtering
            if filt is not None:

                if 'label' in filt.keys() and datum['label'] not in filt['label'].values:
                    continue

                if 'original_label' in filt.keys() and datum['original_label'] not in filt['original_label'].values:
                    continue

                if coupled:
                    if 'target_id+tce_plnt_num' in filt.keys() and \
                            '{}_{}'.format(datum['target_id'], datum['tce_plnt_num']) not in filt['target_id+tce_plnt']:
                        continue
                else:
                    if 'target_id' in filt.keys() and datum['target_id'] not in filt['target_id']:
                        continue
                    if 'tce_plnt_num' in filt.keys() and datum['tce_plnt_num'] not in filt['tce_plnt_num']:
                        continue

                if 'tce_period' in filt.keys() and \
                        not filt['tce_period'][0] <= datum['tce_period'] <= filt['tce_period'][1]:
                    continue

                if 'tce_duration' in filt.keys() and \
                        not filt['tce_duration'][0] <= datum['tce_duration'] <= filt['tce_duration'][1]:
                    continue

                if 'tce_time0bk' in filt.keys() and \
                        not filt['tce_time0bk'][0] <= datum['tce_time0bk'] <= filt['tce_time0bk'][1]:
                    continue

                if 'mes' in filt.keys() and not filt['mes'][0] <= datum['mes'] <= filt['mes'][1]:
                    continue

                data['selected_idxs'][-1] = True

            # add example
            for field in data_fields:
                data[field].append(datum[field])

    except:
        print('Corrupted TFRecord: {}'.format(tfrecord))

    return data


def get_data_from_tfrecord_kepler(tfrecord, data_fields, label_map=None, filt=None, coupled=False):
    """ Extract data from a tfrecord file.

    :param tfrecord: str, tfrecord filepath
    :param data_fields: list of data fields to be extracted from the tfrecords.
    :param label_map: dict, map between class name and integer value
    :param filt: dict, containing as keys the elements of data_fields or a subset, which are used to filter the
    examples. For 'label', 'kepid' and 'tce_n' the values should be a list; for the other data_fields, it should be a
    two element list that defines the interval of acceptable values
    :param coupled: bool, if True filter examples based on their KeplerID + TCE number (joint)
    :return:
        data: dict, each key value pair is a list of values for a specific data field

    # TODO: add lookup table
    #       right now, it assumes that all examples have the same fields
    #       deal with errors
    """

    EPHEMERIS = {'tce_period': 'tce_period', 'tce_duration': 'tce_duration', 'epoch': 'tce_time0bk', 'MES': 'mes'}
    TIMESERIES = ['global_view', 'local_view', 'local_view_centr', 'global_view_centr', 'global_view_odd',
                  'local_view_odd']

    if filt is not None:
        union_fields = np.union1d(data_fields, list(filt.keys()))
        if 'kepid+tce_n' in list(filt.keys()):
            union_fields = np.concatenate((union_fields, ['kepid', 'tce_n']))
    else:
        union_fields = data_fields

    data = {field: [] for field in data_fields}

    if filt is not None:
        data['selected_idxs'] = []

    # record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=tfrecord)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord)
    try:
        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            if filt is not None:
                data['selected_idxs'].append(False)

            # extracting data fields
            datum = {}
            if 'label' in union_fields:
                datum['label'] = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
                if label_map is not None:
                    datum['label'] = label_map[datum['label']]
                    if 'original label' in union_fields:
                        datum['original label'] = example.features.feature['av_training_set'].bytes_list.value[0].\
                            decode("utf-8")

            if 'kepid' in union_fields:
                datum['kepid'] = example.features.feature['kepid'].int64_list.value[0]

            if 'tce_n' in union_fields:
                datum['tce_n'] = example.features.feature['tce_plnt_num'].int64_list.value[0]

            # ephemeris data
            for ephem in EPHEMERIS:
                if ephem in union_fields:
                    datum[ephem] = example.features.feature[EPHEMERIS[ephem]].float_list.value[0]

            # time series
            for timeseries in TIMESERIES:
                if timeseries in union_fields:
                    datum[timeseries] = example.features.feature[timeseries].float_list.value

            # filtering
            if filt is not None:

                if 'label' in filt.keys() and datum['label'] not in filt['label'].values:
                    continue

                if 'original label' in filt.keys() and datum['original label'] not in filt['original label'].values:
                    continue

                if coupled:
                    if 'kepid+tce_n' in filt.keys() and '{}_{}'.format(datum['kepid'], datum['tce_n']) \
                            not in filt['kepid+tce_n']:
                        continue
                else:
                    if 'kepid' in filt.keys() and datum['kepid'] not in filt['kepid']:
                        continue
                    if 'tce_n' in filt.keys() and datum['tce_n'] not in filt['tce_n']:
                        continue

                if 'tce_period' in filt.keys() and not filt['tce_period'][0] <= datum['tce_period'] <= filt['tce_period'][1]:
                    continue

                if 'tce_duration' in filt.keys() and not filt['tce_duration'][0] <= datum['tce_duration'] <= \
                                                         filt['tce_duration'][1]:
                    continue

                if 'epoch' in filt.keys() and not filt['epoch'][0] <= datum['epoch'] <= filt['epoch'][1]:
                    continue

                if 'MES' in filt.keys() and not filt['MES'][0] <= datum['MES'] <= filt['MES'][1]:
                    continue

                data['selected_idxs'][-1] = True

            # add example
            for field in data_fields:
                data[field].append(datum[field])

    except Exception as e:
        print('Error while reading TFRecords: ', e)
        # print('Corrupted TFRecord: {}'.format(tfrecord))
        # print('Or nonnexistent fields were selected')

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


def create_filtered_tfrecord(src_tfrecord, save_dir, filt, append_name='', kw_filt_args=None):
    """ Create filtered tfrecord from a source tfrecord file.

    :param: src_tfrecord:
    :param save_dir: str, directory in which the new tfrecords are saved
    :param filt: dict, containing as keys the fields used to filter the examples. For 'label', 'kepid' and 'tce_n' the
    values should be a list; for the other data_fields, it should be a two element list that defines the interval of
    acceptable values
    :param append_name: str, string added to the name of the new filtered tfrecord
    :param kw_filt_args: dict, keyword parameters for function get_data_from_tfrecord
    :return:
    """

    if kw_filt_args is None:
        kw_filt_args = {}

    # get boolean indexes for the examples in the tfrecord file
    filt_idx = get_data_from_tfrecord(src_tfrecord, [], label_map=None, filt=filt, **kw_filt_args)['selected_idxs']

    # tfrecord destination filepath
    dest_tfrecord = save_dir + src_tfrecord.split['/'][-1] + append_name

    # write new tfrecord
    with tf.python.python_io.TFRecordWriter(dest_tfrecord) as writer:
        # create iterator for source tfrecord
        record_iterator = tf.python.python_io.tf_record_iterator(path=src_tfrecord)
        # go over the examples in the source tfrecord
        for i, string_record in enumerate(record_iterator):
            if not filt_idx[i]:  # filter out examples
                continue

            # add example to the new tfrecord
            example = tf.train.Example()
            example.ParseFromString(string_record)
            if example is not None:
                writer.write(example.SerializeToString())


if __name__ == '__main__':

    # check features, labels and other parameters stored in the TFRecords
    tfrec_dir = ''
    datasets = ['train', 'val', 'test']
    features_names = ['global_view', 'local_view', 'global_view_centr', 'local_view_centr',
                                 'local_view_even', 'local_view_odd', 'scalar_params']
    tfrec_files = [file for file in os.listdir(tfrec_dir) if file.split('-')[0] in datasets]
    for file in tfrec_files:
        record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(tfrec_dir, file))
        for i, string_record in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            # label_map['kepler'][False][example.features.feature['label'].bytes_list.value[0].decode('utf-8')]
            for feature_name in features_names:
                value = example.features.feature[feature_name].float_list.value
                if not np.all(np.isfinite(value)):  # or value <= 0:
                    print(example.features.feature['target_id'])

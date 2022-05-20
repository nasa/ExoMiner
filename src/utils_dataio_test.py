import tensorflow as tf
import tempfile
import os


class InputFnCV(object):
    """Class that acts as a callable input function for the Estimator."""

    def __init__(self, file_pattern, batch_size, mode, label_map, features_set, data_augmentation=False,
                 data_idxs=None, scalar_params_idxs=None, shuffle_buffer_size=27000, shuffle_seed=24,
                 prefetch_buffer_nsamples=256):
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
        :param data_idxs: indices of examples to get from the TFRecords
        :param scalar_params_idxs: list, indexes of features to extract from the scalar features Tensor
        :param shuffle_buffer_size: int, size of the buffer used for shuffling. Buffer size equal or larger than dataset
        size guarantees perfect shuffling
        :param shuffle_seed: int, shuffle seed
        :param prefetch_buffer_nsamples: int, number of samples which when divided by the batch size gives the number of
        batches prefetched
        :return:
        """

        self._file_pattern = file_pattern
        self._mode = mode
        self.batch_size = batch_size
        self.label_map = label_map
        self.features_set = features_set
        self.data_augmentation = data_augmentation and self._mode == tf.estimator.ModeKeys.TRAIN
        self.scalar_params_idxs = scalar_params_idxs

        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed

        self.data_idxs = data_idxs
        self.prefetch_buffer_size = int(prefetch_buffer_nsamples / self.batch_size)

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

            if self.data_idxs is not None:
                data_fields['idx'] = tf.io.FixedLenFeature([], tf.int64)

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

                # example index
                elif feature_name == 'idx':
                    output[feature_name] = value

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

        def filtidx_func(x, y):
            """ Utility function used to filter examples from the dataset based on their indices.

            :param x: feature tensor
            :param y: label tensor
            :return:
                boolean tensor, True for valid examples, False otherwise
            """

            return tf.math.reduce_any(tf.math.equal(x['idx'], tf.convert_to_tensor(self.data_idxs)))

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

        # shuffle the dataset if training
        # FIXME: for perfect sampling, the buffer_size should be larger than the size of the dataset. Can we handle it?
        #        set variables for buffer size and shuffle seed?
        # if self._mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(self.shuffle_buffer_size, seed=self.shuffle_seed)

        # do not repeat the dataset
        dataset = dataset.repeat(1)

        # map the example parser across the tfrecords dataset to extract the examples and manipulate them
        # (e.g., real-time data augmentation, shuffling, ...)
        # dataset = dataset.map(_example_parser, num_parallel_calls=4)
        # number of parallel calls is set dynamically based on available CPU; it defines number of parallel calls to
        # process asynchronously
        dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # filter the dataset based on the filtering features
        if self.data_idxs is not None:
            dataset = dataset.filter(filtidx_func)

        # creates batches by combining consecutive elements
        dataset = dataset.batch(self.batch_size)

        # prefetches batches determined by the buffer size chosen
        # parallelized processing in the CPU with model computations in the GPU
        dataset = dataset.prefetch(max(1, self.prefetch_buffer_size))

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

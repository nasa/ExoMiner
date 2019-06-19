# 3rd party
import os
import tensorflow as tf

# local
import paths
import src.config


def _example_parser(serialized_example):
    """Parses a single tf.Example into feature and label tensors."""

    feature_size_dict = {'global_view': 2001, 'local_view': 201}

    if centr_flag:
        feature_size_dict = {**feature_size_dict, 'global_view_centr': 2001, 'local_view_centr': 201}

    data_fields = {feature_name: tf.FixedLenFeature([length], tf.float32)
                   for feature_name, length in feature_size_dict.items()}
    if include_labels:
        data_fields['av_training_set'] = tf.FixedLenFeature([], tf.string)

    # Parse the features.
    parsed_features = tf.parse_single_example(serialized_example, features=data_fields)

    output = {'time_series_features': {}}
    label_id = tf.to_int32(0)
    for feature_name, value in parsed_features.items():
        if include_labels and feature_name == 'av_training_set':
            label_id = label_to_id.lookup(value)
            # Ensure that the label_id is non negative to verify a successful hash map lookup.
            assert_known_label = tf.Assert(tf.greater_equal(label_id, tf.to_int32(0)),
                                           ["Unknown label string:", value])
            with tf.control_dependencies([assert_known_label]):
                label_id = tf.identity(label_id)

        else:  # input_config.features[feature_name].is_time_series:

            output['time_series_features'][feature_name] = value

    # return output
    return output, label_id


def get_numpy_dataset(tfrec_filenames, centr_flag):

    tfrec_dict = {'features': {'global_view': [], 'local_view': []},
                  'kepids': [], 'labels': []}

    if centr_flag:
        tfrec_dict['features'] = {'global_view_centr': [], 'local_view_centr': [], **tfrec_dict['features']}

    for file in tfrec_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            kepid = example.features.feature['kepid'].int64_list.value[0]
            tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
            tfrec_dict['labels'] += example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")

            tfrec_dict['kepids'] += [(kepid, tce_n)]

            for timeseries_id in tfrec_dict['features'].keys():
                tfrec_dict['features'][timeseries_id] += [example.features.feature[timeseries_id].float_list.value]

    return tfrec_dict


satellite = 'kepler'
multi_class = False

centr_flag = False
include_labels = True

tfrec_dir = paths.tfrec_dir
dataset = 'test'

tfrec_filenames = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir)]
data_dict = get_numpy_dataset(tfrec_filenames, centr_flag)

# file_pattern = data_dir + '/' + dataset + '*'
# file_patterns = file_pattern.split(",")
# filenames = []
# for p in file_patterns:
#     matches = tf.gfile.Glob(p)
#     if not matches:
#         raise ValueError("Found no input files matching {}".format(p))
#     filenames.extend(matches)
#
# label_map = src.config.label_map[satellite][multi_class]
# table_initializer = tf.contrib.lookup.KeyValueTensorInitializer(
#     keys=list(label_map.keys()),
#     values=list(label_map.values()),
#     key_dtype=tf.string,
#     value_dtype=tf.int32)
#
# label_to_id = tf.contrib.lookup.HashTable(table_initializer, default_value=-1)
#
# filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
# dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
# # dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.repeat(1)  # WHY?
#
# dataset = dataset.map(_example_parser, num_parallel_calls=1)
#
# dataset = dataset.batch(batch_size=16)
#
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
# sess = tf.Session()
# sess.run(iterator.initializer, feed_dict=table_initializer)
# while True:
#     try:
#         # assert i == value
#         value = sess.run(next_element)
#         print(value)
#     except tf.errors.OutOfRangeError:
#         print("End of dataset")  # ==> "End of dataset"
#         break
#
#
# # dataset = dataset.prefetch(max(1, int(256 / self.batch_size)))  # Better to set at None?

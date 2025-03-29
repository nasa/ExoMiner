"""
Utility functions for data input/output for model.
"""

# 3rd party
import tensorflow as tf


def create_tf_data_dataset(tfrec_fps, features_dict, label_map, label_name):
    """ Create a TFRecord Dataset from a list of TFRecord files.

    :param tfrec_fps: list of Path objects for the TFRecord files
    :param features_dict: dict, maps input feature names to be extracted from the TFRecord files to a dictionary with
    the data type 'dtype' and shape 'dim'.
    :param label_map: dict, maps label to an integer 'label_id'
    :param label_name: str, name of the label field to be mapped to a 'label_id'

    :return: parsed examples from TFRecord dataset
    """

    # # create a HashTable mapping label strings to integer ids.
    # table_initializer = tf.lookup.KeyValueTensorInitializer(keys=list(label_map.keys()),
    #                                                         values=list(label_map.values()),
    #                                                         key_dtype=tf.string,
    #                                                         value_dtype=tf.int32
    #                                                         )

    # label_to_id = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

    serialized_examples = tf.data.TFRecordDataset(filenames=[str(fp) for fp in tfrec_fps])  # , compression_type="GZIP")

    def parse_tf_example(serialized_example):
        """ Parse a binary serialized tf.Example. """

        parsed_example = tf.io.parse_single_example(
            serialized_example,
            {feature_name: tf.io.FixedLenFeature([], dtype=feature['dtype'])
            # { feature_name: tf.io.FixedLenFeature(feature['dim'], dtype=feature['dtype'])
        for feature_name, feature in features_dict.items()
            }
        )

        # parsed_example['label_id'] = label_to_id.lookup(parsed_example[label_name])

        return parsed_example

    return serialized_examples.map(parse_tf_example)

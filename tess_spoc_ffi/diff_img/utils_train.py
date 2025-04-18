"""
Utility functions used during training.
- Custom callbacks, ...
"""

# 3rd party
import tensorflow as tf


def filter_examples_tfrecord_unknown_location(parsed_features, label_id):
    """ Filters out examples based on `label_id` being -1.

    Args:
        parsed_features: tf tensor, parsed features for example
        label_id: tf tensor, label id for example

    Returns: tf boolean tensor

    """

    # valid_labels = ['CP', 'KP', 'EB', 'NEB', 'NPC']

    # print(f'Label ID: {label_id}')
    # print(f'Label: {parsed_features["label"]}')

    return label_id != 2

    # return tf.squeeze(tf.reduce_any(tf.equal(parsed_features['label'], valid_labels)))

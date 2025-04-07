import tensorflow as tf


def filter_examples_tfrecord_tce_model_snr(parsed_features, snr_threshold: int = 20):
    """Filters out examples whose tce_model_snr is below a threshold.

    Args:
        parsed_features: tf tensor, parsed features for example
        snr_threshold: float | int, value to be used as filter

    Returns: tf boolean tensor
    """

    return tf.squeeze(parsed_features["tce_model_snr"] >= snr_threshold)

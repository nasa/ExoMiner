import tensorflow as tf
from tensorflow.python.keras import backend as K

def mean_per_class_accuracy(labels, predictions):

    num_classes = len(predictions.shape)

    value, update_op = tf.compat.v1.metrics.mean_per_class_accuracy(labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          name='mean_per_class_accuracy')

    # find all variables created for this metric
    metric_vars = [i for i in tf.compat.v1.local_variables() if 'mean_per_class_accuracy' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def auc_roc(labels, predictions, num_thresholds=1000, summation_method='careful_interpolation'):

    value, update_op = tf.compat.v1.metrics.auc(labels, predictions,
                                      num_thresholds=num_thresholds,
                                      summation_method=summation_method,
                                      curve='ROC',
                                      name='auc_roc')

    # find all variables created for this metric
    metric_vars = [i for i in tf.compat.v1.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def auc_pr(labels, predictions, num_thresholds=1000, summation_method='careful_interpolation'):

    value, update_op = tf.compat.v1.metrics.auc(labels, predictions,
                                      num_thresholds=num_thresholds,
                                      summation_method=summation_method,
                                      curve='PR',
                                      name='auc_pr')

    # find all variables created for this metric
    metric_vars = [i for i in tf.compat.v1.local_variables() if 'auc_pr' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

    # K._get_session().run(tf.compat.v1.local_variables_initializer())
    # tf.keras.backend._compat.v1.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())
    # with tf.control_dependencies([update_op]):
    #     value = tf.identity(value)
    # return value

    # return update_op


def create_metrics(model):
    """Builds TensorFlow operations to compute model evaluation metrics.

    Args:
        model
    # labels: Tensor with shape [batch_size].
    # predictions: Tensor with shape [batch_size, output_dim].
    # weights: Tensor with shape [batch_size].
    # batch_losses: Tensor with shape [batch_size].
    # output_dim: Dimension of model output

    Returns:
    A dictionary {metric_name: (metric_value, update_op).
    """

    # initialize dictionary to save metrics
    metrics = {}

    if model.output_size == 1:  # 1 output

        assert model.predictions.shape[1] == 1

        # removes dimensions of size 1 from the shape of a tensor.
        predictions = tf.squeeze(model.predictions, axis=[1], name='squeezed_predictions')

        # thresholding scores at 0.5
        predicted_labels = tf.cast(tf.greater(predictions, 0.5, name='thresholding'), name="predicted_labels",
                                   dtype=tf.int32)

    else:  # 2 or more outputs

        # num_samples x num_classes, 2 dimensions
        assert len(model.predictions.shape) == 2

        predictions = model.predictions

        # index with the largest score across the output (class) axis
        # num_samples x 1
        predicted_labels = tf.argmax(model.predictions, axis=1, name="predicted_labels", output_type=tf.int32)

        # TODO: implement one-hot encoding
        # one_hot_labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)

    labels = model.labels

    num_classes = max(model.config['label_map'].values()) + 1

    # compute metrics

    # across-class accuracy
    metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predicted_labels, name='accuracy')
    metrics['mean_per_class_accuracy'] = tf.metrics.mean_per_class_accuracy(labels=labels,
                                                                            predictions=predicted_labels,
                                                                            num_classes=num_classes,
                                                                            name='mean_per_class_accuracy')

    # TODO: implement these metrics for multiclass classification
    if model.output_size == 1:  # metrics that can only be computed for binary classification

        metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predicted_labels, name='precision')
        metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predicted_labels, name='recall')

        # define the number of thresholds used
        metrics['prec thr'] = tf.metrics.precision_at_thresholds(labels, predictions,
                                                                 np.linspace(0, 1, num=1000, endpoint=True,
                                                                             dtype='float32'),
                                                                 name='precision_at_thresholds')
        metrics['rec thr'] = tf.metrics.recall_at_thresholds(labels, predictions,
                                                             np.linspace(0, 1, num=1000, endpoint=True,
                                                                         dtype='float32'),
                                                             name='recall_at_thresholds')

        metrics["roc auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                            summation_method='careful_interpolation', curve='ROC', name='roc_auc')
        metrics["pr auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
                                           summation_method='careful_interpolation', curve='PR', name='pr_auc')

        # TODO: make mean avg precision work
        # metrics['avg prec'] = tf.metrics.average_precision_at_k(labels, predictions,
        #                                                         labels.get_shape().as_list()[1])

    # auxiliary functions for computing confusion matrix
    def _metric_variable(name, shape, dtype):
        """ Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
        return tf.get_variable(
            name,
            initializer=tf.zeros(shape, dtype),
            trainable=False,
            collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])

    def _count_condition(name, labels_value, predicted_value):
        """ Creates a counter for given values of predictions and labels. """
        count = _metric_variable(name, [], tf.float32)
        is_equal = tf.cast(tf.logical_and(tf.equal(labels, labels_value),
                                          tf.equal(predicted_labels, predicted_value)), dtype=tf.float32)
        update_op = tf.assign_add(count, tf.reduce_sum(tf.ones_like(model.labels, dtype=tf.float32) * is_equal))
        return count.read_value(), update_op

    # confusion matrix metrics
    num_labels = 2 if not model.config['multi_class'] else num_classes
    for label in range(num_labels):
        for pred_label in range(num_labels):
            metric_name = "label_{}_pred_{}".format(label, pred_label)
            metrics[metric_name] = _count_condition(metric_name, labels_value=label, predicted_value=pred_label)

    return metrics
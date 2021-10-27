# import tensorflow as tf
# from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np
# from tensorflow.compat.v1.metrics import precision_at_k


def get_metrics(clf_threshold=0.5, num_thresholds=1000):
    """ Setup metrics to be monitored.

    :param clf_threshold: float, classification threshold (between 0 and 1)
    :param num_thresholds: int, number of thresholds to be tested
    :return:
        metrics_list: list, metrics to be monitored
    """

    threshold_range = list(np.linspace(0, 1, num=num_thresholds))

    auc_pr = keras.metrics.AUC(num_thresholds=num_thresholds,
                               summation_method='interpolation',
                               curve='PR',
                               name='auc_pr')
    auc_roc = keras.metrics.AUC(num_thresholds=num_thresholds,
                                summation_method='interpolation',
                                curve='ROC',
                                name='auc_roc')

    binary_acc = keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=clf_threshold)
    precision = keras.metrics.Precision(thresholds=clf_threshold, name='precision')
    recall = keras.metrics.Recall(thresholds=clf_threshold, name='recall')

    precision_thr = keras.metrics.Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
    recall_thr = keras.metrics.Recall(thresholds=threshold_range, top_k=None, name='rec_thr')

    tp = keras.metrics.TruePositives(name='tp', thresholds=threshold_range)
    fp = keras.metrics.FalsePositives(name='fp', thresholds=threshold_range)
    tn = keras.metrics.TrueNegatives(name='tn', thresholds=threshold_range)
    fn = keras.metrics.FalseNegatives(name='fn', thresholds=threshold_range)

    metrics_list = [binary_acc, precision, recall, precision_thr, recall_thr, auc_pr, auc_roc, tp, fp, tn, fn]

    return metrics_list

def get_metrics_multiclass():
    """ Setup metrics to be monitored for multiclass. 
    
    :return:
        metrics_list: list, metrics to be monitored
    """

    acc = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    metrics_list = [acc]

    return metrics_list
    

def compute_precision_at_k(labels, k_vals):
    """ Computes precision at k.

    :param labels: NumPy array, labels sorted by score value (ascending order)
    :param k_vals: NumPy array, values at which to compute precision
    :return:
    """

    num_examples = len(labels)
    precision_at_k = {f'precision_at{k_val}': np.nan for k_val in k_vals}

    for k_val in k_vals:
        if num_examples >= k_val:
            precision_at_k[f'precision_at_{k_val}'] = np.sum(labels[-k_val:]) / k_val

    return precision_at_k

# def mean_per_class_accuracy(labels, predictions):
#
#     num_classes = len(predictions.shape)
#
#     value, update_op = tf.compat.v1.metrics.mean_per_class_accuracy(labels=labels,
#                                                           predictions=predictions,
#                                                           num_classes=num_classes,
#                                                           name='mean_per_class_accuracy')
#
#     # find all variables created for this metric
#     metric_vars = [i for i in tf.compat.v1.local_variables() if 'mean_per_class_accuracy' in i.name.split('/')[1]]
#
#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)
#
#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value
#
#
# def auc_roc(labels, predictions, num_thresholds=1000, summation_method='careful_interpolation'):
#
#     value, update_op = tf.compat.v1.metrics.auc(labels, predictions,
#                                       num_thresholds=num_thresholds,
#                                       summation_method=summation_method,
#                                       curve='ROC',
#                                       name='auc_roc')
#
#     # find all variables created for this metric
#     metric_vars = [i for i in tf.compat.v1.local_variables() if 'auc_roc' in i.name.split('/')[1]]
#
#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)
#
#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value
#
#
# def auc_pr(labels, predictions, num_thresholds=1000, summation_method='careful_interpolation'):
#
#     value, update_op = tf.compat.v1.metrics.auc(labels, predictions,
#                                       num_thresholds=num_thresholds,
#                                       summation_method=summation_method,
#                                       curve='PR',
#                                       name='auc_pr')
#
#     # find all variables created for this metric
#     metric_vars = [i for i in tf.compat.v1.local_variables() if 'auc_pr' in i.name.split('/')[1]]
#
#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)
#
#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value
#
#     # K._get_session().run(tf.compat.v1.local_variables_initializer())
#     # tf.keras.backend._compat.v1.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())
#     # with tf.control_dependencies([update_op]):
#     #     value = tf.identity(value)
#     # return value
#
#     # return update_op


# def create_metrics(model):
#     """Builds TensorFlow operations to compute model evaluation metrics.
#
#     Args:
#         model
#     # labels: Tensor with shape [batch_size].
#     # predictions: Tensor with shape [batch_size, output_dim].
#     # weights: Tensor with shape [batch_size].
#     # batch_losses: Tensor with shape [batch_size].
#     # output_dim: Dimension of model output
#
#     Returns:
#     A dictionary {metric_name: (metric_value, update_op).
#     """
#
#     # initialize dictionary to save metrics
#     metrics = {}
#
#     if model.output_size == 1:  # 1 output
#
#         assert model.predictions.shape[1] == 1
#
#         # removes dimensions of size 1 from the shape of a tensor.
#         predictions = tf.squeeze(model.predictions, axis=[1], name='squeezed_predictions')
#
#         # thresholding scores at 0.5
#         predicted_labels = tf.cast(tf.greater(predictions, 0.5, name='thresholding'), name="predicted_labels",
#                                    dtype=tf.int32)
#
#     else:  # 2 or more outputs
#
#         # num_samples x num_classes, 2 dimensions
#         assert len(model.predictions.shape) == 2
#
#         predictions = model.predictions
#
#         # index with the largest score across the output (class) axis
#         # num_samples x 1
#         predicted_labels = tf.argmax(model.predictions, axis=1, name="predicted_labels", output_type=tf.int32)
#
#         # TODO: implement one-hot encoding
#         # one_hot_labels = tf.argmax(model.labels, 1, name="true_labels", output_type=tf.int32)
#
#     labels = model.labels
#
#     num_classes = max(model.config['label_map'].values()) + 1
#
#     # compute metrics
#
#     # across-class accuracy
#     metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predicted_labels, name='accuracy')
#     metrics['mean_per_class_accuracy'] = tf.metrics.mean_per_class_accuracy(labels=labels,
#                                                                             predictions=predicted_labels,
#                                                                             num_classes=num_classes,
#                                                                             name='mean_per_class_accuracy')
#
#     # TODO: implement these metrics for multiclass classification
#     if model.output_size == 1:  # metrics that can only be computed for binary classification
#
#         metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predicted_labels, name='precision')
#         metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predicted_labels, name='recall')
#
#         # define the number of thresholds used
#         metrics['prec thr'] = tf.metrics.precision_at_thresholds(labels, predictions,
#                                                                  np.linspace(0, 1, num=1000, endpoint=True,
#                                                                              dtype='float32'),
#                                                                  name='precision_at_thresholds')
#         metrics['rec thr'] = tf.metrics.recall_at_thresholds(labels, predictions,
#                                                              np.linspace(0, 1, num=1000, endpoint=True,
#                                                                          dtype='float32'),
#                                                              name='recall_at_thresholds')
#
#         metrics["roc auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
#                                             summation_method='careful_interpolation', curve='ROC', name='roc_auc')
#         metrics["pr auc"] = tf.metrics.auc(labels, predictions, num_thresholds=1000,
#                                            summation_method='careful_interpolation', curve='PR', name='pr_auc')
#
#         # TODO: make mean avg precision work
#         # metrics['avg prec'] = tf.metrics.average_precision_at_k(labels, predictions,
#         #                                                         labels.get_shape().as_list()[1])
#
#     # auxiliary functions for computing confusion matrix
#     def _metric_variable(name, shape, dtype):
#         """ Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
#         return tf.get_variable(
#             name,
#             initializer=tf.zeros(shape, dtype),
#             trainable=False,
#             collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])
#
#     def _count_condition(name, labels_value, predicted_value):
#         """ Creates a counter for given values of predictions and labels. """
#         count = _metric_variable(name, [], tf.float32)
#         is_equal = tf.cast(tf.logical_and(tf.equal(labels, labels_value),
#                                           tf.equal(predicted_labels, predicted_value)), dtype=tf.float32)
#         update_op = tf.assign_add(count, tf.reduce_sum(tf.ones_like(model.labels, dtype=tf.float32) * is_equal))
#         return count.read_value(), update_op
#
#     # confusion matrix metrics
#     num_labels = 2 if not model.config['multi_class'] else num_classes
#     for label in range(num_labels):
#         for pred_label in range(num_labels):
#             metric_name = "label_{}_pred_{}".format(label, pred_label)
#             metrics[metric_name] = _count_condition(metric_name, labels_value=label, predicted_value=pred_label)
#
#     return metrics
#
#
# class PrecisionAtK(keras.metrics.Metric):
#
#     def __init__(self, k, clf_thr=0.5, name='precision_at_k', **kwargs):
#         super(PrecisionAtK, self).__init__(name=name, **kwargs)
#         # super(PrecisionAtK, self).__init__(precision_at_k, k=k, name=name)
#
#         self.k = k
#         self.clf_thr = clf_thr
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#         self.false_positives = self.add_weight(name='fp', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
#
#         sorted_idxs = tf.argsort(y_pred, axis=0, direction='DESCENDING')
#
#         threshold = tf.cast(self.clf_thr, y_pred.dtype)
#         k_batch = tf.math.minimum(y_true.shape[0], self.k)
#
#         y_pred = tf.cast(y_pred > threshold, tf.bool)
#
#         y_true = tf.cast(y_true, tf.bool)
#         # y_pred = tf.cast(y_pred, tf.bool)
#
#         y_true = tf.gather(y_true, sorted_idxs)
#         y_pred = tf.gather(y_pred, sorted_idxs)
#
#         k_idxs = tf.range(k_batch, delta=1, dtype=tf.int32)
#         y_true = tf.gather(y_true, k_idxs, axis=0)
#         y_pred = tf.gather(y_pred, k_idxs, axis=0)
#
#         pos_values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
#         pos_values = tf.cast(pos_values, self.dtype)
#
#         neg_values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
#         neg_values = tf.cast(neg_values, self.dtype)
#
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, self.dtype)
#
#             pos_sample_weight = tf.broadcast_to(sample_weight, pos_values.shape)
#             pos_values = tf.multiply(pos_values, pos_sample_weight)
#
#             neg_sample_weight = tf.broadcast_to(sample_weight, neg_values.shape)
#             neg_values = tf.multiply(neg_values, neg_sample_weight)
#
#         self.true_positives.assign_add(tf.reduce_sum(pos_values))
#         self.false_positives.assign_add(tf.reduce_sum(neg_values))
#
#     def result(self):
#
#         return tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)

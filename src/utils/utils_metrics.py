""" Utility script that defines metrics to be monitored by the models during training, evaluation and prediciton. """

from tensorflow import keras
import numpy as np


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


def get_metrics_multiclass(label_map):
    """ Setup metrics to be monitored for multiclass. 

    :param label_map: dict, map from label to label id
    :return:
        metrics_list: list, metrics to be monitored
    """

    metrics_list = []

    # acc = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    acc = keras.metrics.CategoricalAccuracy(name='accuracy')

    metrics_list.append(acc)

    for label, label_id in label_map.items():
        metrics_list.append(keras.metrics.Recall(name=f'recall_{label}', class_id=label_id, thresholds=0.5))
        metrics_list.append(keras.metrics.Precision(name=f'precision_{label}', class_id=label_id, thresholds=0.5))

    return metrics_list


def compute_precision_at_k(labels, k_vals):
    """ Computes precision at k.

    :param labels: NumPy array, labels sorted by score value (ascending order). Assumes 1 and 0 for the positive and
    negative classes, respectively
    :param k_vals: NumPy array, values at which to compute precision
    :return:
        precision_at_k: NumPy array, precision-at-k
    """

    num_examples = len(labels)
    precision_at_k = {f'precision_at_{k_val}': np.nan for k_val in k_vals}

    for k_val in k_vals:
        if num_examples >= k_val:
            precision_at_k[f'precision_at_{k_val}'] = np.sum(labels[-k_val:]) / k_val

    return precision_at_k

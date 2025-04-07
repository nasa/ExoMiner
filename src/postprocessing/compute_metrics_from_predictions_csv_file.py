"""
Compute performance metrics for a set of model scores.
"""

# 3rd party
import pandas as pd
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import numpy as np
from pathlib import Path
import tensorflow as tf


def compute_metrics_from_predictions(predictions_tbl, cats, num_thresholds, clf_threshold, top_k_vals,
                                     class_name='label_id', cat_name='label', multiclass=False,
                                     multiclass_target_score=None):
    """ Generate performance metrics for set of predictions in a csv file.

    Args:
        predictions_tbl: pandas DataFrame, csv file with predictions. The predictions file must contain the
        following columns: a `cat_name` column from where the category of the example is retrieved; a `class_name`
        columns from where the class id of the example is obtained; if binary class setting, a `score` column with the
        model prediction for the example. Otherwise, two or more columns 'score_{category_name}' that show the score for
        each category in the multiclass setting. `multiclass_target_score` defines which category is used as positive
        class to compute binary classification metrics (kind of one-vs-all setting).
        cats: dict, categories `str` of predictions. Should be a supraset of the categories in the `cat_name` column in
        the predictions csv file. Each category is a key that maps to an integer label id. This depends on the
        experiment that generated the predictions file.
        E.g.: binary classification scenario planet vs non-planet {'PC': 1, 'AFP': 0, 'NTP': 0}
        num_thresholds: int, number of thresholds to compute metrics that are defined for a set of thresholds
        clf_threshold:  float, classification threshold ([0, 1])
        top_k_vals: list, k values used to compute precision-at-top k
        class_name: str, column name for column used a ground truth integer class id
        cat_name: str, column name for column used a ground truth category
        multiclass: bool, set to True if multiclass scenario (i.e, `class_name` column has more than two unique label
        ids)
        multiclass_target_score: str, for the multiclass scenario, use one category in `cat_name` as the positive class.
        All other categories are seen as negative class.

    Returns: pandas DataFrame with compute metrics for the set of predictions in the csv file

    """

    class_ids = np.unique(list(cats.values()))  # get unique list of class ids

    # define list of metrics to be computed
    metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'avg_precision']
    metrics_lst += [f'precision_at_{k_val}' for k_val in top_k_vals]
    metrics_lst += [f'recall_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'precision_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'n_{class_id}' for class_id in class_ids]
    metrics_lst += [f'recall_{cat}' for cat in cats]
    metrics_lst += [f'n_{cat}' for cat in cats]

    data_to_tbl = {col: [] for col in metrics_lst}

    # compute predictions based on scores and classification threshold
    predictions_tbl['predicted_class'] = 0
    if multiclass:
        predictions_tbl.loc[predictions_tbl[f'score_{multiclass_target_score}' > clf_threshold], 'predicted_class'] = 1
    else:
        predictions_tbl.loc[predictions_tbl['score'] > clf_threshold, 'predicted_class'] = 1

    # compute metrics
    auc_pr = AUC(num_thresholds=num_thresholds,
                 summation_method='interpolation',
                 curve='PR',
                 name='auc_pr')
    auc_roc = AUC(num_thresholds=num_thresholds,
                  summation_method='interpolation',
                  curve='ROC',
                  name='auc_roc')

    precision = Precision(name='precision', thresholds=clf_threshold)
    recall = Recall(name='recall', thresholds=clf_threshold)

    binary_accuracy = BinaryAccuracy(name='binary_accuracy', threshold=clf_threshold)

    # if it is multi classification get specific score
    scores = predictions_tbl[multiclass_target_score].tolist() if multiclass else predictions_tbl['score'].tolist()

    # if multiclass change labels
    if multiclass:
        label_1 = cats[multiclass_target_score.split('_')[-1]]
        labels = [int(label == label_1) for label in predictions_tbl[class_name].tolist()]
    else:
        labels = predictions_tbl[class_name].tolist()

    _ = auc_pr.update_state(labels, scores)
    data_to_tbl['auc_pr'].append(auc_pr.result().numpy())

    _ = auc_roc.update_state(labels, scores)
    data_to_tbl['auc_roc'].append(auc_roc.result().numpy())

    _ = precision.update_state(labels, scores)
    data_to_tbl['precision'].append(precision.result().numpy())
    _ = recall.update_state(labels, scores)
    data_to_tbl['recall'].append(recall.result().numpy())

    _ = binary_accuracy.update_state(labels, scores)
    data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

    data_to_tbl['balanced_accuracy'].append(balanced_accuracy_score(labels,
                                                                    predictions_tbl['predicted_class']))

    data_to_tbl['avg_precision'].append(average_precision_score(labels, scores))

    for class_id in class_ids:  # computing recall per class id
        data_to_tbl[f'recall_class_{class_id}'].append(
            ((predictions_tbl[class_name] == class_id) & (predictions_tbl['predicted_class'] == class_id)).sum() /
            (predictions_tbl[class_name] == class_id).sum())
        data_to_tbl[f'n_{class_id}'].append((predictions_tbl[class_name] == class_id).sum())

    for class_id in class_ids:  # computing precision per class id
        data_to_tbl[f'precision_class_{class_id}'].append(
            ((predictions_tbl[class_name] == class_id) & (predictions_tbl['predicted_class'] == class_id)).sum() /
            (predictions_tbl['predicted_class'] == class_id).sum())

    for cat, cat_lbl in cats.items():  # computing recall per category
        data_to_tbl[f'recall_{cat}'].append(
            ((predictions_tbl[cat_name] == cat) & (predictions_tbl['predicted_class'] == cat_lbl)).sum() / (
                    predictions_tbl[cat_name] == cat).sum())
        data_to_tbl[f'n_{cat}'].append((predictions_tbl[cat_name] == cat).sum())

    for k_val in top_k_vals:  # computing precision-at-k
        precision_at_k = Precision(name=f'precision_at_{k_val}', thresholds=clf_threshold, top_k=k_val)
        if len(predictions_tbl[class_name]) >= k_val:
            _ = precision_at_k.update_state(labels, scores)
            data_to_tbl[f'precision_at_{k_val}'].append(precision_at_k.result().numpy())
        else:
            data_to_tbl[f'precision_at_{k_val}'].append(np.nan)

    metrics_df = pd.DataFrame(data_to_tbl)

    return metrics_df


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # mapping between category label and class id in predictions file
    cats = {
        # Kepler
        #  'PC': 1,
        # 'AFP': 0,
        # 'NTP': 0,
        # Kepler Sim
        # 'INJ1': 1,
        # # # 'Not INJ1': 0,
        # 'INJ2': 0,
        # 'INJ3': 0,
        # 'INV': 0,
        # 'SCR1': 0,
        # 'SCR2': 0,
        # # 'SCR3': 0,
        # TESS
        'KP': 1,
        'CP': 1,
        'EB': 0,
        # 'B': 0,
        'FP': 0,
        # 'J': 0,
        # 'FA': 0,
        'NEB': 0,
        'NPC': 0,
        'NTP': 0,
        'BD': 0,
    }
    num_thresholds = 1000
    clf_threshold = 0.5
    top_k_vals = [50, 100, 150, 200, 500, 1000, 2000, 3000]
    # top_k_vals = []
    class_name = 'label_id'
    cat_name = 'label'
    multiclass = False
    multiclass_target_score = None

    # predictions table filepath
    predictions_tbl_fp = Path(f"/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_valffionly_noduplicate2mindata_2-18-2025_1135/ranked_predictions_allfolds.csv")
    # save path
    save_fp = Path(f"/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_valffionly_noduplicate2mindata_2-18-2025_1135/metrics_2min_only.csv")

    predictions_tbl = pd.read_csv(predictions_tbl_fp)
    predictions_tbl = predictions_tbl.loc[predictions_tbl['obs_type'] == '2min']

    predictions_tbl['label_id'] = predictions_tbl.apply(lambda x: cats[x['label']], axis=1)
    metrics_df = compute_metrics_from_predictions(predictions_tbl, cats, num_thresholds, clf_threshold, top_k_vals,
                                                  class_name, cat_name)

    metrics_df.to_csv(save_fp, index=False)

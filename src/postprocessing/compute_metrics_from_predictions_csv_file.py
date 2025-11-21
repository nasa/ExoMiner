"""
Compute performance metrics for a set of model scores.
"""

# 3rd party
import pandas as pd
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy, F1Score, PrecisionAtRecall, RecallAtPrecision
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import numpy as np
from pathlib import Path
import tensorflow as tf


def map_softmax_predictions_to_class(row, pred_cols, label_map, clf_thr=0):
    
    # get the column with the highest score
    max_col = row[pred_cols].idxmax()
    
    # extract the class name from the column name (e.g., 'score_CP' -> 'CP')
    class_name = max_col.replace('score_', '')
    
    label_id = label_map.get(class_name, -1)
    
    if row[max_col] < clf_thr:
        label_id = -1
    
    return label_id


def compute_metrics_from_predictions(predictions_tbl, cats, num_thresholds, clf_threshold, top_k_vals,
                                     class_name='label_id', cat_name='label', multiclass=False,
                                     multiclass_target_score=None, recall_at_precision_thr=0.95, precision_at_recall_thr=0.95):
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
        recall_at_precision_thr: float, precision value used to compute recall
        precision_at_recall_thr: float, recall value used to compute precision

    Returns: pandas DataFrame with compute metrics for the set of predictions in the csv file

    """

    class_ids = np.unique(list(cats.values()))  # get unique list of class ids

    # define list of metrics to be computed
    metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'avg_precision', 'f1_score', 'precision_at_recall', 'recall_at_precision']
    metrics_lst += [f'precision_at_{k_val}' for k_val in top_k_vals]
    metrics_lst += [f'recall_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'precision_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'n_{class_id}' for class_id in class_ids]
    metrics_lst += [f'recall_{cat}' for cat in cats]
    metrics_lst += [f'n_{cat}' for cat in cats]

    data_to_tbl = {col: [] for col in metrics_lst}

    # compute predictions based on scores and classification threshold
    
    if multiclass:
        # predictions_tbl.loc[predictions_tbl[f'score_{multiclass_target_score}'] > clf_threshold, 'predicted_class'] = 1
        pred_columns = [f'score_{cat}' for cat in cats]
        predictions_tbl['predicted_class'] = predictions_tbl.apply(lambda row: map_softmax_predictions_to_class(row, pred_columns, cats, clf_thr=0), axis=1)
    else:
        predictions_tbl['predicted_class'] = 0
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
    
    f1_score = F1Score(name='f1_score', threshold=clf_threshold)
    
    prec_at_rec = PrecisionAtRecall(precision_at_recall_thr, name='precision_at_recall', num_thresholds=num_thresholds)
    rec_at_prec = RecallAtPrecision(recall_at_precision_thr, name='recall_at_precision', num_thresholds=num_thresholds)

    # if it is multi classification get specific score
    scores = predictions_tbl[f'score_{multiclass_target_score}'].tolist() if multiclass else predictions_tbl['score'].tolist()
    
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
    
    data_to_tbl['avg_precision'].append(average_precision_score(labels, scores))

    _ = precision.update_state(labels, scores)
    data_to_tbl['precision'].append(precision.result().numpy())
    _ = recall.update_state(labels, scores)
    data_to_tbl['recall'].append(recall.result().numpy())

    _ = binary_accuracy.update_state(labels, scores)
    data_to_tbl['accuracy'].append(binary_accuracy.result().numpy())

    _ = f1_score.update_state(np.array(labels).reshape(-1, 1), np.array(scores).reshape(-1, 1))
    data_to_tbl['f1_score'].append(f1_score.result().numpy()[0])
    
    _ = prec_at_rec.update_state(labels, scores)
    data_to_tbl['precision_at_recall'].append(prec_at_rec.result().numpy())
    
    _ = rec_at_prec.update_state(labels, scores)
    data_to_tbl['recall_at_precision'].append(rec_at_prec.result().numpy())
    
    data_to_tbl['balanced_accuracy'].append(balanced_accuracy_score(labels,
                                                                    predictions_tbl['predicted_class']))


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
    
    # add metadata
    metrics_df.attrs['clf_thr'] = clf_threshold
    metrics_df.attrs['num_thresholds'] = num_thresholds
    if multiclass:
        metrics_df.attrs['multiclass'] = multiclass
        metrics_df.attrs['multiclass_to_binary_class'] = multiclass_target_score
    metrics_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))

    return metrics_df


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # mapping between category label and class id in predictions file
    label_map = {
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
        # # TESS
        # 'KP': 1,
        # 'CP': 1,
        # 'EB': 0,
        # # 'B': 0,
        # 'FP': 0,
        # # 'J': 0,
        # # 'FA': 0,
        # 'NEB': 0,
        # 'NPC': 0,
        # 'NTP': 0,
        # 'BD': 0,
        # multiclass
        'CP': 1,
        'KP': 1,
        'EB': 0,
        'FP': 0,
        'BD': 0,
        'NTP': 0,
    }
    num_thresholds = 1000
    clf_threshold = 0.5
    top_k_vals = [50, 100, 150, 200, 500, 1000, 2000, 3000]
    # top_k_vals = []
    class_name = 'label_id'
    cat_name = 'label'
    multiclass = False  # it True, converts multiclass to binary classification by choosing positive class as `multiclass_target_score`; all other classes become negative
    multiclass_target_score = 'KP'

    # predictions table filepath
    predictions_tbl_fp = Path(f"/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_exomninerpp_11-4-2025_1353/predictions_testset_allfolds.csv")
    # save path
    save_fp = Path(f"/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_exomninerpp_11-4-2025_1353/metrics_predictions_testset_allfolds_ffi.csv")

    predictions_tbl = pd.read_csv(predictions_tbl_fp, comment='#')
    predictions_tbl = predictions_tbl.loc[predictions_tbl['obs_type'] == 'ffi']

    predictions_tbl['label_id'] = predictions_tbl.apply(lambda x: label_map[x['label']], axis=1)
    metrics_df = compute_metrics_from_predictions(predictions_tbl, label_map, num_thresholds, clf_threshold, top_k_vals,
                                                  class_name, cat_name, multiclass, multiclass_target_score)

    metrics_df.attrs['predictions_table'] = str(predictions_tbl_fp)
    metrics_df.attrs['label_map'] = label_map
    with open(save_fp, "w") as f:
        for key, value in metrics_df.attrs.items():
            f.write(f"# {key}: {value}\n")
        metrics_df.to_csv(f, index=False)
    

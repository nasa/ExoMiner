"""Use softmax predictions and threshold to compute predicted classes.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import Accuracy  # AUC, Precision, Recall, BinaryAccuracy
from sklearn.metrics import balanced_accuracy_score  # , average_precision_score
import numpy as np
from pathlib import Path
# import tensorflow as tf

#%% set paths

exp_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/test_exominer_architectures/exominer-new_samefeatmapdim-multiclass-planet-fp-ntp_tess-spoc-2min-s1-s88_10-28-2025_1554/model0/')

# get prediction CSV files
pred_fps = list(exp_dir.rglob('ranked_predictions*.csv'))
print(f'Found {len(pred_fps)} prediction files:\n{pred_fps}')

#%% compute predicted class

def map_softmax_predictions_to_class(row, pred_cols, label_map, clf_thr=0):
    
    # get the column with the highest score
    max_col = row[pred_cols].idxmax()
    
    # extract the class name from the column name (e.g., 'score_CP' -> 'CP')
    class_name = max_col.replace('score_', '')
    
    label_id = label_map.get(class_name, -1)
    
    if row[max_col] < clf_thr:
        label_id = -1
    
    return label_id

    
clf_thr = 0
label_map = {
    'CP': 1,
    'KP': 1,
    'EB': 2,
    'FP': 2,
    'BD': 2,
    'NTP': 0,
}

pred_columns = [f'score_{disp}' for disp in label_map]

new_pred_tbls = []
for pred_fp in pred_fps:
    
    print(f'Iterating through prediction file {pred_fp}')
    
    pred_df = pd.read_csv(pred_fp, comment='#')
    
    pred_df['predicted_class'] = pred_df.apply(lambda row: map_softmax_predictions_to_class(row, pred_columns, label_map, clf_thr), axis=1)
    
    pred_df['dataset'] = pred_fp.stem.split('_')[-1][:-3]
    
    new_pred_tbls.append(pred_df)

new_pred_df = pd.concat(new_pred_tbls, axis=0)

# map label to label id
new_pred_df['label_id'] = new_pred_df['label'].apply(lambda x: label_map.get(x, -1))  # -1 as default

# add metadata
new_pred_df.attrs['experiment'] = exp_dir.name
new_pred_df.attrs['label map'] =  label_map
new_pred_df.attrs['clf_thr'] = clf_thr
new_pred_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
with open(exp_dir / 'predictions_all_datasets.csv', "w") as f:
    for key, value in new_pred_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    new_pred_df.to_csv(f, index=False)

#%% compute metrics


def compute_metrics_multiclass(predictions_tbl, cats, class_name='label_id', cat_name='label'):
    
    class_ids = np.unique(list(cats.values()))  # get unique list of class ids

    # define list of metrics to be computed
    # metrics_lst = ['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'avg_precision']
    metrics_lst = ['accuracy', 'balanced_accuracy']

    metrics_lst += [f'recall_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'precision_class_{class_id}' for class_id in class_ids]
    metrics_lst += [f'n_{class_id}' for class_id in class_ids]

    metrics_lst += [f'recall_{cat}' for cat in cats]
    metrics_lst += [f'n_{cat}' for cat in cats]

    data_to_tbl = {col: [] for col in metrics_lst}

    # # compute predictions based on scores and classification threshold
    # predictions_tbl['predicted_class'] = 0
    # if multiclass:
    #     predictions_tbl.loc[predictions_tbl[f'score_{multiclass_target_score}' > clf_threshold], 'predicted_class'] = 1
    # else:
    #     predictions_tbl.loc[predictions_tbl['score'] > clf_threshold, 'predicted_class'] = 1

    # compute metrics
    # auc_pr = AUC(num_thresholds=num_thresholds,
    #              summation_method='interpolation',
    #              curve='PR',
    #              name='auc_pr')
    # auc_roc = AUC(num_thresholds=num_thresholds,
    #               summation_method='interpolation',
    #               curve='ROC',
    #               name='auc_roc')

    # precision = Precision(name='precision', thresholds=clf_threshold)
    # recall = Recall(name='recall', thresholds=clf_threshold)

    labels = predictions_tbl[class_name].tolist()
    
    accuracy = Accuracy(name='accuracy')
    data_to_tbl['accuracy'].append(accuracy.result().numpy())
    
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


    metrics_df = pd.DataFrame(data_to_tbl)
    
    return metrics_df


pred_df = pd.read_csv(exp_dir / 'predictions_all_datasets.csv', comment='#')

metrics_df_lst = []
for dataset, preds_dataset in pred_df.groupby('dataset'):
    
    print(f'Iterating on predictions for dataset {dataset}...')
    
    metrics_dataset = compute_metrics_multiclass(preds_dataset, label_map, class_name='label_id', cat_name='label')
    metrics_dataset['dataset'] = dataset

    metrics_df_lst.append(metrics_dataset)
    
metrics_df = pd.concat(metrics_df_lst, axis=0)
metrics_df.set_index('dataset', inplace=True)

# add metadata
metrics_df.attrs['experiment'] = exp_dir.name
metrics_df.attrs['label map'] =  label_map
metrics_df.attrs['clf_thr'] = clf_thr
metrics_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
with open(exp_dir / 'metrics_all_datasets_multiclass.csv', "w") as f:
    for key, value in metrics_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    metrics_df.to_csv(f, index=True)

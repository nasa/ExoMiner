"""
Script purpose:

1. Get average error scores for each label
2. Get recalls at different thresholds for each label
3. Create graph with recall and precision thresholds
3. Used to see if a model is performing better in terms of scores, or if it is more "confident,"
even if the difference cannot be seen in metrics like auc_pr

Recall/Precision Threshold: recall/precision threshold at x means that any examples where the score is between 0.5-x and
0.5+x are not included in calculating either metric.
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# get directory of cv
cv_run_dir = Path('C:/Users/cyates2/experiments/cvs/cv_merged_conv_7-10-2023_50')

# get ranked scores table from cv
ranking_tbl = pd.read_csv(cv_run_dir / f'ensemble_ranked_predictions_allfolds.csv')

# establish thresholds
class_threshold = 0.5  # score needed to be classified as PC
non_class_thresholds = [0.1, 0.2, 0.3, 0.4, 0.495]  # list of thresholds to calculate

class_name = 'label_id'
cat_name = 'label'

#%% compute precision and recall for different thresholds

# set columns for csv
metrics_list = ['label', 'avg err score', 'recall']
metrics_list += [f'recall at {thresh}' for thresh in non_class_thresholds]

# table to turn into csv
data_to_tbl = {col: [] for col in metrics_list}

labels = ['T-KP', 'T-CP', 'T-NTP', 'T-FA', 'T-EB', 'T-FP']

# calc metrics for each label
for label in labels:
    data_to_tbl['label'].extend([label])

    # calc avg score of erroneous classifications
    wrong_scores = ranking_tbl.loc[(ranking_tbl['predicted class'] != ranking_tbl[class_name]) &
                                   (ranking_tbl[cat_name] == label)]['score']
    data_to_tbl['avg err score'].append(np.average(wrong_scores))

    # calc recall at each threshold (including 0)
    for thresh in [0] + non_class_thresholds:
        # get amount in dataset of the given label outside the exclusion threshold
        total_label = ranking_tbl.loc[(ranking_tbl[cat_name] == label) &
                                  ((ranking_tbl['score'] < class_threshold - thresh) |
                                   (ranking_tbl['score'] > class_threshold + thresh))].shape[0]

        # get total correctly clssified outside exclusion threshold
        total_correct = ranking_tbl.loc[(ranking_tbl['predicted class'] == ranking_tbl[class_name]) &
                                        (ranking_tbl[cat_name] == label) &
                                        ((ranking_tbl['score'] < class_threshold - thresh) |
                                         (ranking_tbl['score'] > class_threshold + thresh))].shape[0]

        # add data to table
        if thresh == 0:
            data_to_tbl['recall'].append(total_correct/total_label)
        else:
            data_to_tbl[f'recall at {thresh}'].append(total_correct / total_label)

# save table to cv directory
metrics_df = pd.DataFrame(data_to_tbl)
metrics_df.to_csv(cv_run_dir / f'additional_metrics.csv', index=False)


#%% make graphs with threshold for precision and recall

recalls = []
precisions = []

non_class_thresholds = [0] + non_class_thresholds

# calculate prec/rec for each threshold for entire dataset to create graph
for thresh in non_class_thresholds:
    # get total amount of PCs in dataset
    total_1 = ranking_tbl.loc[(ranking_tbl[class_name] == 1) &
                                  ((ranking_tbl['score'] < class_threshold - thresh) |
                                   (ranking_tbl['score'] > class_threshold + thresh))].shape[0]

    # get total amount classified as a PC by model
    total_class = ranking_tbl.loc[(ranking_tbl['predicted class'] == 1) &
                                  (ranking_tbl['score'] > thresh + class_threshold)].shape[0]

    # get total amount classified as 1 and is a PC
    total_correct = ranking_tbl.loc[(ranking_tbl[class_name] == 1) &
                                    (ranking_tbl['score'] > class_threshold + thresh)].shape[0]
    recalls.append(total_correct/total_1)
    precisions.append(total_correct/total_class)

# plot prec/rec graph
recall_curve = plt.plot(non_class_thresholds, recalls, 'r', label='recall')
precision_curve = plt.plot(non_class_thresholds, precisions, 'b', label='precision')
plt.xlabel('Threshold')
plt.ylabel('Performance')
plt.legend()
plt.savefig(cv_run_dir / 'rec_prec_thresh.png')



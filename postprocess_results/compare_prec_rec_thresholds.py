"""
Script purpose:

1. Plot recall/precision threshold values as function of distance to classification threshold of multiple experiments
on same graph.
2. Compare recall/precision values among experiments.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

experiments_dir = Path('C:/Users/cyates2/experiments/cvs/')

# locations of each experiment to compare
cv_run_dirs = [Path('C:/Users/cyates2/experiments/cvs/cv_TESS_conv_7-17-2023_50'),
               Path('C:/Users/cyates2/experiments/cvs/cv_TESS_base_7-12-2023_1028'),
               Path('C:/Users/cyates2/experiments/cvs/cv_TESS_fluxvar_7-17-2023_50')]

# get ranked scores of each experiment
ranking_tbls = [pd.read_csv(cv_run_dir / f'ensemble_ranked_predictions_allfolds.csv')
                for cv_run_dir in cv_run_dirs]

class_name = 'label_id'

class_threshold = 0.5   # score need to be classified as PC
non_class_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.495]   # thresholds to compute

# make graphs with threshold for precision and recall

recalls = []
precisions = []

for ranking_tbl in ranking_tbls:
    rec = []
    prec = []

    # get amount of PCs in dataset
    total_1 = ranking_tbl.loc[ranking_tbl[class_name] == 1].shape[0]

    # calculate metrics for each threshold
    for thresh in non_class_thresholds:
        # get total correctly classified as PC based on threshold
        total_correct = ranking_tbl.loc[(ranking_tbl[class_name] == 1) &
                                        (ranking_tbl['score'] > class_threshold + thresh)].shape[0]

        # get total classified as PC based on threshold
        total_class = ranking_tbl.loc[(ranking_tbl['predicted class'] == 1) &
                                      (ranking_tbl['score'] > thresh + class_threshold)].shape[0]

        # get total number of true PCs which are outside of exclusion threshold
        total_1 = ranking_tbl.loc[(ranking_tbl[class_name] == 1) &
                                        ((ranking_tbl['score'] < class_threshold - thresh) |
                                         (ranking_tbl['score'] > class_threshold + thresh))].shape[0]
        rec.append(total_correct/total_1)
        prec.append(total_correct/total_class)

    recalls.append(rec)
    precisions.append(prec)

# plot recalls for each experiment
for i in range(len(ranking_tbls)):
    plt.plot(non_class_thresholds, recalls[i], label=cv_run_dirs[i].name.split('_')[2])
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.legend()
plt.savefig(experiments_dir / 'compare_recall_threshold.png')
plt.close()

# plot precisions for each experiment
for i in range(len(ranking_tbls)):
    plt.plot(non_class_thresholds, precisions[i], label=cv_run_dirs[i].name.split('_')[2])
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.legend()
plt.savefig(experiments_dir / 'compare_precision_threshold.png')
plt.close()



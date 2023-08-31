"""
Script purpose:

1. Get average misclassified score for each category.
2. Compare average misclassified score among experiments.
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path

# get experiment paths + names
cv_run_dirs = [Path('/Users/cyates2/experiments/cvs/cv_merged_conv_7-30-2023_50'),
               Path('/Users/cyates2/experiments/cvs/cv_merged_base_7-30-2023_1028'),
               Path('/Users/cyates2/experiments/cvs/cv_TESS_conv_7-17-2023_50'),
               Path('/Users/cyates2/experiments/cvs/cv_merged_conv_7-10-2023_50'),
               Path('/Users/cyates2/experiments/cvs/cv_TESS_fluxvar_7-17-2023_50'),
               Path('/Users/cyates2/experiments/cvs/cv_merged_base_6-30-2023_1028')]
exp_names = ['Comb Conv Allvar (K+T)',
             'Combined Baseline (K+T)',
             'Comb Conv Allvar (T)',
             'Unfolded (K+T)',
             'Combined Allvar (T)',
             'Baseline (K+T)']

class_name = 'label_id'
cat_name = 'label'
labels = ['T-KP', 'T-CP', 'T-NTP', 'T-FA', 'T-EB', 'T-FP']

metrics_list = ['Experiment']
metrics_list += ['avg_misclf_score_' + label.split("-")[-1] for label in labels]

data_to_tbl = {col: [] for col in metrics_list}

for i, cv_run_dir in enumerate(cv_run_dirs):

    data_to_tbl['Experiment'].extend([exp_names[i]])

    ranking_tbl = pd.read_csv(cv_run_dir / f'ensemble_ranked_predictions_allfolds.csv')

    for label in labels:
        wrong_scores = ranking_tbl.loc[(ranking_tbl['predicted class'] != ranking_tbl[class_name]) &
                                       (ranking_tbl[cat_name] == label)]['score']
        data_to_tbl['avg_misclf_score_' + label.split('-')[-1]].append(np.average(wrong_scores))

metrics_df = pd.DataFrame(data_to_tbl)
metrics_df.to_csv('/Users/cyates2/experiments/cvs/' + f'avg_misclf_score.csv', index=False)

""" Combine predictions from all CV folds. """

# 3rd party
from pathlib import Path

import numpy as np
import pandas as pd

# %% Combine predictions from all CV iterations in the used dataset

cv_run_dir = Path(
    '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_noplanets_ruwe-l1.2_8-26-2022_1148')

cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_testset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    cv_iters_tbls.append(ranking_tbl)

ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
ranking_tbl_cv.to_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv', index=False)

# %% Combine predictions from all CV iterations in the not-used dataset

cv_pred_run_dir = Path(
    '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_predict_kepler_noplanets_ruwe-l1.2_8-26-2022_1148')

cv_iters_dirs = [fp for fp in cv_pred_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    cv_iters_tbls.append(ranking_tbl)

ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
ranking_tbl_cv.to_csv(cv_pred_run_dir / 'ensemble_ranked_predictions_allfolds.csv', index=False)

# %% Get mean and std score across all folds for the prediction on the not-used dataset

tbl = None
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    ranking_tbl[f'score_fold_{cv_iter_dir.name.split("_")[-1]}'] = ranking_tbl['score']
    if tbl is None:
        tbl = ranking_tbl
    else:
        tbl = pd.merge(tbl, ranking_tbl[['target_id', 'tce_plnt_num', f'score_fold_{cv_iter_dir.name.split("_")[-1]}']],
                       on=['target_id', 'tce_plnt_num'])

tbl.drop(columns=['fold', 'score', 'predicted class'], inplace=True)
tbl['mean_score'] = tbl[[f'score_fold_{i}' for i in range(9)]].mean(axis=1)
tbl['std_score'] = tbl[[f'score_fold_{i}' for i in range(9)]].std(axis=1)
tbl.to_csv(cv_pred_run_dir / 'ensemble_ranked_predictions_allfolds_avg.csv', index=False)

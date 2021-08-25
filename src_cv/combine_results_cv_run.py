""" Combine predicitons from all CV folds. """

# 3rd party
from pathlib import Path

import pandas as pd

# %% Combine predictions from all CV iterations in the used dataset

cv_run_dir = Path('/data5/tess_project/experiments/current_experiments/cv_experiments/'
                  'cv_keplerq1q17dr25_exominer_configk_newsec_rbacnt0n_split_tce_8-24-2021')

cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_testset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    cv_iters_tbls.append(ranking_tbl)

ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
ranking_tbl_cv.to_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv', index=False)

# %% Combine predictions from all CV iterations in the not-used dataset

cv_pred_run_dir = Path('/data5/tess_project/experiments/current_experiments/cv_experiments/'
                       'cv_keplerq1q17dr25_exominer_configk_newsec_rbacnt0n_8-23-2021/predict_not_used_8-24-2021')

cv_iters_dirs = [fp for fp in cv_pred_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    cv_iters_tbls.append(ranking_tbl)

ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
ranking_tbl_cv.to_csv(cv_pred_run_dir / 'ensemble_ranked_predictions_allfolds.csv', index=False)

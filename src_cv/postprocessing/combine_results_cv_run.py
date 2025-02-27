""" Combine predictions from all CV folds. """

# 3rd party
from pathlib import Path
import pandas as pd

# %% Combine predictions from all CV iterations in the used dataset

cv_run_dirs = [
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_valffionly_noduplicate2mindata_2-18-2025_1135'),
               ]
for cv_run_dir in cv_run_dirs:

    cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

    cv_iters_tbls = []
    for cv_iter_dir in cv_iters_dirs:
        ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_model' / 'ranked_predictions_testset.csv')
        ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
        cv_iters_tbls.append(ranking_tbl)

    ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
    ranking_tbl_cv.to_csv(cv_run_dir / 'ranked_predictions_allfolds.csv', index=False)

# %% Combine predictions from all CV iterations in the not-used dataset

cv_pred_run_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_2min/toi_2095/cv_predict_tess-spoc-2min_s14-s78_toi-2095_exominerplus2_keplertrain_2-19-2025_1605')

cv_iters_dirs = [fp for fp in cv_pred_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

cv_iters_tbls = []
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ranked_predictions_predictset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    cv_iters_tbls.append(ranking_tbl)

ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
ranking_tbl_cv.to_csv(cv_pred_run_dir / 'ranked_predictions_allfolds.csv', index=False)

# %% Get mean and std score across all folds for the prediction on the not-used dataset

tbl = None
n_folds = len(cv_iters_dirs)
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ranked_predictions_predictset.csv')
    ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
    ranking_tbl[f'score_fold_{cv_iter_dir.name.split("_")[-1]}'] = ranking_tbl['score']
    if tbl is None:
        tbl = ranking_tbl
    else:
        tbl = pd.merge(tbl, ranking_tbl[['uid', f'score_fold_{cv_iter_dir.name.split("_")[-1]}']],
                       on=['uid'])

tbl.drop(columns=['fold', 'score'], inplace=True)
tbl['mean_score'] = tbl[[f'score_fold_{i}' for i in range(n_folds)]].mean(axis=1)
tbl['std_score'] = tbl[[f'score_fold_{i}' for i in range(n_folds)]].std(axis=1)
tbl.to_csv(cv_pred_run_dir / 'ranked_predictions_allfolds_avg.csv', index=False)

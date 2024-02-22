""" Combine predictions from all CV folds. """

# 3rd party
from pathlib import Path
import pandas as pd

# %% Combine predictions from all CV iterations in the used dataset

# cv_run_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/single_branch_experiments')
cv_run_dirs = [
    Path('//Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xrp_year1/cv_tess_nodiffimg_2-13-2024_1039'),
    # Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_fpflags_4-2023/single_branch_experiments/cv_kepler_single_branch_fpflags_ntl_4-5-2023_0007')
               ]  # [fp for fp in cv_run_root_dir.iterdir() if fp.is_dir()]
for cv_run_dir in cv_run_dirs:

    # cv_run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler_single_branch_3-2023/cv_kepler_single_branch_full_exominer_3-9-2023_1147')

    cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

    cv_iters_tbls = []
    for cv_iter_dir in cv_iters_dirs:
        ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_model' / 'ensemble_ranked_predictions_testset.csv')
        ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
        cv_iters_tbls.append(ranking_tbl)

    ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
    ranking_tbl_cv.to_csv(cv_run_dir / 'ensemble_ranked_predictions_allfolds.csv', index=False)

# %% Combine predictions from all CV iterations in the not-used dataset

cv_pred_run_dir = Path(
    '/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_q1q17dr25_simdata/experiments/cv_exominer_obs_2-15-2024_1114_predict_sim_2-16-1436')

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
n_folds = 10
for cv_iter_dir in cv_iters_dirs:
    ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
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
tbl.to_csv(cv_pred_run_dir / 'ensemble_ranked_predictions_allfolds_avg.csv', index=False)

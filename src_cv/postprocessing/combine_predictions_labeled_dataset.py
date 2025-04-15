""" Combine predictions from all CV folds. """

# 3rd party
from pathlib import Path
import pandas as pd


def aggregate_cv_fold_predictions(cv_run_dir, tbl_sub_dir):
    """ Combine predictions from all CV iterations in the labeled dataset. `cv_run_dir` should contain directories for
    each CV iteration named 'cv_iter_<iteration_number>'. Each CV iteration directory should contain a
    file under `tbl_sub_dir` with the predictions for the labeled dataset named 'ranked_predictions_predictset.csv'.

    The resulting table will contain the predictions for all CV iterations, with an additional column 'fold'.

    Args:
        cv_run_dir: Path, directory containing the CV iterations directories
        tbl_sub_dir: str, directory inside `cv_run_dir` containing the tables with the predictions for each CV iteration

    Returns: ranking_tbl_cv, DataFrame with the predictions for all CV folds

    """
    cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

    cv_iters_tbls = []
    for cv_iter_dir in cv_iters_dirs:
        ranking_tbl = pd.read_csv(cv_iter_dir / tbl_sub_dir / 'ranked_predictions_testset.csv')
        ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
        cv_iters_tbls.append(ranking_tbl)

    ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)

    return ranking_tbl_cv


if __name__ == '__main__':

    tbl_sub_dir = 'ensemble_model'  # 'ensemble_model'  # models/model0'  # 'ensemble_model'

    cv_run_dirs = [
        Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_patience50ffi_exominernew_4-11-2025_1255'),
    ]
    for cv_run_dir in cv_run_dirs:

        ranking_tbl_cv = aggregate_cv_fold_predictions(cv_run_dir, tbl_sub_dir)

        ranking_tbl_cv.to_csv(cv_run_dir / 'ranked_predictions_allfolds.csv', index=False)
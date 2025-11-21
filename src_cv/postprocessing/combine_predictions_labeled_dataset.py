""" Combine predictions from all CV folds. """

# 3rd party
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def aggregate_cv_fold_predictions(cv_run_dir, tbl_sub_dir, tbl_fn='predictions_testset.csv'):
    """ Combine predictions from all CV iterations in the labeled dataset. `cv_run_dir` should contain directories for
    each CV iteration named 'cv_iter_<iteration_number>'. Each CV iteration directory should contain a
    file under `tbl_sub_dir` with the predictions for the labeled dataset named 'ranked_predictions_predictset.csv'.

    The resulting table will contain the predictions for all CV iterations, with an additional column 'fold'.

    Args:
        cv_run_dir: Path, directory containing the CV iterations directories
        tbl_sub_dir: str, directory inside `cv_run_dir` containing the tables with the predictions for each CV iteration
        tbl_fn: str, filename for predictions table within each CV iteration

    Returns: ranking_tbl_cv, DataFrame with the predictions for all CV folds

    """

    cv_iters_dirs = sorted([fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')],
                            key=lambda d: int(d.name.split('_')[-1]))

    cv_iters_tbls = []
    for cv_iter_dir in tqdm(cv_iters_dirs, desc='CV iteration', total=len(cv_iters_dirs)):
        cv_iter_tbl_fp = cv_iter_dir / tbl_sub_dir / f'{tbl_fn}.csv'
        if not cv_iter_tbl_fp.exists():
            print(f'Table for CV iteration {cv_iter_dir} not found. Skipping...')
            continue
        
        ranking_tbl = pd.read_csv(cv_iter_tbl_fp, comment='#')
        ranking_tbl['cv_iteration'] = cv_iter_dir.name.split('_')[-1]
        cv_iters_tbls.append(ranking_tbl)

    ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)

    return ranking_tbl_cv


if __name__ == '__main__':

    tbl_sub_dir = 'ensemble_model'  # 'ensemble_model'  # models/model0'  # 'ensemble_model'
    tbl_fn = 'predictions_testset'  # which set to aggregate across CV iterations (test set is unique; validation and training might not be)

    cv_run_dirs = [
        Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_exomninerpp_11-18-2025_1505'),
    ]
    for cv_run_dir in cv_run_dirs:

        prediction_tbl_cv = aggregate_cv_fold_predictions(cv_run_dir, tbl_sub_dir, tbl_fn=tbl_fn)
        # add metadata
        prediction_tbl_cv.attrs['CV experiment'] = str(cv_run_dir)
        prediction_tbl_cv.attrs['created'] = str(pd.Timestamp.now().floor('min'))
        with open(cv_run_dir / f'{tbl_fn}_allfolds.csv', "w") as f:
            for key, value in prediction_tbl_cv.attrs.items():
                f.write(f"# {key}: {value}\n")
            prediction_tbl_cv.to_csv(f, index=True)
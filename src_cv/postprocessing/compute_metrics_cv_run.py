"""
Computing metrics for each CV fold and for the whole dataset (aggregates all separated test CV folds.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

# local
from src.postprocessing.compute_metrics_from_predictions_csv_file import compute_metrics_from_predictions

def compute_metrics_stats_cv_run(cv_run_dir, top_k_vals, datasets, label_map, clf_threshold=0.5, num_thresholds=1000, class_name='label_id', cat_name='label', pred_tbl_prefix='predictions', results_sub_dir='ensemble', compute_mean_std_metrics=True, compute_metrics_all_dataset=True):
    """ Compute evaluation metrics for a cross-validation (CV) experiment across multiple datasets and CV iterations.

    This function aggregates metrics computed from prediction tables generated during a CV run. It supports:
    - Per-fold metrics for each dataset.
    - Mean and standard deviation of metrics across CV folds.
    - Optional computation of metrics for the entire dataset by combining folds (for non-overlapping CV splits).
    - Saving aggregated metrics tables with metadata.

    Parameters
    ----------
    cv_run_dir : pathlib.Path
        Path to the directory containing the CV run results. Each CV iteration should be in a subdirectory named `cv_iter*`.
    top_k_vals : list of int
        List of top-k values for ranking-based metrics (e.g., top-1, top-5 accuracy).
    datasets : list of str
        Names of datasets to process (e.g., ['train', 'val', 'test']).
    label_map : dict
        Mapping from categorical labels to numeric IDs.
    clf_threshold : float, optional (default=0.5)
        Classification threshold for binary/multi-class predictions.
    num_thresholds : int, optional (default=1000)
        Number of thresholds for computing metrics like AUC.
    class_name : str, optional (default='label_id')
        Column name for numeric class labels in the predictions table.
    cat_name : str, optional (default='label')
        Column name for categorical labels in the predictions table.
    pred_tbl_prefix : str, optional (default='predictions')
        Prefix for prediction table filenames.
    results_sub_dir : str, optional (default='ensemble')
        Subdirectory within each CV iteration directory containing prediction tables.
    compute_mean_std_metrics : bool, optional (default=True)
        Whether to compute and append mean and standard deviation of metrics across CV folds.
    compute_metrics_all_dataset : bool, optional (default=True)
        Whether to compute metrics for the entire dataset by combining folds (only valid for non-overlapping splits).

    Returns
    -------
    pandas.DataFrame
        Aggregated metrics for all datasets and CV folds, including mean and standard deviation if requested.
        The DataFrame includes metadata in its `.attrs` attribute.

    Side Effects
    ------------
    - Writes per-dataset metrics tables and an aggregated metrics table to `cv_run_dir`.
    - Adds metadata (CV experiment path, timestamp, thresholds, label map) as comments in saved CSV files.

    Notes
    -----
    - Assumes prediction tables exist in each CV iteration directory under `results_sub_dir`.
    - For combined test set metrics, requires a file named `{pred_tbl_prefix}_testset_allfolds.csv` in `cv_run_dir`.
    - Uses CPU for metric computation to avoid GPU overhead.

    Example
    -------
    >>> compute_metrics_stats_cv_run(
    ...     cv_run_dir=Path('/path/to/cv_run'),
    ...     top_k_vals=[1, 5],
    ...     datasets=['val', 'test'],
    ...     label_map={'cat': 0, 'dog': 1}
    ... )
    """

    print(f'Getting metrics for experiment {cv_run_dir}...')

    all_datasets_metrics = []
    
    for dataset in datasets:

        print(f'[Dataset {dataset}] Getting metrics for experiment {cv_run_dir}...')

        # get directories of cv iterations in cv run
        cv_iters_dirs = [fp for fp in cv_run_dir.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

        metrics_df = []
        for cv_iter_dir in tqdm(sorted(cv_iters_dirs), desc='CV iteration', total=len(cv_iters_dirs)):  # iterate through each cv iteration
            
            pred_tbl_fp = cv_iter_dir / results_sub_dir / f'{pred_tbl_prefix}_{dataset}set.csv'
            if not pred_tbl_fp.exists():
                print(f'Predictions table {pred_tbl_fp} not found. Skipping...')
                continue
            
            ranking_tbl = pd.read_csv(pred_tbl_fp, comment='#')
            ranking_tbl['label_id'] = ranking_tbl.apply(lambda x: label_map[x[cat_name]], axis=1)

            # compute metrics
            with tf.device('/cpu:0'):
                metrics_df_cv_iter = compute_metrics_from_predictions(ranking_tbl, label_map, num_thresholds,
                                                                      clf_threshold, top_k_vals, class_name, cat_name)
                metrics_df_cv_iter['fold'] = cv_iter_dir.name

            metrics_df.append(metrics_df_cv_iter)

        metrics_df = pd.concat(metrics_df, axis=0)

        # mean and std across all CV folds
        if compute_mean_std_metrics:
            # print(metrics_df)
            numeric_cols = metrics_df.select_dtypes(include='number').columns
            numeric_cols = [col for col in numeric_cols if col != 'fold']

            mean_df = metrics_df[numeric_cols].mean(axis=0).to_frame().T
            mean_df['fold'] = 'mean'
            std_df = metrics_df[numeric_cols].std(axis=0).to_frame().T
            std_df['fold'] = 'std'
            metrics_df = pd.concat([metrics_df, mean_df, std_df])

        metrics_df['dataset'] = dataset
        metrics_df.set_index('fold', inplace=True)
    
        all_datasets_metrics.append(metrics_df)

    if 'test' in datasets and compute_metrics_all_dataset:
        # compute metrics for the whole dataset by combining the test set folds from all CV iterations
        # ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!
        
        pred_tbl_fp = cv_run_dir / f'{pred_tbl_prefix}_testset_allfolds.csv'
        if not pred_tbl_fp.exists():
            print(f'Cannot compute metrics for {pred_tbl_fp} since it was not found.')
        else:  
            ranking_tbl = pd.read_csv(pred_tbl_fp, comment='#')
            ranking_tbl['label_id'] = ranking_tbl.apply(lambda x: label_map[x[cat_name]], axis=1)

            # compute metrics
            metrics_df = compute_metrics_from_predictions(ranking_tbl, label_map, num_thresholds, clf_threshold,
                                                        top_k_vals, class_name, cat_name)
            metrics_df.attrs['predictions_table'] = str(pred_tbl_fp)
            metrics_df.attrs['label_map'] = label_map
            with open(cv_run_dir / f'metrics_{dataset}_all.csv', "w") as f:
                for key, value in metrics_df.attrs.items():
                    f.write(f"# {key}: {value}\n")
                metrics_df.to_csv(f, index=False)

    # create table with mean and std metrics for all datasets
    if compute_mean_std_metrics:
        all_datasets_metrics = pd.concat(all_datasets_metrics, axis=0).reset_index(names='fold/statistic')
        all_datasets_metrics = all_datasets_metrics.set_index('dataset')
        
        # add metadata
        all_datasets_metrics.attrs['CV experiment'] = str(cv_run_dir)
        all_datasets_metrics.attrs['created'] = str(pd.Timestamp.now().floor('min'))
        all_datasets_metrics.attrs['num_thresholds'] = num_thresholds
        all_datasets_metrics.attrs['classification_threshold'] = clf_threshold
        all_datasets_metrics.attrs['label_map'] = label_map
        with open(cv_run_dir / 'metrics_allfolds_with_stats.csv', "w") as f:
            for key, value in all_datasets_metrics.attrs.items():
                f.write(f"# {key}: {value}\n")
            all_datasets_metrics.to_csv(f, index=True)

if __name__ == '__main__':
    
    tf.config.set_visible_devices([], 'GPU')
    
    num_thresholds = 1000  # number of thresholds used to compute AUC
    clf_threshold = 0.5  # classification threshold used to compute accuracy, precision and recall
    multiclass = False  # multiclass or bin class?
    target_score = 'score_AFP'  # get auc_pr metrics for different class labels
    class_name = 'label_id'
    cat_name = 'label'  # 'obs_type'  # 'label'
    label_map = {
        'CP': 1,
        'KP': 1,
        'FP': 0,
        'EB': 0,
        'NTP': 0,
        'BD': 0,
    }
    # mapping of category/disposition to label id
    # cats = None
    class_ids = [0, 1]  # should match unique label ids in 'cats'
    top_k_vals = [50, 100, 150, 200, 500, 1000, 2000, 3000]
    # top_k_vals = {
    #     'train': [50, 100, 250, 500, 1000, 2000],  # , 2500]
    #     'val': [25, 50, 100, 200],
    #     'test': [25, 50, 100, 200],
    #               }
    datasets = [
        'train',
        'val',
        'test'
    ]

    # compute metrics for the whole dataset by combining the test set folds from all CV iterations
    # ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!
    compute_metrics_all_dataset = True
    # if True, computes mean and std metrics' values across CV iterations
    compute_mean_std_metrics = True

    results_sub_dir = 'ensemble_model'  # 'ensemble_model'  # 'models/model0'  # 'ensemble_model'
    pred_tbl_prefix = 'predictions'

    # cv experiment directories
    cv_run_dirs = [
        Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_exomninernew-nolayernorm_11-27-2025_1124'),
    ]
    for cv_run_dir in cv_run_dirs:  # iterate through multiple CV runs

        compute_metrics_stats_cv_run(cv_run_dir, top_k_vals, datasets, label_map, clf_threshold=clf_threshold, num_thresholds=num_thresholds, class_name=class_name, cat_name=cat_name, pred_tbl_prefix=pred_tbl_prefix, results_sub_dir=results_sub_dir, compute_mean_std_metrics=compute_mean_std_metrics, compute_metrics_all_dataset=compute_metrics_all_dataset)
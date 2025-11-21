"""
"""

# 3rd party
import pandas as pd
from pathlib import Path


def aggregate_loss_and_metrics_cv_iterations(cv_iteration_dirs):
    """Aggregate loss and performance metrics across CV iterations.

    :param list cv_iteration_dirs: contains Path objects to each CV iteration directory
    :raises ValueError: CSV file named "loss_and_performance_metrics.csv" not found in the CV iteration directory
    :raises ValueError: more than one CSV file named "loss_and_performance_metrics.csv" not found in the CV iteration directory
    :return pandas DataFrame: table with aggregate loss and performance metrics across CV iterations
    """
    
    metric_tbl_name = 'loss_and_performance_metrics.csv'
    
    metrics_tbls = []
    for cv_iteration_dir in sorted(cv_iteration_dirs):
        
        cv_iter_id = cv_iteration_dir.name
        
        cv_iter_metrics_tbl_fp = list(cv_iteration_dir.rglob(metric_tbl_name))
        if len(cv_iter_metrics_tbl_fp) == 0:
            raise ValueError(f'Could not find performance metrics CSV file named {metric_tbl_name} for CV iteration {cv_iter_id} in {cv_iteration_dir}.')
        elif len(cv_iter_metrics_tbl_fp) > 1:
            raise ValueError(f'Found {len(cv_iter_metrics_tbl_fp)} performance metrics CSV file named {metric_tbl_name} for CV iteration {cv_iter_id} in {cv_iteration_dir}.')
        else:
            cv_iter_metrics_tbl_fp = cv_iter_metrics_tbl_fp[0]
            
        cv_iter_metrics_tbl = pd.read_csv(cv_iter_metrics_tbl_fp, index_col='metrics', comment='#')
        cv_iter_metrics_tbl.columns = pd.MultiIndex.from_product([[cv_iter_id], cv_iter_metrics_tbl.columns])
        
        metrics_tbls.append(cv_iter_metrics_tbl)
    
    metrics_tbl = pd.concat(metrics_tbls, axis=1)
    
    return metrics_tbl


if __name__ == '__main__':
    
    cv_exp_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_exomninerpp_11-18-2025_1505')
    cv_iters = list(cv_exp_dir.glob('cv_iter_*'))
    print(f'Found {len(cv_iters)} CV iteration directories')
    agg_metrics_tbl = aggregate_loss_and_metrics_cv_iterations(cv_iters)
    # add metadata
    agg_metrics_tbl.attrs['CV experiment'] = str(cv_exp_dir)
    agg_metrics_tbl.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(cv_exp_dir / 'agg_loss_and_performance_metrics.csv', "w") as f:
        for key, value in agg_metrics_tbl.attrs.items():
            f.write(f"# {key}: {value}\n")
        agg_metrics_tbl.to_csv(f, index=True)
    
    print('Done.')

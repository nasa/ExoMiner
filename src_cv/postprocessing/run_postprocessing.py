"""Run postprocessing pipeline to generate results after finishing CV run.
"""

# 3rd party
import pandas as pd
import tensorflow as tf
from pathlib import Path
import argparse
import yaml

# local
from src_cv.postprocessing.aggregate_metrics_cv_iterations import aggregate_loss_and_metrics_cv_iterations
from src_cv.postprocessing.combine_predictions_labeled_dataset import aggregate_cv_fold_predictions
from src_cv.postprocessing.compute_metrics_cv_run import compute_metrics_stats_cv_run


def run_postprocessing_pipeline(cv_run: Path, pipeline_config: dict):
    """Run postprocessing pipeline.

    :param Path cv_run: CV run directory
    :param dict pipeline_config: postprocessing run parameters
    """
    
    # aggregate CV fold predictions
    print('Aggregating predictions across CV iterations...')
    prediction_tbl_cv = aggregate_cv_fold_predictions(cv_run, pipeline_config['tbl_sub_dir'], tbl_fn=pipeline_config['tbl_fn'])
    # add metadata
    prediction_tbl_cv.attrs['CV experiment'] = str(cv_run)
    prediction_tbl_cv.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(cv_run / f'{pipeline_config["tbl_fn"]}_allfolds.csv', "w") as f:
        for key, value in prediction_tbl_cv.attrs.items():
            f.write(f"# {key}: {value}\n")
        prediction_tbl_cv.to_csv(f, index=False)
    print('Done.')
    
    # aggregate
    print('Aggregating loss and performance metrics computed for each CV iteration...')
    # cv_iters = list(cv_run.glob('cv_iter_*'))
    print(f'Found {len(pipeline_config["cv_iters"])} CV iteration directories')
    agg_metrics_tbl = aggregate_loss_and_metrics_cv_iterations(pipeline_config['cv_iters'])
    # add metadata
    agg_metrics_tbl.attrs['CV experiment'] = str(cv_run)
    agg_metrics_tbl.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(cv_run / 'agg_loss_and_performance_metrics.csv', "w") as f:
        for key, value in agg_metrics_tbl.attrs.items():
            f.write(f"# {key}: {value}\n")
        agg_metrics_tbl.to_csv(f, index=True)
    print('Done.')
    
    # compute metrics
    print('Computing CV metrics...')
    compute_metrics_stats_cv_run(
        cv_run, 
        pipeline_config['top_k_vals'], 
        pipeline_config['datasets'], 
        pipeline_config['label_map'], 
        clf_threshold=pipeline_config['clf_threshold'], 
        num_thresholds=pipeline_config['num_thresholds'], 
        recall_at_precision_thr=pipeline_config['recall_at_precision_thr'], 
        precision_at_recall_thr=pipeline_config['precision_at_recall_thr'], 
        class_name=pipeline_config['class_name'], 
        cat_name=pipeline_config['cat_name'], 
        pred_tbl_prefix=pipeline_config['pred_tbl_prefix'], 
        results_sub_dir=pipeline_config['results_sub_dir'], 
        compute_mean_std_metrics=pipeline_config['compute_mean_std_metrics'], 
        compute_metrics_all_dataset=pipeline_config['compute_metrics_all_dataset'],
        multiclass=pipeline_config['multiclass'],
        multiclass_target=pipeline_config['target_score'],
        )
    print('Done.')
    

if __name__ == '__main__':
    
    tf.config.set_visible_devices([], 'GPU')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_run', type=str, help='CV run directory.', required=True)
    parser.add_argument('--config_fp', type=str, help='Postprocessing configuration YAML.', required=True)
    args = parser.parse_args()
    
    cv_run = Path(args.cv_run)
    
    with open(args.config_fp) as f:
        pipeline_config = yaml.safe_load(f)
    
    pipeline_config['cv_iters'] = list(cv_run.glob('cv_iter_*'))

    run_postprocessing_pipeline(cv_run, pipeline_config)
    
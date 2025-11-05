""" Utility functions for evaluating a Keras model. """

# 3rd party
import pandas as pd
import numpy as np


def write_performance_metrics_to_txt_file(save_dir, datasets, res_eval):
    """ Write performance metrics in dictionary `res_eval` based on a model's evaluation to a TXT file.

    Args:
        save_dir: Path, save directory
        datasets: list, data sets for which to save metrics
        res_eval: dict, performance metrics for each data set (should include data sets in `datasets`)

    Returns:

    """

    # write results to a txt file
    with open(save_dir / 'loss_and_performance_metrics.txt', 'w') as res_file:

        str_aux = f'Performance metrics for the model\n'
        res_file.write(str_aux)

        for dataset in datasets:  # iterate over data sets
            if dataset != 'predict':  # no metrics for unlabeled data set

                # grab metrics names for data set
                res_eval_dataset_metrics_names = [metric_name for metric_name in res_eval.keys()
                                                  if dataset in metric_name]

                str_aux = f'Dataset: {dataset}\n'
                res_file.write(str_aux)

                for metric in res_eval_dataset_metrics_names:
                    if isinstance(res_eval[metric], float):  # only write metrics that are scalars
                        str_aux = f'{metric}: {res_eval[f"{metric}"]}\n'
                        res_file.write(str_aux)

            res_file.write('\n')


def write_performance_metrics_to_csv_file(save_dir, datasets, res_eval, logger=None):
    """ Write performance metrics in dictionary `res_eval` based on a model's evaluation to a CSV file.

    Args:
        save_dir: Path, save directory
        datasets: list, data sets for which to save metrics
        res_eval: dict, performance metrics for each data set (should include data sets in `datasets`)
        logger: Python Logger, logger

    Returns:
    """

    metrics_dict = {dataset: [] for dataset in datasets if dataset != 'predict'}
    metric_names_for_csv_file = []
    for dataset in datasets:  # iterate over data sets
        if dataset != 'predict':  # no metrics for unlabeled data set

            # grab metrics names for data set
            res_eval_dataset_metrics_names = ['_'.join(metric_name.split('_')[1:]) 
                                              for metric_name in res_eval.keys()
                                              if dataset in metric_name]

            if len(metric_names_for_csv_file) == 0:
                for metric in res_eval_dataset_metrics_names:
                    if isinstance(res_eval[f'{dataset}_{metric}'], float):  # only write metrics that are scalars    
                        metric_names_for_csv_file.append(metric)
                        
            for metric in metric_names_for_csv_file:
                if f'{dataset}_{metric}' not in res_eval:
                    if logger is None:
                        print(f'Loss/metric "{dataset}_{metric}" not computed. Setting it to NaN.')
                    else:
                        logger.info(f'Loss/metric "{dataset}_{metric}" not computed for dataset {dataset}. Setting it to NaN.')
                    
                    metrics_dict[dataset].append(np.nan)
                else:    
                    metrics_dict[dataset].append(res_eval[f'{dataset}_{metric}'])
                    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df['metrics'] = res_eval_dataset_metrics_names
    metrics_df.set_index('metrics', inplace=True)
                        
    # add metadata
    metrics_df.attrs['dataset'] = str(save_dir)
    metrics_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(save_dir / 'loss_and_performance_metrics.csv', "w") as f:
        for key, value in metrics_df.attrs.items():
            f.write(f"# {key}: {value}\n")
        metrics_df.to_csv(f, index=True)

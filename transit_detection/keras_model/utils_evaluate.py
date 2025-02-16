import numpy as np
import csv

def write_performance_metrics_from_np_to_csv_file(np_path, csv_path):
    """
    TODO: Add input/Return val
    
    """
    np_dict = np.load(str(np_path), allow_pickle=True).item() # pickle for results dict

    res = {}

    for key, value in np_dict.items():
        if isinstance(value, float): # only write scalar values
            dataset_name = key.split('_')[0]
            metric_name = key[len(dataset_name) + 1:] # skip first _

            if dataset_name not in res:
                res[dataset_name] = {}

            res[dataset_name][metric_name] = value
        
    all_metrics = set(metric for metrics in res.values() for metric in metrics.keys())

    all_metrics = sorted(all_metrics)
    with open(csv_path, mode='w', newline="") as file:
        writer = csv.writer(file)

        header = ["Dataset"] + all_metrics
        writer.writerow(header)

        for dataset_name, metrics in res.items():
            row = [dataset_name] + [metrics.get(metric, None) for metric in all_metrics]
            writer.writerow(row)
        
    print(f"Evaluation metrics saved to {csv_path}")
    

def write_performance_metrics_to_csv_file(save_dir, datasets, res_eval):
    """ Write performance metrics in dictionary `res_eval` based on a model's evaluation.

    Args:
        save_dir: Path, save directory
        datasets: list, data sets for which to save metrics
        res_eval: dict, performance metrics for each data set (should include data sets in `datasets`)

    Returns:

    """

    # write results to a txt file
    with open(save_dir / 'loss_and_performance_metrics.csv', 'w') as res_file:

        str_aux = f'Performance metrics for the model\n'
        res_file.write(str_aux)

        writer = csv.writer(res_file)
        csv_fields = ["Dataset"] + [metric_name for metric_name in res_eval.keys()]
        

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

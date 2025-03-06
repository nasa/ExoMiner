""" Utility functions for evaluating a Keras model. """


def write_performance_metrics_to_txt_file(save_dir, datasets, res_eval):
    """ Write performance metrics in dictionary `res_eval` based on a model's evaluation.

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

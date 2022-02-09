""" Utility functions for the predicting script. """

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# local
import paths

if 'home6' in paths.path_hpoconfigs:
    plt.switch_backend('agg')


def print_metrics(res, datasets, metrics_names, prec_at_top, models_filepaths):
    """ Print results.

    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, dataset names
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :param models_filepaths: list, models' filepaths
    :return:
    """

    print('#' * 100)
    print(f'Performance metrics for the ensemble ({len(models_filepaths)} models)\n')
    for dataset in datasets:
        if dataset != 'predict':
            print(dataset)
            for metric in metrics_names:
                if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                    print(f'{metric}: {res[f"{dataset}_{metric}"]}\n')

            for k in prec_at_top[dataset]:
                print(f'{"{dataset}_precision_at_{k}"}: {res[f"{dataset}_precision_at_{k}"]}\n')
    print('#' * 100)


def save_metrics_to_file(save_path, res, datasets, metrics_names, prec_at_top, models_filepaths, print_res=False):
    """ Write results to a txt file.

    :param save_path: Path, saving directory
    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, dataset names
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :param models_filepaths: list, models' filepaths
    :param print_res: bool, if True it prints results to std_out
    :return:
    """

    # write results to a txt file
    with open(save_path / 'results_ensemble.txt', 'w') as res_file:

        str_aux = f'Performance metrics for the ensemble ({len(models_filepaths)} models)\n'
        res_file.write(str_aux)
        if print_res:
            print(str_aux)

        for dataset in datasets:

            if dataset != 'predict':
                str_aux = f'Dataset: {dataset}\n'
                res_file.write(str_aux)
                if print_res:
                    print(str_aux)
                for metric in metrics_names:
                    if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                        str_aux = f'{metric}: {res[f"{dataset}_{metric}"]}\n'
                        res_file.write(str_aux)
                        if print_res:
                            print(str_aux)

                if prec_at_top is not None:
                    for k in prec_at_top[dataset]:
                        str_aux = f'{f"{dataset}_precision_at_{k}"}: {res[f"{dataset}_precision_at_{k}"]}\n'
                        res_file.write(str_aux)
                        if print_res:
                            print(str_aux)

            res_file.write('\n')

        res_file.write(f'{"-" * 100}')

        res_file.write('\nModels used:\n')

        for model_filepath in models_filepaths:
            res_file.write(f'{model_filepath}\n')

        res_file.write('\n')


# TODO: separate functions for each curve?
def plot_prcurve_roc(res, save_path, dataset):
    """ Plot ROC and PR curves.

    :param res: dict, each key is a specific dataset ('train', 'val', ...) and the values are dicts that contain
    metrics for each dataset
    :param save_path: str, path to save directory
    :param dataset: str, dataset for which the plot is generated
    :return:
    """

    # count number of samples per class to compute TPR and FPR
    num_samples_per_class = {'positive': 0, 'negative': 0}
    num_samples_per_class['positive'] = res[f'{dataset}_tp'][0] + res[f'{dataset}_fn'][0]
    num_samples_per_class['negative'] = res[f'{dataset}_fp'][0] + res[f'{dataset}_tn'][0]

    # ROC and PR curves
    f = plt.figure(figsize=(9, 6))
    lw = 2
    ax = f.add_subplot(111, label='PR ROC')
    ax.plot(res[f'{dataset}_rec_thr'], res[f'{dataset}_prec_thr'], color='darkorange', lw=lw,
            label=f'PR ROC curve (area = {res[f"{dataset}_auc_pr"]:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax2 = f.add_subplot(111, label='AUC ROC', frame_on=False)
    ax2.plot(res[f'{dataset}_fp'] / num_samples_per_class['negative'],
             res[f'{dataset}_tp'] / num_samples_per_class['positive'],
             color='r', lw=lw, linestyle='--',
             label=f'AUC ROC curve (area = {res[f"{dataset}_auc_roc"]:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xticks(np.arange(0, 1.05, 0.05))
    ax2.set_yticks(np.arange(0, 1.05, 0.05))
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.set_label_position('top')
    ax2.legend(loc="lower left")

    # f.suptitle(f'PR/ROC Curves - {dataset}')
    f.savefig(os.path.join(save_path, f'ensemble_pr-roc_curves_{dataset}.png'))
    plt.close()


def create_ranking(scores, scores_clf, label_map, multiclass, data_examples=None):
    """ Create ranking of examples based on scores produced by a model/

    :param scores: NumPy array, scores output by the model
    :param scores_clf: NumPy array, classifications after thresholding scores
    :param label_map: dict, label to class ID map
    :param multiclass: bool, True for multiclass classification
    :param data_examples: dict, additional data for the ranked examples
    :return:
        data_df: pandas DataFrame, ranking
    """

    if data_examples is None:
        data_examples = {}

    # add predictions to the data dict
    if not multiclass:
        data_examples['score'] = scores.ravel()
        data_examples['predicted class'] = scores_clf.ravel()
    else:
        for class_label, label_id in label_map.items():
            data_examples[f'score_{class_label}'] = scores[:, label_id]
        data_examples['predicted class'] = scores_clf

    data_df = pd.DataFrame(data_examples)

    # sort in descending order of output
    if not multiclass:
        data_df.sort_values(by='score', ascending=False, inplace=True)

    return data_df

""" Plot utility functions for training/predicting. """

import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(output_cl, save_path, norm=False, log=True):
    """ Plot histogram of the class distribution as a function of the predicted output.

    :param output_cl: dict, keys are the different classes and the values are the scores for all items from that class
    :param save_path: str, filepath used to save the plots figure
    :param norm: bool, if True normalizes the number of examples per class
    :param log: bool, if True the count axis is set to logarithmic scale
    :return:
    """

    bins = np.linspace(0, 1, 11, True)
    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set', 'predict': 'Predict set'}

    hist, bin_edges = {}, {}
    for class_label in output_cl:
        counts_cl = list(np.histogram(output_cl[class_label], bins, density=False, range=(0, 1)))
        if norm:
            counts_cl[0] = counts_cl[0] / len(output_cl[class_label])
        hist[class_label] = counts_cl[0]
        bin_edges[class_label] = counts_cl[1]

    bins_multicl = np.linspace(0, 1, len(output_cl) * 10 + 1, True)
    bin_width = bins_multicl[1] - bins_multicl[0]
    bins_cl = {}
    for i, class_label in enumerate(output_cl):
        bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                for idx in range(i, len(bins_multicl) - 1, len(output_cl))]

    f, ax = plt.subplots()
    for class_label in output_cl:
        ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
    if norm:
        ax.set_ylabel('Class fraction')
    else:
        ax.set_ylabel('Number of examples')
    ax.set_xlabel('Predicted output')
    ax.set_xlim([0, 1])
    if norm:
        ax.set_ylim([0, 1])
    if log:
        ax.set_yscale('log')
    ax.set_xticks(np.linspace(0, 1, 11, True))
    ax.legend()
    ax.set_title('Output distribution')
    plt.savefig(save_path)
    plt.close()


def plot_precision_at_k(labels_ord, k_curve_arr, save_path):
    """ Plot precision-at-k and misclassified-at-k curves.

    :param labels_ord: NumPy array, labels of the items ordered by the scores given by the model
    :param k_curve_arr: NumPy array, values of k to compute precision-at-k
    :param save_path: str, filepath used to save the plots figure
    :return:
    """

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr}
    for k_i in range(len(k_curve_arr)):
        if len(labels_ord) < k_curve_arr[k_i]:
            precision_at_k[k_curve_arr[k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[k_i]] = \
                np.sum(labels_ord[-k_curve_arr[k_i]:]) / k_curve_arr[k_i]

    # precision at k curve
    f, ax = plt.subplots()
    ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
    ax.set_ylim(top=1)
    f.savefig(str(save_path) + '_precisionatk.svg')
    plt.close()

    # misclassified examples at k curve
    f, ax = plt.subplots()
    kvalues = np.array(list(precision_at_k.keys()))
    precvalues = np.array(list(precision_at_k.values()))
    num_misclf_examples = kvalues - kvalues * precvalues
    ax.plot(kvalues, num_misclf_examples)
    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
    f.savefig(str(save_path) + '_misclassifiedatk.svg')
    plt.close()

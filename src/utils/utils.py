""" Utility functions for training, evaluating and running inference with a Keras model. """

# 3rd party
import tensorflow as tf
import matplotlib.pyplot as plt


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


def set_tf_data_type_for_features(features_set):
    """ Set TF data types for features in the feature set.

    Args:
        features_set: dict, each key is the name of a feature that maps to a dictionary with keys 'dim' and 'dtype'.
        'dim' is a list that describes the dimensionality of the feature and 'dtype' the data type of the feature.
        'dtype' should be a string (either 'float' - mapped to tf.float32; or 'int' - mapped to tf.int64).

    Returns:
        features_set: the data type is now a TensorFlow data type

    """

    # choose features set
    for feature_name, feature in features_set.items():
        if feature['dtype'] == 'float':
            features_set[feature_name]['dtype'] = tf.float32
        if feature['dtype'] == 'int':
            features_set[feature_name]['dtype'] = tf.int64
        if feature['dtype'] == 'string':
            features_set[feature_name]['dtype'] = tf.string

    return features_set


def plot_loss_metric(res, epochs, save_path, ep_idx=-1, opt_metric=None):
    """ Plot loss and evaluation metric plots.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set)
    :param epochs: Numpy array, epochs
    :param save_path: str, filepath used to save the plots figure
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :return:
    """

    if opt_metric is None:
        f, ax = plt.subplots()
        ax.plot(epochs, res['loss'], label='Training', color='b')
        val_test_str = 'Loss\n'
        if 'val_loss' in res:
            ax.plot(epochs, res['val_loss'], label='Validation', color='r')
            val_test_str += f'Val {res["val_loss"][ep_idx]:.4} '
        if 'test_loss' in res:
            ax.scatter(epochs[ep_idx], res['test_loss'], c='k', label='Test')
            val_test_str += f'Test {res["test_loss"]:.4} '
        ax.set_xlim([0, epochs[-1] + 1])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(val_test_str)
        ax.legend(loc="upper right")
        ax.grid(True)
    else:
        f, ax = plt.subplots(1, 2)
        ax[0].plot(epochs, res['loss'], label='Training', color='b')
        val_test_str = 'Loss\n'
        if 'val_loss' in res:
            ax[0].plot(epochs, res['val_loss'], label='Validation', color='r')
            val_test_str += f'Val {res["val_loss"][ep_idx]:.4} '
        if 'test_loss' in res:
            ax[0].scatter(epochs[ep_idx], res['test_loss'], c='k', label='Test')
            val_test_str += f'Test {res["test_loss"]:.4} '
        ax[0].set_xlim([0, epochs[-1] + 1])
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title(val_test_str)
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        ax[1].plot(epochs, res[opt_metric], label='Training')
        val_test_str = f'{opt_metric}\n'
        if f'val_{opt_metric}' in res:
            ax[1].plot(epochs, res[f'val_{opt_metric}'], label='Validation', color='r')
            ax[1].scatter(epochs[ep_idx], res[f'val_{opt_metric}'][ep_idx], c='r')
            val_test_str += f'Val {res[f"val_{opt_metric}"][ep_idx]:.4} '
        if f'test_{opt_metric}' in res:
            ax[1].scatter(epochs[ep_idx], res[f'test_{opt_metric}'], label='Test', c='k')
            val_test_str += f'Test {res[f"test_{opt_metric}"]:.4} '
        ax[1].set_xlim([0, epochs[-1] + 1])
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid(True)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel(opt_metric)
        ax[1].set_title(val_test_str)
        ax[1].legend(loc="lower right")
    f.suptitle(f'Epochs = {epochs[-1]}(Best:{epochs[ep_idx]:})')
    # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


def plot_metric_from_res_file(res, save_path, logscale=False):
    """ Plot loss/metric from results NumPy file.

    :param res: dict, keys are loss and metrics on the training, validation and test set
    :param save_path: str, filepath used to save the plots figure
    :param logscale: bool, whether to log scale or not the y-axis
    :return:
    """

    f, ax = plt.subplots()
    for metric_name, metric_vals in res.items():
        ax.plot(metric_vals['epochs'], metric_vals['values'], label=metric_name)
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch Number')
    if logscale:
        ax.set_yscale('log')
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

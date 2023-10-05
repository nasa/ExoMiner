"""
Utility functions used during training, evaluating and predicting.
- Custom callbacks, data augmentation techniques, ...
"""

# 3rd party
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# local
import paths
from src.utils_dataio import InputFnv2 as InputFn


if 'home6' in paths.path_hpoconfigs:
    plt.switch_backend('agg')


class LayerOutputCallback(tf.keras.callbacks.Callback):

    def __init__(self, input_fn, batch_size, layer_name, summary_writer, buckets, log_dir, num_batches=None,
                 description='custom', ensemble=False):
        """ Callback that writes to a histogram summary the output of a given layer.

        :param input_fn: input function
        :param batch_size: int, batch size
        :param layer_name: str, name of the layer
        :param summary_writer: summary writer
        :param buckets: int, bucket size
        :param log_dir:
        :param num_batches: int, number of batches to extract
        :param description: str, optional description
        :param ensemble: bool, True if dealing with an ensemble of models.
        """

        super(LayerOutputCallback, self).__init__()
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.layer_name = layer_name
        self.summary_writer = summary_writer
        self.description = description
        self.log_dir = log_dir
        self.buckets = buckets
        self.ensemble = ensemble
        self.num_batches = num_batches if num_batches is not None else np.inf
        self.csv_fp = Path(self.log_dir) / f'{self.layer_name}.csv'

    # def on_batch_end(self, batch, logs=None):
    #
    #     layer = [l for l in self.model.layers if l.name == 'convbranch_wscalar_concat'][0]
    #     get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #     print('Batch {} | Input to FC block: '.format(batch), get_layer_output.outputs)

    # def on_train_batch_end(self, batch, logs=None):
    #
    #     num_batches = int(24000 / self.batch_size)
    #     if batch in np.linspace(0, num_batches, 10, endpoint=False, dtype='uint32'):
    #         # pass
    #         layer = [l for l in self.model.layers if l.name == self.layer_name][0]
    #         get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
    #         for batch_i, (batch_input, batch_label) in enumerate(self.inputFn):
    #             if batch_i == batch:
    #                 batch_output_layer = get_layer_output(batch_input)
    #                 # print('Batch {} | Input to FC block: '.format(batch), batch_output_layer)
    #                 with self.summaryWriter.as_default():
    #                     tf.summary.histogram(data=tf.convert_to_tensor(batch_output_layer, dtype=tf.float32), name='{}_output'.format(self.layer_name), step=batch, buckets=30, description='aaa')

    def get_data(self):

        if self.ensemble:
            models_in_ensemble = [l for l in self.model.layers if 'model' in l.name]
            data = []
            for model in models_in_ensemble:
                data_model = []
                layer = [l for l in model.layers if l.name == self.layer_name][0]
                get_layer_output = tf.keras.backend.function(inputs=model.input, outputs=layer.output)
                for batch_i, (batch_input, batch_label) in enumerate(self.input_fn):
                    batch_output_layer = get_layer_output(batch_input)
                    data_batch = np.concatenate([batch_output_layer, np.expand_dims(batch_label.numpy(), axis=1)],
                                                axis=1)
                    data_model.append(data_batch)
                    if len(data_model) == self.num_batches:
                        break
                data.append(data_model)
        else:
            layer = [l for l in self.model.layers if l.name == self.layer_name][0]
            get_layer_output = tf.keras.backend.function(inputs=self.model.input, outputs=layer.output)
            data = []
            for batch_i, (batch_input, batch_label) in enumerate(self.input_fn):
                batch_output_layer = get_layer_output(batch_input)
                data_batch = np.concatenate([batch_output_layer, np.expand_dims(batch_label.numpy(), axis=1)],
                                                axis=1)
                data.append(data_batch)
                if len(data) == self.num_batches:
                    break

        return data

    def write_to_csv(self, data):

        if self.ensemble:
            for model_i, model_data in enumerate(data):
                model_csv_fp = self.csv_fp.parent / f'{self.csv_fp.stem}_model{model_i + 1}.csv'
                for batch_data in model_data:
                    data_df = pd.DataFrame(batch_data)
                    if model_csv_fp.exists():
                        data_df.to_csv(model_csv_fp, index=False, mode='a', header=None)
                    else:
                        data_df.to_csv(model_csv_fp, index=False, mode='w')
        else:
            for batch_data in data:
                data_df = pd.DataFrame(batch_data)
                if self.csv_fp.exists():
                    data_df.to_csv(self.csv_fp, index=False, mode='a', header=None)
                else:
                    data_df.to_csv(self.csv_fp, index=False, mode='w')

    # def on_predict_end(self, logs=None):
    #
    #     data = self.get_data()
    #     with self.summary_writer.as_default():
    #
    #         tf.summary.histogram(data=tf.convert_to_tensor([data], dtype=tf.float32),
    #                              name=self.layer_name,
    #                              step=1,
    #                              buckets=self.buckets,
    #                              description=self.description
    #                              )

    def on_test_end(self, logs=None):

        data = self.get_data()

        self.write_to_csv(data)

        # with self.summary_writer.as_default():
        #     tf.summary.histogram(data=tf.convert_to_tensor([np.ndarray.flatten(np.array(data))], dtype=tf.float32),
        #                          name=self.layer_name,
        #                          step=1,
        #                          buckets=self.buckets,
        #                          description=self.description
        #                          )

    def on_epoch_end(self, epoch, logs=None):
        """ Write to a summary the output of a given layer at the end of each epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        data = self.get_data()

        with self.summary_writer.as_default():

            tf.summary.histogram(data=tf.convert_to_tensor([data], dtype=tf.float32),
                                 name=self.layer_name,
                                 step=epoch,
                                 buckets=self.buckets,
                                 description=self.description
                                 )


def print_metrics(model_id, res, datasets, ep_idx, metrics_names, prec_at_top):
    """ Print results.

    :param model_id: int, model ID
    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, dataset names
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :return:
    """

    print('#' * 100)
    print('Model {}'.format(model_id))
    print('Performance on epoch ({})'.format(ep_idx + 1))
    for dataset in datasets:
        print(dataset)
        for metric in metrics_names:
            if not any([metric_arr in metric for metric_arr in ['prec_thr', 'rec_thr', 'fn', 'fp', 'tn', 'tp']]):

                if dataset == 'train':
                    print('{}: {}\n'.format(metric, res['{}'.format(metric)][-1]))
                elif dataset == 'test':
                    print('{}: {}\n'.format(metric, res['test_{}'.format(metric)]))
                elif dataset == 'val':
                    print('{}: {}\n'.format(metric, res['val_{}'.format(metric)][-1]))

        if prec_at_top is not None and dataset in prec_at_top:
            for k in prec_at_top[dataset]:
                print('{}: {}\n'.format('{}_precision_at_{}'.format(dataset, k),
                                        res['{}_precision_at_{}'.format(dataset, k)]))

    print('#' * 100)


def save_metrics_to_file(model_dir_sub, res, datasets, ep_idx, metrics_names, prec_at_top=None):
    """ Write results to a txt file.

    :param model_dir_sub: std, model directory
    :param res: dict, loss and metric values for the different datasets
    :param datasets: list, datasets names
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :param metrics_names: list, metrics and losses names
    :param prec_at_top: dict, top-k values for different datasets
    :return:
    """

    with open(model_dir_sub / 'results.txt', 'w') as res_file:

        res_file.write(f'Performance metrics at epoch {ep_idx + 1} \n')

        for dataset in datasets:

            res_file.write(f'Dataset: {dataset}\n')

            for metric in metrics_names:
                if not any([metric_arr in metric for metric_arr in ['prec_thr', 'rec_thr', 'fn', 'fp', 'tn', 'tp']]):

                    if dataset == 'train':
                        res_file.write('{}: {}\n'.format(metric, res['{}'.format(metric)][-1]))

                    elif dataset == 'test':
                        res_file.write('{}: {}\n'.format(metric, res['test_{}'.format(metric)]))

                    elif dataset == 'val':
                        res_file.write('{}: {}\n'.format(metric, res['val_{}'.format(metric)][-1]))

            if prec_at_top is not None and dataset in prec_at_top:
                for k in prec_at_top[dataset]:
                    res_file.write('{}: {}\n'.format('{}_precision_at_{}'.format(dataset, k),
                                                     res['{}_precision_at_{}'.format(dataset, k)]))

            res_file.write('\n')

        res_file.write(f'{"-" * 100}')

        res_file.write('\n')


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


def plot_metric_from_res_file(res, save_path):
    """ Plot loss/metric from results NumPy file.

    :param res: dict, keys are loss and metrics on the training, validation and test set
    :param save_path: str, filepath used to save the plots figure
    :return:
    """

    f, ax = plt.subplots()
    for metric_name, metric_vals in res.items():
        ax.plot(metric_vals['epochs'], metric_vals['values'], label=metric_name)
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch Number')
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


def plot_prec_rec_roc_auc_pr_auc(res, epochs, ep_idx, save_path):
    """  Plot precision, recall, roc auc, pr auc curves for the validation and test sets.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set)
    :param epochs: Numpy array, epochs
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :param save_path: str, filepath used to save the plots figure
    :return:
    """

    f, ax = plt.subplots()
    ax.plot(epochs, res['val_precision'], label='Val Precision')
    ax.plot(epochs, res['val_recall'], label='Val Recall')
    ax.plot(epochs, res['val_auc_roc'], label='Val ROC AUC')
    ax.plot(epochs, res['val_auc_pr'], label='Val PR AUC')
    ax.scatter(epochs[ep_idx], res['test_precision'], label='Test Precision')
    ax.scatter(epochs[ep_idx], res['test_recall'], label='Test Recall')
    ax.scatter(epochs[ep_idx], res['test_auc_roc'], label='Test ROC AUC')
    ax.scatter(epochs[ep_idx], res['test_auc_pr'], label='Test PR AUC')
    ax.grid(True)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    ax.set_xlim([0, epochs[-1] + 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.set_title('Evaluation Metrics\nVal/Test')
    # ax[1].legend(loc="lower right")
    # f.suptitle('Epochs = {:.0f}'.format(res['epochs'][-1]))
    # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
    f.savefig(save_path)
    plt.close()


def plot_pr_curve(res, ep_idx, save_path):
    """ Plot PR curve.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set)
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :param model_id: int, identifies the model being used
    :param save_path: str, filepath used to save the plots figure
    :return:
    """

    f, ax = plt.subplots()
    if 'val_rec_thr' in res:
        ax.plot(res['val_rec_thr'][ep_idx], res['val_prec_thr'][ep_idx],
                label='Val (AUC={:.3f})'.format(res['val_auc_pr'][ep_idx]), color='r')
    if 'test_rec_thr' in res:
        ax.plot(res['test_rec_thr'], res['test_prec_thr'],
                label='Test (AUC={:.3f})'.format(res['test_auc_pr']), color='k')
    if 'rec_thr' in res:
        ax.plot(res['rec_thr'][ep_idx], res['prec_thr'][ep_idx],
                label='Train (AUC={:.3f})'.format(res['auc_pr'][ep_idx]), color='b')
    # ax.scatter(res['val_rec_thr'][ep_idx],
    #            res['val_prec_thr'][ep_idx], c='r')
    # ax.scatter(res['test_rec_thr'],
    #            res['test_prec_thr'], c='k')
    # ax.scatter(res['rec_thr'][ep_idx],
    #            res['prec_thr'][ep_idx], c='b')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.grid(True)
    ax.legend(loc='lower left')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_title('Precision Recall curve')
    f.savefig(save_path)
    plt.close()


def plot_roc(res, ep_idx, datasets, save_path):
    """ Plot ROC.

    :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
    for the test set)
    :param ep_idx: idx of the epoch in which the test set was evaluated
    :param datasets: list, datasets for which to plot ROC
    :param save_path: str, filepath used to save the plots figure
    :return:
    """

    # count number of samples per class to compute TPR and FPR
    num_samples_per_class = {dataset: {'positive': 0, 'negative': 0} for dataset in datasets}

    # plot roc
    f, ax = plt.subplots()
    if 'train' in datasets:
        num_samples_per_class['train'] = {'positive': res['tp'][0][0] + res['fn'][0][0],
                                          'negative': res['fp'][0][0] + res['tn'][0][0]}
        ax.plot(res['fp'][ep_idx] / num_samples_per_class['train']['negative'],
                res['tp'][ep_idx] / num_samples_per_class['train']['positive'], 'b',
                label='Train (AUC={:.3f})'.format(res['auc_roc'][ep_idx]))
    if 'val' in datasets:
        ax.plot(res['val_fp'][ep_idx] / num_samples_per_class['val']['negative'],
                res['val_tp'][ep_idx] / num_samples_per_class['val']['positive'], 'r',
                label='Val (AUC={:.3f})'.format(res['val_auc_roc'][ep_idx]))
        num_samples_per_class['val'] = {'positive': res['val_tp'][0][0] + res['val_fn'][0][0],
                                        'negative': res['val_fp'][0][0] + res['val_tn'][0][0]}
    if 'test' in datasets:
        num_samples_per_class['test'] = {'positive': res['test_tp'][0] + res['test_fn'][0],
                                         'negative': res['test_fp'][0] + res['test_tn'][0]}
        ax.plot(res['test_fp'] / num_samples_per_class['test']['negative'],
                res['test_tp'] / num_samples_per_class['test']['positive'], 'k',
                label='Test (AUC={:.3f})'.format(res['test_auc_roc']))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.grid(True)
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    f.savefig(save_path)
    plt.close()


class PredictDuringFitCallback(tf.keras.callbacks.Callback):
    """
    Callback for predicting scores at the end of the epoch for different data sets.
    """
    def __init__(self, dataset_fps, root_save_dir, batch_size, label_map, features_set, multi_class, md_features,
                 feature_map, data_to_tbl, verbose=False):
        """ Initialize callback.

        Args:
            dataset_fps: dict, each data set key maps to a list of TFRecord file paths
            root_save_dir: Path, root saving directory for the model
            batch_size: int, batch size for inference
            label_map: dict, map between category and label id
            features_set: dict, features used by the model
            multi_class: bool, if True running multiclassification setting
            md_features: bool, if True features with dimension >= 2 are being used
            feature_map: dict, map between TFRecord feature name and model feature name
            data_to_tbl: dict, each data set key maps to dict with features to be integrated in the score table such as examples' ids
            verbose:
        """

        super(PredictDuringFitCallback, self).__init__()
        self.dataset_fps = dataset_fps
        self.save_dir = root_save_dir / 'scores_per_epoch'
        self.save_dir.mkdir()
        self.batch_size = batch_size
        self.label_map = label_map
        self.features_set = features_set
        self.multi_class = multi_class
        self.md_features = md_features
        self.feature_map = feature_map
        self.data_to_tbl = data_to_tbl
        self.verbose = verbose

    def create_score_table(self, scores, data_to_tbl):
        """ Create score table from scores output by the model.

        Args:
            scores: NumPy array, scores for examples
            data_to_tbl: dict, data to display in the scores table besides scores.

        Returns:

        """

        if not self.multi_class:
            data_to_tbl['score'] = scores.ravel()
            # data['predicted class'] = scores_classification[dataset].ravel()
        else:
            for class_label, label_id in self.label_map.items():
                data_to_tbl[f'score_{class_label}'] = scores[:, label_id]
            # data['predicted class'] = scores_classification[dataset]

        data_df = pd.DataFrame(data_to_tbl)

        # sort in descending order of output
        if not self.multi_class:
            data_df.sort_values(by='score', ascending=False, inplace=True)

        return data_df

    def on_epoch_end(self, epoch, logs=None):
        """ Write scores by running inference with the model at the end of the epoch.

        :param epoch: int, epoch
        :param logs: dict, logs
        :return:
        """

        for dataset, fps in self.dataset_fps.items():
            predict_input_fn = InputFn(
                file_paths=fps,
                batch_size=self.batch_size,
                mode='PREDICT',
                label_map=self.label_map,
                features_set=self.features_set,
                multiclass=self.multi_class,
                use_transformer=self.md_features,
                feature_map=self.feature_map
            )

            scores = self.model.predict(
                predict_input_fn(),
                batch_size=None,
                verbose=self.verbose,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
            )

            tbl = self.create_score_table(scores, self.data_to_tbl[dataset])

            tbl.to_csv(self.save_dir / f'scores_{dataset}set_epoch{epoch}.csv', index=False)

"""
Perform inference using an ensemble of TensorFlow Estimator models. Set up for single model inference in parallel.

TODO: make draw_plots function compatible with inference
"""

# 3rd party
import os
import multiprocessing
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve, auc
import pandas as pd

# local
import paths
from src.old.estimator_util import get_data_from_tfrecord_kepler, CNN1dPlanetFinderv1
from src.old import predict_single_model

if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_plots(res, save_path, output_cl):
    """ Plot ROC and PR curves.

    :param res: dict, each key is a specific dataset ('train', 'val', ...) and the values are dicts that contain
    metrics for each dataset
    :param save_path: str, path to save directory
    :param output_cl: dict, each key is a specific dataset ('train', 'val', ...) and the values are dicts that contain
    the scores for each class ('PC', 'non-PC', for example)
    :return:
    """

    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set', 'predict': 'Predict set'}

    # ROC and PR curves
    for dataset in res:
        f = plt.figure(figsize=(9, 6))
        lw = 2
        ax = f.add_subplot(111, label='PR ROC')
        ax.plot(res[dataset]['Rec thr'], res[dataset]['Prec thr'], color='darkorange', lw=lw,
                label='PR ROC curve (area = %0.3f)' % res[dataset]['PR AUC'])
        # ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xticks(np.arange(0, 1.05, 0.05))
        ax.set_yticks(np.arange(0, 1.05, 0.05))
        ax.legend(loc="lower right")
        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax2 = f.add_subplot(111, label='AUC ROC', frame_on=False)
        ax2.plot(res[dataset]['FPR'], res[dataset]['TPR'], color='darkorange', lw=lw, linestyle='--',
                 label='AUC ROC curve (area = %0.3f)' % res[dataset]['ROC AUC'])
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

        f.suptitle('PR/ROC Curves - {}'.format(dataset_names[dataset]))
        f.savefig(os.path.join(pathsaveres, 'pr_roc_{}.svg'.format(dataset)))
        plt.close()

    # plot histogram of the class distribution as a function of the predicted output
    bins = np.linspace(0, 1, 11, True)
    for dataset in output_cl:

        hist, bin_edges = {}, {}
        for class_label in output_cl[dataset]:
            counts_cl = list(np.histogram(output_cl[dataset][class_label], bins, density=False, range=(0, 1)))
            counts_cl[0] = counts_cl[0] / max(len(output_cl[dataset][class_label]), 1e-7)
            hist[class_label] = counts_cl[0]
            bin_edges[class_label] = counts_cl[1]

        bins_multicl = np.linspace(0, 1, len(output_cl[dataset]) * 10 + 1, True)
        bin_width = bins_multicl[1] - bins_multicl[0]
        bins_cl = {}
        for i, class_label in enumerate(output_cl[dataset]):
            bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                    for idx in range(i, len(bins_multicl) - 1, len(output_cl[dataset]))]

        f, ax = plt.subplots()
        for class_label in output_cl[dataset]:
            ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
        if dataset == 'predict':
            ax.set_ylabel('Dataset fraction')
        else:
            ax.set_ylabel('Class fraction')
        ax.set_xlabel('Predicted output')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Output distribution - {}'.format(dataset_names[dataset]))
        ax.set_xticks(np.linspace(0, 1, 11, True))
        if dataset != 'predict':
            ax.legend()
        plt.savefig(os.path.join(save_path, 'class_predoutput_distribution_{}.svg'.format(dataset)))
        plt.close()


def main(model_dir, data_dir, num_processes, base_model, kp_dict, res_dir, datasets, threshold=0.5, fields=None,
         filter_data=None, inference_only=False, generate_csv_pred=False, sess_config=None, proc_to_gpu_mapping=None):
    """ Test single model/ensemble of models.

    :param model_dir: str, directory with saved models
    :param data_dir: str, data directory with tfrecords
    :param num_processes: int, number of processes spawn in parallel
    :param kp_dict: dict, each key is a Kepler ID and the value is the Kepler magnitude (Kp) of the star
    :param res_dir: str, save directory
    :param datasets: list, datasets in which the model(s) is(are) applied to. The datasets names should be strings that
    match a part of the tfrecord filename - 'train', 'val', 'test', 'predict'
    :param threshold: float, classification threshold
    :param fields: additional fields to be extracted from the tfrecords. If generate_csv_pred is True, these fields
    are also written to the csv file
    :param filter_data: dict, containing as keys the names of the datasets. Each value is a dict containing as keys the
    elements of data_fields or a subset, which are used to filter the examples. For 'label', 'kepid' and 'tce_n' the
    values should be a list; for the other data_fields, it should be a two element list that defines the interval of
    acceptable values
    :param inference_only: bool, if True the labels are not extracted from the tfrecords
    :param generate_csv_pred: bool, if True a csv file is generated per dataset containing the ranked model(ensemble)
    outputs and predicted classes for each example in the dataset. If fields is not None, then the values for those
    fields will also be written to the csv file
    :param sess_config:
    :param proc_to_gpu_mapping: list, maps processes to GPUs
    :return:
    """

    if filter_data is None:
        filter_data = {dataset: None for dataset in datasets}

    # get models' paths
    model_filenames = [model_dir + '/' + file for file in os.listdir(model_dir)]

    label_map = np.load('{}/config.npy'.format(model_filenames[0]), allow_pickle=True).item()['label_map']

    # get labels for each dataset
    if fields is None:
        fields = []

    if not inference_only:
        if 'label' not in fields:
            fields += ['label']

    data = {dataset: {field: [] for field in fields} for dataset in datasets}

    tfrec_files = [file for file in os.listdir(tfrec_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # find which dataset the TFRecord is from
        dataset = tfrec_file.split('-')[0]

        aux = get_data_from_tfrecord_kepler(os.path.join(tfrec_dir, tfrec_file), fields, label_map,
                                            filt=filter_data[dataset], coupled=True)

        for field in aux:
            data[dataset][field].extend(aux[field])

    # converting label array to numpy array
    if 'label' in data[dataset]:
        for dataset in datasets:
            data[dataset]['label'] = np.array(data[dataset]['label'], dtype='uint8')

    args_predict = (data_dir, base_model, datasets, sess_config, proc_to_gpu_mapping)

    tf.logging.info("Launching {} processes".format(num_processes))

    # create a pool of runConfig.numProcesses child processes
    pool = multiprocessing.Pool(num_processes)
    # run each one in parallel in a different CPU
    async_results = [pool.apply_async(predict_single_model.predict, (model_i, model_filename) + args_predict)
                     for model_i, model_filename in enumerate(model_filenames)]
    pool.close()

    # get output from each child process
    # instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
    predictions_models = [async_result.get() for async_result in async_results]

    # TODO: make it compatible with multiclass classification case or output size greater than 1
    predictions_ensemble = {dataset: None for dataset in datasets}
    for dataset in datasets:
        predictions_models_for_dataset = [prediction_model[dataset] for prediction_model in predictions_models]
        predictions_ensemble[dataset] = np.mean(predictions_models_for_dataset, axis=0)

    # select only indexes of interest that were not filtered out
    for dataset in predictions_ensemble:
        if 'selected_idxs' in data[dataset]:
            print('Filtering predictions for dataset {}'.format(dataset))
            predictions_ensemble[dataset] = predictions_ensemble[dataset][data[dataset]['selected_idxs']]

    # save results in a numpy file
    print('Saving predicted output to a numpy file {}...'.format(res_dir + 'predictions_per_dataset'))
    np.save(res_dir + 'predictions_per_dataset', predictions_ensemble)

    # TODO: implemented only for labeled data
    # sort predictions per class based on ground truth labels
    if not inference_only:
        output_cl = {dataset: {} for dataset in datasets}
        for dataset in output_cl:
            # map_labels
            for class_label in label_map:

                output_cl[dataset][class_label] = predictions_ensemble[dataset][
                    np.where(data[dataset]['label'] == label_map[class_label])]

                # if class_label == 'AFP':
                #     continue
                # elif class_label == 'NTP':
                #     output_cl[dataset]['NTP+AFP'] = \
                #         predictions_dataset[dataset][np.where(data[dataset]['label'] ==
                #                                               config['label_map'][class_label])]
                # else:
                #     output_cl[dataset][class_label] = \
                #         predictions_dataset[dataset][np.where(data[dataset]['label'] ==
                #                                               config['label_map'][class_label])]

    # dict with performance metrics
    res = {dataset: None for dataset in datasets if dataset != 'predict'}
    # dict with classification predictions
    pred_classification = {dataset: np.zeros(predictions_ensemble[dataset].shape, dtype='uint8') for dataset in datasets}
    for dataset in res:

        # threshold for classification
        # TODO: again, not suited for output size larger than 1 or multiclass classification
        pred_classification[dataset][predictions_ensemble[dataset] >= threshold] = 1

        if not inference_only:

            # compute and save performance metrics for the ensemble
            acc = accuracy_score(data[dataset]['label'], pred_classification[dataset])

            # TODO: expecting binary classification
            prec = precision_score(data[dataset]['label'], pred_classification[dataset], average='binary')
            rec = recall_score(data[dataset]['label'], pred_classification[dataset], average='binary')

            # if in multiclass classification, macro average does not take into account label imbalance
            # in the multiclass case, the scores must sum to 1 [nsamplesxnclasses]
            roc_auc = roc_auc_score(data[dataset]['label'], pred_classification[dataset], average='macro')
            avg_prec = average_precision_score(data[dataset]['label'], pred_classification[dataset], average='macro')

            # functions only implemented for the binary classification task
            fpr, tpr, _ = roc_curve(data[dataset]['label'], predictions_ensemble[dataset])
            preroc, recroc, _ = precision_recall_curve(data[dataset]['label'], predictions_ensemble[dataset])

            # general function used to compute an AUC
            pr_auc = auc(recroc, preroc)

            res[dataset] = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec,
                            'Threshold': threshold, 'Number of models': len(model_filenames), 'PR AUC': pr_auc,
                            'FPR': fpr, 'TPR': tpr, 'Prec thr': preroc, 'Rec thr': recroc, 'Avg Precision': avg_prec}

    # save results in a numpy file
    if not inference_only:
        print('Saving metrics to a numpy file {}...'.format(res_dir + 'res_eval.npy'))
        np.save(res_dir + 'res_eval', res)

        print('Plotting ROC and PR AUCs...')
        # draw evaluation plots
        draw_plots(res, res_dir, output_cl)

        print('Saving metrics to a {}...'.format(res_dir + "res_eval.txt"))
        # write results to a txt file
        with open(res_dir + "res_eval.txt", "a") as res_file:
            res_file.write('{} Performance ensemble (nmodels={}) {}\n'.format('#' * 10, len(model_filenames), '#' * 10))
            for dataset in res:
                res_file.write('Dataset: {}\n'.format(dataset))
                for metric in res[dataset]:
                    if metric not in ['Prec thr', 'Rec thr', 'TPR', 'FPR']:
                        res_file.write('{}: {}\n'.format(metric, res[dataset][metric]))
                res_file.write('\n')

        print('#' * 100)
        print('Performance ensemble (nmodels={})'.format(len(model_filenames)))
        for dataset in res:
            print(dataset)
            for metric in res[dataset]:
                if metric not in ['Prec thr', 'Rec thr', 'TPR', 'FPR']:
                    print('{}: {}'.format(metric, res[dataset][metric]))
        print('#' * 100)

    if generate_csv_pred:

        print('Generating csv file(s) with ranking(s)...')

        # add predictions to the data dict
        # TODO: make this compatible with multiclass classification
        for dataset in datasets:
            data[dataset]['output'] = predictions_ensemble[dataset]
            data[dataset]['predicted class'] = pred_classification[dataset]

            # # add Kepler magnitude of the target star
            # print('adding Kp to dataset {}'.format(dataset))
            # data[dataset]['Kp'] = [kp_dict[kepid] for kepid in data[dataset]['kepid']]

        # write results to a txt file
        for dataset in datasets:
            print('Saving ranked predictions in dataset {}'
                  ' to {}...'.format(dataset, res_dir + "ranked_predictions_{}".format(dataset)))
            data_df = pd.DataFrame(data[dataset])

            # sort in descending order of output
            data_df.sort_values(by='output', ascending=False, inplace=True)
            data_df.to_csv(res_dir + "ranked_predictions_{}set".format(dataset), index=False)


if __name__ == "__main__":

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    ######### SCRIPT PARAMETERS #############################################

    # study folder name
    study = ''

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = CNN1dPlanetFinderv1

    # assuming 1 model per process; if None, number of processes equals number of models in the model root directory
    num_processes = None

    # GPU options
    config_sess = tf.ConfigProto(log_device_placement=False, gpu_options=None)
    # if true, the allocator does not pre - allocate the entire specified GPU memory region, instead starting small and
    # growing as needed.
    config_sess.gpu_options.allow_growth = True
    # A value between 0 and 1 that indicates what fraction of the available GPU memory to pre-allocate for each process.
    # 1 means to pre-allocate all of the GPU memory, 0.5 means the process allocates ~50% of the available GPU memory.
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
    number_models_per_gpu = np.floor(1 / config_sess.gpu_options.per_process_gpu_memory_fraction)
    num_gpus = [1]  # number of GPUs in each node; none to run models on the CPU(s)

    # tfrecord files directory
    tfrec_dir = os.path.join(paths.path_tfrecs,
                             'Kepler/tfrecordkeplerdr25_flux-centroid_selfnormalized-oddeven'
                             '_nonwhitened_gapped_2001-201')

    kp_dict = None  # np.load('/data5/tess_project/Data/Ephemeris_tables/kp_KSOP2536.npy').item()

    # datasets used; choose from 'train', 'val', 'test', 'predict'
    datasets = ['train', 'val', 'test']
    # datasets = ['predict']

    # fields to be extracted from the tfrecords
    # set to None if not adding other fields to
    # fields = ['kepid', 'label', 'MES', 'tce_period', 'tce_duration', 'epoch']
    fields = ['kepid', 'label', 'tce_n', 'tce_period', 'tce_duration', 'epoch', 'original label']

    # perform only inference
    inference_only = False

    # generate prediction ranking when inferencing
    generate_csv_pred = True

    threshold = 0.5  # threshold on binary classification
    multi_class = False  # multiclass classification
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    # set to None to not filter any data in the datasets
    filter_data = None  # np.load('/data5/tess_project/Data/tfrecords/filter_datasets/cmmn_kepids_spline-whitened.npy').item()

    ######### SCRIPT PARAMETERS ###############################################

    # path to trained models' weights for the selected config
    models_path = os.path.join(paths.pathtrainedmodels, study, 'models')

    # path to save results
    pathsaveres = os.path.join(paths.pathsaveres_get_pcprobs, study)

    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    if num_processes is None:  # it is set to the number of models in the model root directory; 1 model per process
        num_processes = len(os.listdir(models_path))

    if num_gpus is None:
        proc_to_gpu_mapping = None
    else:
        proc_to_gpu_mapping = []
        for proc_i in range(num_processes):
            proc_to_gpu_mapping += [str(gpu_id) for gpu_id in np.repeat(np.arange(num_gpus[proc_i]), number_models_per_gpu)]

    main(model_dir=models_path,
         data_dir=tfrec_dir,
         num_processes=num_processes,
         base_model=BaseModel,
         kp_dict=kp_dict,
         res_dir=pathsaveres,
         datasets=datasets,
         threshold=threshold,
         fields=fields,
         filter_data=filter_data,
         inference_only=inference_only,
         generate_csv_pred=generate_csv_pred,
         sess_config=config_sess,
         proc_to_gpu_mapping=proc_to_gpu_mapping)

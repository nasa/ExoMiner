"""
Test ensemble of models trained using the best configuration obtained in a hyperparameter optimization study.

TODO: add multiprocessing option, maybe from inside Python, but that would only work internally to the node; other
    option would be to have two scripts: one that tests the models individually, the other that gathers their
    predictions into the ensemble and generates the results for it.
    does the code work for only one model?
    load config from json file in the model's folder
"""

# 3rd party
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve
import pandas as pd

# local
from src.estimator_util import InputFn, ModelFn, CNN1dModel, get_data_from_tfrecord
import src.config
import src_hpo.utils_hpo as utils_hpo
import paths
import baseline_configs

if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_plots(res, save_path, output_cl):
    """ Plot ROC and PR curves.

    :param res:
    :param save_path: str, path to save directory
    :param output_cl:
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
        f.savefig(pathsaveres + 'pr_roc_{}.png'.format(dataset))
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
        ax.set_title('Output distribution')
        ax.set_xticks(np.linspace(0, 1, 11, True))
        if dataset != 'predict':
            ax.legend()
        ax.set_title(dataset_names[dataset])
        plt.savefig(save_path + 'class_predoutput_distribution_{}.png'.format(dataset))
        plt.close()


def main(config, model_dir, data_dir, res_dir, datasets, threshold=0.5, fields=None, filter_data=None,
         inference_only=False, generate_csv_pred=False):
    """ Test single model/ensemble of models.

    :param config: dict, model and dataset configurations
    :param model_dir: str, directory with saved models
    :param data_dir: str, data directory with tfrecords
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
    fields will also be written to the csv file.
    :return:
    """

    if filter_data is None:
        filter_data = {dataset: None for dataset in datasets}

    # get models' paths
    model_filenames = [model_dir + '/' + file for file in os.listdir(model_dir)]

    # get labels for each dataset
    if fields is None:
        fields = []

    if not inference_only:
        if 'label' not in fields:
            fields += ['label']

    data = {dataset: {field: [] for field in fields} for dataset in datasets}
    for tfrec_file in os.listdir(tfrec_dir):

        dataset_idx = np.where([dataset in tfrec_file for dataset in datasets])[0][0]
        dataset = datasets[dataset_idx]

        aux = get_data_from_tfrecord(os.path.join(tfrec_dir, tfrec_file), fields, config['label_map'],
                                     filt=filter_data[dataset])

        for field in aux:
            data[dataset][field].extend(aux[field])

    if 'label' in data[dataset]:
        for dataset in datasets:
            data[dataset]['label'] = np.array(data[dataset]['label'], dtype='uint8')

    # predict on given datasets
    predictions_dataset = {dataset: [] for dataset in datasets}
    for dataset in predictions_dataset:

        for i, model_filename in enumerate(model_filenames):
            print('Predicting in dataset %s for model %i in %s' % (dataset, i + 1, model_filename))

            # REMOVE CONFIG FROM THE ARGUMENTS OF THE FUNCTION
            # config = np.load('{}/config.npy'.format(model_filename))
            # features_set = np.load('{}/features_set.npy'.format(model_filename))

            predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*', batch_size=config['batch_size'],
                                       mode=tf.estimator.ModeKeys.PREDICT, label_map=config['label_map'],
                                       centr_flag=config['centr_flag'])
            # predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*', batch_size=config['batch_size'],
            #                            mode=tf.estimator.ModeKeys.PREDICT, label_map=config['label_map'],
            #                            features_set=features_set)

            config_sess = tf.ConfigProto(log_device_placement=False)

            estimator = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                               config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                             session_config=config_sess),
                                               model_dir=model_filename)

            prediction_lst = []
            for predictions in estimator.predict(predict_input_fn):
                assert len(predictions) == 1
                prediction_lst.append(predictions[0])

            predictions_dataset[dataset].append(prediction_lst)

        # average across models
        predictions_dataset[dataset] = np.mean(predictions_dataset[dataset], axis=0)

    # select only indexes of interest
    for dataset in predictions_dataset:
        if 'selected_idxs' in data[dataset]:
            predictions_dataset[dataset] = predictions_dataset[dataset][data[dataset]['selected_idxs']]
            print(predictions_dataset[dataset].shape, dataset)

    # save results in a numpy file
    print('Saving predicted output to a numpy file {}...'.format(res_dir + 'predictions_per_dataset'))
    np.save(res_dir + 'predictions_per_dataset', predictions_dataset)

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in datasets}
    for dataset in output_cl:
        # map_labels
        for class_label in config['label_map']:

            if class_label == 'AFP':
                continue
            elif class_label == 'NTP':
                output_cl[dataset]['NTP+AFP'] = \
                    predictions_dataset[dataset][np.where(data[dataset]['label'] == config['label_map'][class_label])]
            else:
                output_cl[dataset][class_label] = \
                    predictions_dataset[dataset][np.where(data[dataset]['label'] == config['label_map'][class_label])]

    # dict with performance metrics
    res = {dataset: None for dataset in datasets if dataset != 'predict'}
    # dict with classification predictions
    pred_classification = {dataset: np.zeros(predictions_dataset[dataset].shape, dtype='uint8') for dataset in datasets}
    for dataset in datasets:
        # threshold for classification
        pred_classification[dataset][predictions_dataset[dataset] >= threshold] = 1

        if not inference_only:
            # compute and save performance metrics for the ensemble
            acc = accuracy_score(data[dataset]['label'], pred_classification[dataset])
            roc_auc = roc_auc_score(data[dataset]['label'], pred_classification[dataset], average='macro')
            pr_auc = average_precision_score(data[dataset]['label'], pred_classification[dataset], average='macro')
            prec = precision_score(data[dataset]['label'], pred_classification[dataset], average='binary')
            rec = recall_score(data[dataset]['label'], pred_classification[dataset], average='binary')

            fpr, tpr, _ = roc_curve(data[dataset]['label'], predictions_dataset[dataset])
            preroc, recroc, _ = precision_recall_curve(data[dataset]['label'], predictions_dataset[dataset])

            res[dataset] = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec,
                            'Threshold': threshold, 'Number of models': len(model_filenames), 'PR AUC': pr_auc,
                            'FPR': fpr, 'TPR': tpr, 'Prec thr': preroc, 'Rec thr': recroc}

    # save results in a numpy file
    if not inference_only:
        print('Saving metrics to a numpy file {}...'.format(res_dir + 'res_eval.npy'))
        np.save(res_dir + 'res_eval', res)

    print('Plotting ROC and PR AUCs...')
    # draw evaluation plots
    draw_plots(res, res_dir, output_cl)

    if not inference_only:
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

        # add predictions to the data dict
        for dataset in datasets:
            data[dataset]['output'] = predictions_dataset[dataset]
            data[dataset]['predicted class'] = pred_classification[dataset]

        # write results to a txt file
        for dataset in datasets:
            print('Saving ranked predictions in dataset {}'
                  ' to {}...'.format(dataset, res_dir + "ranked_predictions_{}".format(dataset)))
            data_df = pd.DataFrame(data[dataset])
            # sort in descending order of output
            data_df.sort_values(by='output', ascending=False, inplace=True)
            data_df.to_csv(res_dir + "ranked_predictions_{}set".format(dataset), index=False)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    # # load Shallue's best config
    # shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
    #                         'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': None,
    #                         'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
    #                         'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    ######### SCRIPT PARAMETERS #############################################

    # study folder name
    study = 'study_bohb_dr25_tcert_spline2'
    # set configuration manually, None to load it from a HPO study
    config = None

    # load test data
    tfrec_dir = paths.tfrec_dir['DR25']['spline']['TCERT']

    # datasets used; choose from 'train', 'val', 'test', 'predict'
    datasets = ['train', 'val', 'test']

    #
    fields = None  # ['kepid', 'label', 'MES', 'tce_period', 'tce_duration', 'epoch']

    # perform only inference
    inference_only = False

    # generate prediction ranking when inferencing
    generate_csv_pred = False

    threshold = 0.5  # threshold on binary classification
    multi_class = False
    use_kepler_ce = True
    centr_flag = False
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    # set to None to not filter any data in the datasets
    filter_data = None  # np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/
    # cmmn_kepids.npy').item()

    # load best config from HPO study
    if config is None:
        # res = utils_hpo.logged_results_to_HBS_result('/data5/tess_project/pedro/HPO/Run_3', '')
        res = utils_hpo.logged_results_to_HBS_result(paths.path_hpoconfigs + study,
                                                     '_{}'.format(study)
                                                     )
        # res = hpres.logged_results_to_HBS_result(paths.path_hpoconfigs + 'study_rs')

        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        config = id2config[incumbent]['config']
        # best_config = id2config[(13, 0, 7)]['config']

    print('Configuration loaded:', config)

    ######### SCRIPT PARAMETERS ###############################################

    # path to trained models' weights for the selected config
    models_path = paths.pathtrainedmodels + study + '/ES-weightedloss_300-p20_34k' + '/models'

    # path to save results
    pathsaveres = paths.pathsaveres_get_pcprobs + study + '/'

    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    # add dataset parameters
    config = src.config.add_dataset_params(tfrec_dir, satellite, multi_class, centr_flag, use_kepler_ce, config)

    # add missing parameters in hpo with default values
    config = src.config.add_default_missing_params(config=config)
    print('Configuration used: ', config)

    main(config=config,
         model_dir=models_path,
         data_dir=tfrec_dir,
         res_dir=pathsaveres,
         datasets=datasets,
         threshold=threshold,
         fields=fields,
         filter_data=filter_data,
         inference_only=inference_only,
         generate_csv_pred=generate_csv_pred)

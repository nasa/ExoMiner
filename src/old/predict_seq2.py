"""
Test ensemble of models trained using the best configuration obtained in a hyperparameter optimization study.

TODO: add multiprocessing option, maybe from inside Python, but that would only work internally to the node; other
    option would be to have two scripts: one that tests the models individually, the other that gathers their
    predictions into the ensemble and generates the results for it.
    does the code work for only one model?
    load config from json file in the model's folder
"""

# 3rd party
import sys
sys.path.append('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve, auc
import pandas as pd
import itertools

# local
# from src.estimator_util import InputFn, ModelFn, CNN1dModel, get_data_from_tfrecord
from src.old.estimator_util_bc import InputFn, ModelFn, CNN1dModel, get_data_from_tfrecord
import src.old.config
import src_hpo.utils_hpo as utils_hpo
import paths

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

    lw = 2
    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set', 'predict': 'Predict set'}

    # ROC and PR curves
    for config in res:
        for dataset in res[config]:
            f = plt.figure(figsize=(9, 6))
            ax = f.add_subplot(111, label='PR ROC')
            ax.plot(res[config][dataset]['Rec thr'], res[config][dataset]['Prec thr'], color='darkorange', lw=lw,
                    label='PR ROC curve (area = %0.3f)' % res[config][dataset]['PR AUC'])
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
            ax2.plot(res[config][dataset]['FPR'], res[config][dataset]['TPR'], color='darkorange', lw=lw,
                     linestyle='--',
                     label='AUC ROC curve (area = %0.3f)' % res[config][dataset]['ROC AUC'])
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

            f.suptitle('PR/ROC Curves - {} | Model {}'.format(dataset_names[dataset], config))
            f.savefig(pathsaveres + 'pr_roc_{}_{}.svg'.format(dataset, config))
            plt.close()

            if config == 'model2':
                f = plt.figure(figsize=(9, 6))
                ax = f.add_subplot(111, label='PR ROC_nNTP')
                ax.plot(res[config][dataset]['Rec thr_nNTP'], res[config][dataset]['Prec thr_nNTP'], color='darkorange',
                        lw=lw, label='PR ROC curve (area = %0.3f)' % res[config][dataset]['PR AUC_nNTP'])
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
                ax2.plot(res[config][dataset]['FPR_nNTP'], res[config][dataset]['TPR_nNTP'], color='darkorange', lw=lw,
                         linestyle='--',
                         label='AUC ROC curve (area = %0.3f)' % res[config][dataset]['ROC AUC_nNTP'])
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

                f.suptitle('PR/ROC nNTP Curves - {} | Model {}'.format(dataset_names[dataset], config))
                f.savefig(pathsaveres + 'pr_roc_{}_{}_nNTP.svg'.format(dataset, config))
                plt.close()

    # plot histogram of the class distribution as a function of the predicted output
    bins = np.linspace(0, 1, 11, True)
    for config in output_cl:
        for dataset in output_cl[config]:

            hist, bin_edges = {}, {}
            for class_label in output_cl[config][dataset]:
                counts_cl = list(np.histogram(output_cl[config][dataset][class_label], bins, density=False,
                                              range=(0, 1)))
                counts_cl[0] = counts_cl[0] / max(len(output_cl[config][dataset][class_label]), 1e-7)
                hist[class_label] = counts_cl[0]
                bin_edges[class_label] = counts_cl[1]

            bins_multicl = np.linspace(0, 1, len(output_cl[config][dataset]) * 10 + 1, True)
            bin_width = bins_multicl[1] - bins_multicl[0]
            bins_cl = {}
            for i, class_label in enumerate(output_cl[config][dataset]):
                bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                        for idx in range(i, len(bins_multicl) - 1, len(output_cl[config][dataset]))]

            f, ax = plt.subplots()
            for class_label in output_cl[config][dataset]:
                ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
            if dataset == 'predict':
                ax.set_ylabel('Dataset fraction')
            else:
                ax.set_ylabel('Class fraction')
            ax.set_xlabel('Predicted output')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title('Output distribution - {} | {}'.format(dataset_names[dataset], config))
            ax.set_xticks(np.linspace(0, 1, 11, True))
            if dataset != 'predict':
                ax.legend()
            plt.savefig(save_path + 'class_predoutput_distribution_{}_{}.svg'.format(dataset, config))
            plt.close()


def main(configs, model_dir, data_dir, res_dir, datasets, threshold=0.5, fields=None, filter_data=None,
         inference_only=False, generate_csv_pred=False, features_set=None):
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
    model_filenames = {}
    for config in model_dir:
        model_filenames[config] = [model_dir[config] + '/' + file for file in os.listdir(model_dir[config])]

    # get labels for each dataset
    if fields is None:
        fields = []

    if not inference_only:
        if 'label' not in fields:
            fields += ['label']

    # data = {config: {dataset: {field: [] for field in fields} for dataset in datasets} for config in configs}
    data = {dataset: {field: [] for field in fields} for dataset in datasets}

    # if data[datasets[0]] is not None:
    #     for dataset in datasets:
    #         data[dataset]['selected_idxs'] = []

    # for config in configs:

    for tfrec_file in os.listdir(tfrec_dir):

        dataset_idx = np.where([dataset in tfrec_file for dataset in datasets])[0][0]
        dataset = datasets[dataset_idx]

        aux = get_data_from_tfrecord(os.path.join(tfrec_dir, tfrec_file),
                                     fields,
                                     configs['model1']['label_map'],
                                     filt=filter_data[dataset],
                                     coupled=False)

        for field in aux:
            data[dataset][field].extend(aux[field])

    # converting label array to numpy array
    if 'label' in fields:
        # for config in configs:
        for dataset in datasets:
            data[dataset]['label'] = np.array(data[dataset]['label'], dtype='uint8')

    # converting label array to numpy array
    if 'original label' in fields:
        # for config in configs:
        for dataset in datasets:
            data[dataset]['original label'] = np.array(data[dataset]['original label'])

    # predict on given datasets
    predictions_dataset = {config: {dataset: [] for dataset in datasets} for config in configs}
    for config in configs:
        for dataset in predictions_dataset[config]:
            for i, model_filename in enumerate(model_filenames[config]):
                print('Predicting in dataset %s for model %i (%s) in %s)' % (dataset, i + 1, config, model_filename))

                # REMOVE CONFIG FROM THE ARGUMENTS OF THE FUNCTION
                # config = np.load('{}/config.npy'.format(model_filename))
                # features_set = np.load('{}/features_set.npy'.format(model_filename))

                predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*',
                                           batch_size=configs[config]['batch_size'],
                                           mode=tf.estimator.ModeKeys.PREDICT,
                                           label_map=configs[config]['label_map'],
                                           centr_flag=configs[config]['centr_flag'],
                                           features_set=features_set[config])

                config_sess = tf.ConfigProto(log_device_placement=False)

                estimator = tf.estimator.Estimator(ModelFn(CNN1dModel, configs[config]),
                                                   config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                                 session_config=config_sess),
                                                   model_dir=model_filename)

                prediction_lst = []
                for predictions in estimator.predict(predict_input_fn):
                    assert len(predictions) == 1
                    prediction_lst.append(predictions[0])

                predictions_dataset[config][dataset].append(prediction_lst)

            # average across models
            predictions_dataset[config][dataset] = np.mean(predictions_dataset[config][dataset], axis=0)

    # select only indexes of interest that were not filtered out
    for config in configs:
        for dataset in predictions_dataset[config]:
            # if 'selected_idxs' in data[config][dataset]:
            if 'selected_idxs' in data[dataset]:
                print('Filtering predictions for dataset {} ({})'.format(dataset, config))
                predictions_dataset[config][dataset] = \
                    predictions_dataset[config][dataset][data[dataset]['selected_idxs']]
                # print(predictions_dataset[dataset].shape, dataset)

    predictions_dataset['modelseq'] = {dataset: predictions_dataset['model1'][dataset] for dataset in datasets}
    for dataset in datasets:
        idxs_abvthr = np.where(predictions_dataset['model1'][dataset] >= threshold)[0]
        # kepids1 = np.array(data['model1'][dataset]['kepid'])[idxs_abvthr]
        # tce_n1 = np.array(data['model1'][dataset]['tce_n'])[idxs_abvthr]
        # ex1 = [str(el[0]) + str(el[1]) for el in zip(kepids1, tce_n1)]
        # kepids2 = np.array(data['model2'][dataset]['kepid'])
        # tce_n2 = np.array(data['model2'][dataset]['tce_n'])
        # ex2 = [str(el[0]) + str(el[1]) for el in zip(kepids2, tce_n2)]
        # _, _, ex2_idxs = np.intersect1d(ex1, ex2, assume_unique=True, return_indices=True)
        idxs_abvthr2 = np.where(predictions_dataset['model2'][dataset][idxs_abvthr] >= threshold)[0]
        print('Dataset {}:\n'
              'Number of examples classified as PC by model1 = {}\n'
              'Number of examples classified as PC by model2 (%) = {} ({})'.format(dataset,
                                                                                   len(idxs_abvthr),
                                                                                   len(idxs_abvthr2),
                                                                                   len(idxs_abvthr2) /
                                                                                   len(idxs_abvthr)))
        predictions_dataset['modelseq'][dataset][idxs_abvthr] = predictions_dataset['model2'][dataset][idxs_abvthr]

    # save results in a numpy file
    print('Saving predicted output to a numpy file {}...'.format(res_dir + 'predictions_per_dataset'))
    np.save(res_dir + 'predictions_per_dataset', predictions_dataset)

    # sort predictions per class based on ground truth labels
    output_cl = {config: {dataset: {} for dataset in datasets} for config in predictions_dataset}
    for config in output_cl:
        for dataset in output_cl[config]:
            for class_label in ['PC', 'NTP', 'AFP']:
                output_cl[config][dataset][class_label] = \
                    predictions_dataset[config][dataset][np.where(data[dataset]['original label'] == class_label)]

    # dict with performance metrics
    res = {config: {dataset: None for dataset in datasets if dataset != 'predict'} for config in predictions_dataset}
    # dict with classification predictions
    pred_classification = {config: {dataset: np.zeros(predictions_dataset[config][dataset].shape, dtype='uint8')
                                    for dataset in datasets} for config in predictions_dataset}
    for config in predictions_dataset:
        for dataset in predictions_dataset[config]:
            # threshold for classification
            pred_classification[config][dataset][predictions_dataset[config][dataset] >= threshold] = 1

            if not inference_only:
                # nclasse = len(np.unique(data[dataset]['label']))
                # compute and save performance metrics for the ensemble
                acc = accuracy_score(data[dataset]['label'], pred_classification[config][dataset])
                # if nlclasses == 2:
                prec = precision_score(data[dataset]['label'], pred_classification[config][dataset], average='binary')
                rec = recall_score(data[dataset]['label'], pred_classification[config][dataset], average='binary')

                # if in multiclass classification, macro average does not take into account label imbalance
                roc_auc = roc_auc_score(data[dataset]['label'], pred_classification[config][dataset], average='macro')
                avp_prec = average_precision_score(data[dataset]['label'], pred_classification[config][dataset],
                                                  average='macro')

                fpr, tpr, _ = roc_curve(data[dataset]['label'], predictions_dataset[config][dataset])
                preroc, recroc, _ = precision_recall_curve(data[dataset]['label'], predictions_dataset[config][dataset])

                pr_auc = auc(recroc, preroc)

                if config == 'model2':
                    idxs_nntp = np.where(data[dataset]['original label'] != 'NTP')[0]

                    accnntp = accuracy_score(data[dataset]['label'][idxs_nntp],
                                         pred_classification[config][dataset][idxs_nntp])

                    precnntp = precision_score(data[dataset]['label'][idxs_nntp],
                                               pred_classification[config][dataset][idxs_nntp], average='binary')
                    recnntp = recall_score(data[dataset]['label'][idxs_nntp],
                                           pred_classification[config][dataset][idxs_nntp], average='binary')

                    # if in multiclass classification, macro average does not take into account label imbalance
                    roc_aucnntp = roc_auc_score(data[dataset]['label'][idxs_nntp],
                                            pred_classification[config][dataset][idxs_nntp], average='macro')
                    avg_precnntp = average_precision_score(data[dataset]['label'][idxs_nntp],
                                                      pred_classification[config][dataset][idxs_nntp],
                                                      average='macro')

                    fprnntp, tprnntp, _ = roc_curve(data[dataset]['label'][idxs_nntp],
                                                    predictions_dataset[config][dataset][idxs_nntp])
                    prerocnntp, recrocnntp, _ = precision_recall_curve(data[dataset]['label'][idxs_nntp],
                                                               predictions_dataset[config][dataset][idxs_nntp])

                    pr_aucnntp = auc(recrocnntp, prerocnntp)

                # else:
                #     prec = precision_score(data[dataset]['label'], pred_classification[dataset], average='binary')
                #     rec = recall_score(data[dataset]['label'], pred_classification[dataset], average='binary')
                #
                #     roc_auc = roc_auc_score(data[dataset]['label'], pred_classification[dataset], average='macro')
                #     pr_auc = average_precision_score(data[dataset]['label'], pred_classification[dataset], average='macro')
                #     pr_auc = auc(data[dataset]['label'], pred_classification[dataset], average='macro')
                #
                #     fpr, tpr, _ = roc_curve(data[dataset]['label'], predictions_dataset[dataset])
                #     preroc, recroc, _ = precision_recall_curve(data[dataset]['label'], predictions_dataset[dataset])

                if config == 'model2':
                    res[config][dataset] = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec,
                                            'Threshold': threshold, 'Number of models': len(model_filenames[config]),
                                            'PR AUC': pr_auc, 'FPR': fpr, 'TPR': tpr, 'Prec thr': preroc,
                                            'Rec thr': recroc, 'Avg Precision': avp_prec, 'Accuracy_nNTP': accnntp,
                                            'ROC AUC_nNTP': roc_aucnntp, 'Precision_nNTP': precnntp,
                                            'Recall_nNTP': recnntp, 'PR AUC_nNTP': pr_aucnntp, 'FPR_nNTP': fprnntp,
                                            'TPR_nNTP': tprnntp, 'Prec thr_nNTP': prerocnntp,
                                            'Rec thr_nNTP': recrocnntp, 'Avg Precision_nNTP': avg_precnntp}
                else:
                    res[config][dataset] = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec,
                                            'Threshold': threshold, 'Number of models': len(model_filenames['model1']),
                                            'PR AUC': pr_auc, 'FPR': fpr, 'TPR': tpr, 'Prec thr': preroc,
                                            'Rec thr': recroc, 'Avg Precision': avp_prec}

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
            res_file.write('{} Performance ensemble (nmodels={}) {}\n'.format('#' * 10, len(model_filenames['model1']),
                                                                              '#' * 10))
            for dataset in datasets:
                res_file.write('Dataset: {}\n'.format(dataset))
                for config in res:
                    res_file.write('Model: {}\n'.format(config))
                    for metric in res[config][dataset]:
                        if metric not in ['Prec thr', 'Rec thr', 'TPR', 'FPR', 'Prec thr_nNTP', 'Rec thr_nNTP',
                                          'TPR_nNTP', 'FPR_nNTP']:
                            res_file.write('{}: {}\n'.format(metric, res[config][dataset][metric]))
                    res_file.write('\n')
                res_file.write('\n')

        print('#' * 100)
        print('Performance ensemble (nmodels={})'.format(len(model_filenames)))
        for dataset in datasets:
            print(dataset)
            for config in res:
                print(config)
                for metric in res[config][dataset]:
                    if metric not in ['Prec thr', 'Rec thr', 'TPR', 'FPR', 'Prec thr_nNTP', 'Rec thr_nNTP',
                                      'TPR_nNTP', 'FPR_nNTP']:
                        print('{}: {}'.format(metric, res[config][dataset][metric]))
        print('#' * 100)

    if generate_csv_pred:

        # add predictions to the data dict
        for config in predictions_dataset:
            for dataset in predictions_dataset[config]:
                data[dataset]['output {}'.format(config)] = predictions_dataset[config][dataset]
                data[dataset]['predicted class {}'.format(config)] = pred_classification[config][dataset]

        # write results to a txt file
        for dataset in datasets:
            print('Saving ranked predictions in dataset {}'
                  ' to {}...'.format(dataset, res_dir + "ranked_predictions_{}".format(dataset)))
            data_df = pd.DataFrame(data[dataset])
            # sort in descending order of output
            data_df.sort_values(by='output modelseq', ascending=False, inplace=True)
            data_df.to_csv(res_dir + "ranked_predictions_{}set".format(dataset), index=False)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    ######### SCRIPT PARAMETERS #############################################

    # study folder name
    study = 'flux-centroid-pc-nonpc__flux-centroid-odd_even-pc-afp'

    # # set configuration manually, None to load it from a HPO study
    # # check baseline_configs.py for some baseline/default configurations
    # config = None

    # load test data
    # tfrec_dir = {'model1': '/data5/tess_project/Data/tfrecords/dr25_koilabels/'
    #                        'tfrecord_dr25_manual_2dkeplernonwhitened_gapped_oddeven_centroid',
    #             'model2': '/data5/tess_project/Data/tfrecords/dr25_koilabels/'
    #                       'tfrecord_dr25_manual_2dkeplernonwhitened_gapped_oddeven_centroid_afp_pc'}
    tfrec_dir = '/data5/tess_project/Data/tfrecords/dr25_koilabels/' \
                'tfrecord_dr25_manual_2dkeplernonwhitened_gapped_oddeven_centroid'

    # datasets used; choose from 'train', 'val', 'test', 'predict'
    datasets = ['train', 'val', 'test']

    # features to be extracted from the datasets
    features_set = {'model1': None, 'model2': None}
    views = ['global_view', 'local_view']
    channels_centr = ['', '_centr']
    channels_oddeven = ['_odd', '_even']
    features_names = [''.join(feature_name_tuple)
                      for feature_name_tuple in itertools.product(views, channels_centr)]
    features_dim = {feature_name: 2001 if 'global' in feature_name else 201 for feature_name in features_names}
    features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    features_set['model1'] = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
                              for feature_name in features_names}

    features_names = [''.join(feature_name_tuple)
                      for feature_name_tuple in itertools.product(views, channels_centr + channels_oddeven)]
    features_dim = {feature_name: 2001 if 'global' in feature_name else 201 for feature_name in features_names}
    features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    features_set['model2'] = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
                              for feature_name in features_names}
    # # example
    # features_set = {'global_view': {'dim': 2001, 'dtype': tf.float32},
    #                 'local_view': {'dim': 201, 'dtype': tf.float32}}

    # set to None if not adding other fields to
    # fields = ['kepid', 'label', 'MES', 'tce_period', 'tce_duration', 'epoch']
    fields = ['kepid', 'label', 'tce_n', 'tce_period', 'tce_duration', 'epoch', 'original label']

    # perform only inference
    inference_only = False

    # generate prediction ranking when inferencing
    generate_csv_pred = True

    threshold = 0.5  # threshold on binary classification
    multi_class = False
    use_kepler_ce = False
    centr_flag = False
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    # set to None to not filter any data in the datasets
    filter_data = None  # np.load('/data5/tess_project/Data/tfrecords/filter_datasets/cmmn_kepids_spline-whitened.npy').item()

    # load configurations
    configs = {'model1': None, 'model2': None}

    res1 = utils_hpo.logged_results_to_HBS_result('/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/'
                                                  'Gapped_Splined_Centroid/hpo_confs/'
                                                  'bohb_dr25tcert_spline_gapped_centroid',
                                                  '_bohb_dr25tcert_spline_gapped_centroid')
    id2config = res1.get_id2config_mapping()
    incumbent = res1.get_incumbent_id()
    configs['model1'] = id2config[incumbent]['config']

    res2 = utils_hpo.logged_results_to_HBS_result('/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/'
                                                  'Gapped_Splined_OddEven_Centroid_No_NTP/hpo_confs/'
                                                  'bohb_dr25tcert_spline_gapped_centroid_no_ntp',
                                                  '_bohb_dr25tcert_spline_gapped_centroid_no_ntp')
    id2config = res2.get_id2config_mapping()
    incumbent = res2.get_incumbent_id()
    configs['model2'] = id2config[incumbent]['config']

    # # load best config from HPO study
    # if config is None:
    #     # res = utils_hpo.logged_results_to_HBS_result('/data5/tess_project/pedro/HPO/Run_3', '')
    #     res = utils_hpo.logged_results_to_HBS_result(paths.path_hpoconfigs + study,
    #                                                  '_{}'.format(study)
    #                                                  )
    #     # res = hpres.logged_results_to_HBS_result(paths.path_hpoconfigs + 'study_rs')
    #
    #     id2config = res.get_id2config_mapping()
    #     incumbent = res.get_incumbent_id()
    #     config = id2config[incumbent]['config']
    #     # select a specific config based on its ID
    #     # config = id2config[(41, 0, 0)]['config']
    #
    print('Configuration loaded:', configs)

    ######### SCRIPT PARAMETERS ###############################################

    # path to trained models' weights for the selected configs
    models_path = {'model1': '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/'
                             'Gapped_Splined_Centroid/models/'
                             'bohb_dr25tcert_spline_gapped_centroid/models',
                   'model2': '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/'
                             'Gapped_Splined_OddEven_Centroid_No_NTP/models/'
                             'bohb_dr25tcert_spline_gapped_centroid_no_ntp/models'}
    # models_path = paths.pathtrainedmodels + study + '/ES_300-p20_34k' + '/models'

    # path to save results
    pathsaveres = paths.pathsaveres_get_pcprobs + study + '/'

    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    # add dataset parameters
    for config in configs:
        configs[config] = src.old.config.add_dataset_params(tfrec_dir, satellite, multi_class, centr_flag,
                                                            use_kepler_ce, configs[config])

    # add missing parameters in hpo with default values
    for config in configs:
        configs[config] = src.old.config.add_default_missing_params(config=configs[config])

    print('Configurations used: ', configs)

    main(configs=configs,
         model_dir=models_path,
         data_dir=tfrec_dir,
         res_dir=pathsaveres,
         datasets=datasets,
         threshold=threshold,
         fields=fields,
         filter_data=filter_data,
         inference_only=inference_only,
         generate_csv_pred=generate_csv_pred,
         features_set=features_set)

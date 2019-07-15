"""
Test ensemble of models trained using the best configuration obtained in a hyperparameter optimization study.

TODO: add multiprocessing option, maybe from inside Python, but that would only work internally to the node; other
    option would be to have two scripts: one that tests the models individually, the other that gathers their
    predictions into the ensemble and generates the results for it.
"""

# 3rd party
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve

# local
from src.estimator_util import InputFn, ModelFn, CNN1dModel, get_data_from_tfrecord
import src.config
import src_hpo.utils_hpo as utils_hpo
import paths

if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_plots(res, save_path, output_cl):
    """ Plot ROC and PR curves.

    :param pathsaveres: str, path to save plots
    :param recroc: numpy array, recall values for the PR curve
    :param preroc: numpy array, precision values for the PR curve
    :param pr_auc: float, precision-recall curve AUC (aka average precision)
    :param tpr: numpy array, true positive rate
    :param fpr: numpy array, false positive rate
    :param roc_auc: float, ROC AUC
    :return:
    """

    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set'}
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
    dataset_names = {'train': 'Training set', 'val': 'Validation set', 'test': 'Test set'}
    for dataset in output_cl:

        hist, bin_edges = {}, {}
        for class_label in output_cl[dataset]:
            counts_cl = list(np.histogram(output_cl[dataset][class_label], bins, density=False, range=(0, 1)))
            counts_cl[0] = counts_cl[0] / len(output_cl[dataset][class_label])
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
        ax.set_ylabel('Class fraction')
        ax.set_xlabel('Predicted output')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Output distribution')
        ax.set_xticks(np.linspace(0, 1, 11, True))
        ax.legend()
        ax.set_title(dataset_names[dataset])
        plt.savefig(save_path + 'class_predoutput_distribution_{}.png'.format(dataset))
        plt.close()


def main(config, model_dir, data_dir, res_dir, threshold=0.5, cmmn_ids=None):
    """ Test ensemble of models.

    :param config: dict, model and dataset configurations
    :param model_dir: str, directory with saved models
    :param data_dir: str, data directory with tfrecords
    :param res_dir: str, save directory
    :param threshold: float, classification threshold
    :return:
    """

    if cmmn_ids is None:
        cmmn_ids = {dataset: None for dataset in ['train', 'val', 'test']}

    # get models' paths
    model_filenames = [model_dir + '/' + file for file in os.listdir(model_dir)]

    # get labels for each dataset
    labels = {dataset: [] for dataset in ['train', 'val', 'test']}
    selected_idxs = {dataset: [] for dataset in ['train', 'val', 'test']}
    for tfrec_file in os.listdir(tfrec_dir):
        dataset_idx = np.where([dataset in tfrec_file for dataset in ['train', 'val', 'test']])[0][0]
        dataset = ['train', 'val', 'test'][dataset_idx]

        aux = get_data_from_tfrecord(os.path.join(tfrec_dir, tfrec_file), ['labels'], config['label_map'],
                                     filt=cmmn_ids[dataset])
        labels[dataset] += aux['labels']
        if cmmn_ids[dataset] is not None:
            selected_idxs[dataset] += aux['selected_idxs']

    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in ['train', 'val', 'test']}

    # # num_afps, num_ntps, num_pcs = 0, 0, 0
    # kepid_vec, glob_vec, loc_vec, labels = [], [], [], []
    # # kepid_vec, glob_vec, loc_vec, ephem_vec, glob_centrvec, loc_centrvec, mes_vec = [], [], [], [], [], [], []
    # for file in tfrecord_filenames:
    #     record_iterator = tf.python_io.tf_record_iterator(path=file)
    #     for string_record in record_iterator:
    #         example = tf.train.Example()
    #         example.ParseFromString(string_record)
    #         kepid = example.features.feature['kepid'].int64_list.value[0]
    #         # tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
    #         label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
    #         # if label == 'AFP':
    #         #     num_afps += 1
    #         # if label == 'NTP':
    #         #     num_ntps += 1
    #         # if label == 'PC':
    #         #     num_pcs += 1
    #         # NEEDS TO BE ADAPTED FOR MULTICLASS AND TESS!!!
    #         if label in ['AFP', 'NTP']:
    #             labels.append(0)
    #         else:
    #             labels.append(1)
    #         # period = example.features.feature['tce_period'].float_list.value[0]
    #         # duration = example.features.feature['tce_duration'].float_list.value[0]
    #         # epoch = example.features.feature['tce_time0bk'].float_list.value[0]
    #         # MES = example.features.feature['mes'].float_list.value[0]
    #         # ephem_vec += [{'period': period, 'duration': duration, 'epoch': epoch}]
    #
    #         glob_view = example.features.feature['global_view'].float_list.value
    #         loc_view = example.features.feature['local_view'].float_list.value
    #         # glob_view_centr = example.features.feature['global_view_centr'].float_list.value
    #         # loc_view_centr = example.features.feature['local_view_centr'].float_list.value
    #
    #         kepid_vec.append(kepid)
    #         glob_vec += [glob_view]
    #         loc_vec += [loc_view]
    #         # glob_centrvec += [glob_view_centr]
    #         # loc_centrvec += [loc_view_centr]
    #         # mes_vec += [MES]
    #
    # # print('number of AFP, ATP, PC: ', num_afps, num_ntps, num_pcs)
    # # features = [np.array(loc_vec), np.array(glob_vec), np.array(glob_centrvec), np.array(loc_centrvec)]
    # features = [np.array(loc_vec), np.array(glob_vec)]
    # features = tuple([np.reshape(i, (i.shape[0], 1, i.shape[-1])) for i in features])

    # def input_fn():
    #     dataset = tf.data.Dataset.from_tensor_slices(features)
    #     dataset.repeat(1)
    #     dataset = dataset.map(parser)
    #
    #     return dataset
    #
    # # def parser(localview, globalview, localview_centr, globalview_centr):
    # def parser(localview, globalview):
    #     # output = {"time_series_features": {'local_view': tf.to_float(localview),
    #     #                                    'global_view': tf.to_float(globalview),
    #     #                                    'global_view_centr': tf.to_float(localview_centr),
    #     #                                    'local_view_centr': tf.to_float(globalview_centr)}}
    #     output = {"time_series_features": {'local_view': tf.to_float(localview),
    #                                        'global_view': tf.to_float(globalview)}}
    #     return output

    # predict on given datasets
    predictions_dataset = {dataset: [] for dataset in ['train', 'val', 'test']}
    for dataset in predictions_dataset:

        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*', batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT, label_map=config['label_map'],
                                   centr_flag=config['centr_flag'])
        for i, model_filename in enumerate(model_filenames):
            print('Predicting in dataset %s for model %i in %s' % (dataset, i + 1, model_filename))

            config_sess = None
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
        if len(selected_idxs[dataset]) > 0:
            predictions_dataset[dataset] = predictions_dataset[dataset][selected_idxs[dataset]]
            print(predictions_dataset[dataset].shape, dataset)

    # save results in a numpy file
    print('Saving predicted output to a numpy file...')
    np.save(res_dir + 'predictions_per_dataset', predictions_dataset)

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in ['train', 'val', 'test']}
    for dataset in output_cl:
        # map_labels
        for class_label in config['label_map']:

            if class_label == 'AFP':
                continue
            elif class_label == 'NTP':
                output_cl[dataset]['NTP+AFP'] = \
                    predictions_dataset[dataset][np.where(labels[dataset] == config['label_map'][class_label])]
            else:
                output_cl[dataset][class_label] = \
                    predictions_dataset[dataset][np.where(labels[dataset] == config['label_map'][class_label])]

    res = {dataset: None for dataset in ['train', 'val', 'test']}
    for dataset in res:
        # threshold for classification
        pred_classification = np.zeros(predictions_dataset[dataset].shape, dtype='uint8')
        pred_classification[predictions_dataset[dataset] >= threshold] = 1

        # compute and save performance metrics for the ensemble
        acc = accuracy_score(labels[dataset], pred_classification)
        roc_auc = roc_auc_score(labels[dataset], pred_classification, average='macro')
        pr_auc = average_precision_score(labels[dataset], pred_classification, average='macro')
        prec = precision_score(labels[dataset], pred_classification, average='binary')
        rec = recall_score(labels[dataset], pred_classification, average='binary')

        fpr, tpr, _ = roc_curve(labels[dataset], predictions_dataset[dataset])
        preroc, recroc, _ = precision_recall_curve(labels[dataset], predictions_dataset[dataset])

        res[dataset] = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec, 'Threshold': threshold,
                        'Number of models': len(model_filenames), 'PR AUC': pr_auc, 'FPR': fpr, 'TPR': tpr,
                        'Prec thr': preroc, 'Rec thr': recroc}

    # save results in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(res_dir + 'res_eval.npy', res)

    print('Plotting ROC and PR AUCs...')
    # draw evaluation plots
    draw_plots(res, res_dir, output_cl)

    print('Saving metrics to a txt file...')
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


    # # Generate the predictions.
    # prediction_lst = []
    # count = 0
    # for predictions in estimator.predict(input_fn):
    #     assert len(predictions) == 1
    #     prediction_lst.append(predictions[0])
    #     count += 1
    #     if (count % 500) == 0:
    #         print(count)
    #     # print(str(predictions[0]))
    #
    #
    # # filepath = '/data5/tess_project/pedro/dr25_total/filt_input_tb.csv'
    # # tce_table = pd.read_csv(filepath, index_col="loc_rowid", comment="#")
    # # mes_dict = {}
    # # for index, row in tce_table.iterrows():
    # #     mes_dict[row['kepid']] = row['mes']
    # # p = pickle.Pickler(open('mes_dict.pkl', "wb+"))
    # # p.fast = True
    # # p.dump(mes_dict)
    #
    # with open('mes_dict.pkl', 'rb+') as fp:
    #     mes_dict = pickle.load(fp)
    #
    # output_list = [{'kepid': kepid, 'prediction': pred_i, 'ephemeris': ephem_i, 'mes': mes_dict[kepid]}
    #                for kepid, pred_i, ephem_i in zip(kepid_vec, prediction_lst, ephem_vec)]
    #
    # ranked_predictions = sorted(output_list, key=lambda x: x['prediction'], reverse=True)
    #
    #
    # print_n_entries_thres = 10000000
    # ranked_predictions = ranked_predictions[:print_n_entries_thres] if len(ranked_predictions) > print_n_entries_thres else ranked_predictions
    #
    # count_row = 1
    # with open('cand_list_2.csv', mode='w') as f:
    #     f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     f_writer.writerow([str('Total datapoints:' + str(len(ranked_predictions)))])
    #     f_writer.writerow(['rowid', 'kepid', 'prediction', 'MES'])
    #     for entry in ranked_predictions:
    #         f_writer.writerow([count_row, entry['kepid'], entry['prediction'], entry['mes']])
    #         count_row += 1


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    # load Shallue's best config
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': None,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}
    ######### SCRIPT PARAMETERS #############################################

    cmmn_ids = None  # np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cmmn_kepids.npy').item()

    study = 'study_bohb_dr25_tcert_spline2'
    # set configuration manually, None to load it from a HPO study
    config = None

    # load test data
    tfrec_dir = paths.tfrec_dir['DR25']['spline']['TCERT']

    # threshold on binary classification
    threshold = 0.5
    multi_class = False
    use_kepler_ce = False
    centr_flag = False
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    # load best config from HPO study
    if config is None:
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
    models_path = paths.pathtrainedmodels + study + '/models'  # + '/100_epochs/models'

    # path to save results
    pathsaveres = paths.pathsaveres_get_pcprobs + study + '/'

    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    # add dataset parameters
    config = src.config.add_dataset_params(tfrec_dir, satellite, multi_class, centr_flag, config)

    # add missing parameters in hpo with default values
    config = src.config.add_default_missing_params(config=config)
    print('Configuration used: ', config)

    main(config=config,
         model_dir=models_path,
         data_dir=tfrec_dir,
         res_dir=pathsaveres,
         threshold=threshold,
         cmmn_ids=cmmn_ids)

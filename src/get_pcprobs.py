"""
Test ensemble of models trained using the best configuration obtained in a hyperparameter optimization study.
"""

import os
# import csv
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve

# if 'nobackup' in os.path.dirname(__file__):
#     from src.estimator_util import ModelFn, CNN1dModel
#     from src.config import Config
# else:
from src.estimator_util import ModelFn, CNN1dModel
from src.config import Config

# from estimator_util import ModelFn, CNN1dModel
# from config import Config


def main(config, model_filenames, tfrecord_filenames, pathsaveres, threshold=0.5):
    """

    :param config: Config class, config object
    :param model_filenames: list, models filepaths
    :param tfrecord_filenames: list, tfrecords filepaths
    :param pathsaveres: str, save directory
    :param threshold: float, classification threshold
    :return:
    """

    # num_afps, num_ntps, num_pcs = 0, 0, 0
    kepid_vec, glob_vec, loc_vec, labels = [], [], [], []
    # kepid_vec, glob_vec, loc_vec, ephem_vec, glob_centrvec, loc_centrvec, mes_vec = [], [], [], [], [], [], []
    for file in tfrecord_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            kepid = example.features.feature['kepid'].int64_list.value[0]
            # tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
            label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
            # if label == 'AFP':
            #     num_afps += 1
            # if label == 'NTP':
            #     num_ntps += 1
            # if label == 'PC':
            #     num_pcs += 1
            # NEEDS TO BE ADAPTED FOR MULTICLASS AND TESS!!!
            if label in ['AFP', 'NTP']:
                labels.append(0)
            else:
                labels.append(1)
            # period = example.features.feature['tce_period'].float_list.value[0]
            # duration = example.features.feature['tce_duration'].float_list.value[0]
            # epoch = example.features.feature['tce_time0bk'].float_list.value[0]
            # MES = example.features.feature['mes'].float_list.value[0]
            # ephem_vec += [{'period': period, 'duration': duration, 'epoch': epoch}]

            glob_view = example.features.feature['global_view'].float_list.value
            loc_view = example.features.feature['local_view'].float_list.value
            # glob_view_centr = example.features.feature['global_view_centr'].float_list.value
            # loc_view_centr = example.features.feature['local_view_centr'].float_list.value

            kepid_vec.append(kepid)
            glob_vec += [glob_view]
            loc_vec += [loc_view]
            # glob_centrvec += [glob_view_centr]
            # loc_centrvec += [loc_view_centr]
            # mes_vec += [MES]

    # print('number of AFP, ATP, PC: ', num_afps, num_ntps, num_pcs)
    # features = [np.array(loc_vec), np.array(glob_vec), np.array(glob_centrvec), np.array(loc_centrvec)]
    features = [np.array(loc_vec), np.array(glob_vec)]
    features = tuple([np.reshape(i, (i.shape[0], 1, i.shape[-1])) for i in features])

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset.repeat(1)
        dataset = dataset.map(parser)

        return dataset

    # def parser(localview, globalview, localview_centr, globalview_centr):
    def parser(localview, globalview):
        # output = {"time_series_features": {'local_view': tf.to_float(localview),
        #                                    'global_view': tf.to_float(globalview),
        #                                    'global_view_centr': tf.to_float(localview_centr),
        #                                    'local_view_centr': tf.to_float(globalview_centr)}}
        output = {"time_series_features": {'local_view': tf.to_float(localview),
                                           'global_view': tf.to_float(globalview)}}
        return output

    prediction_matrix = []
    for i, model_filename in enumerate(model_filenames):
        print('Testing for model %i in %s' % (i + 1, model_filename))

        config_sess = None
        config_sess = tf.ConfigProto(log_device_placement=False)


        estimator = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                           config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                         session_config=config_sess),
                                           model_dir=model_filename)

        prediction_lst = []
        for predictions in estimator.predict(input_fn):
            assert len(predictions) == 1
            prediction_lst.append(predictions[0])

        prediction_matrix.append(prediction_lst)

    labels = np.array(labels, dtype='uint8')
    prediction_matrix = np.array(prediction_matrix)

    # average across models
    ensemble_prediction = np.mean(prediction_matrix, axis=0)

    # threshold for classification
    ensemble_classification = np.zeros(ensemble_prediction.shape, dtype='uint8')
    ensemble_classification[ensemble_prediction >= threshold] = 1

    # compute and save performance metrics for the ensemble
    acc = accuracy_score(labels, ensemble_classification)
    roc_auc = roc_auc_score(labels, ensemble_prediction, average='macro')
    pr_auc = average_precision_score(labels, ensemble_prediction, average='macro')
    prec = precision_score(labels, ensemble_classification, average='binary')
    rec = recall_score(labels, ensemble_classification, average='binary')
    metrics_dict = {'Accuracy': acc, 'ROC AUC': roc_auc, 'Precision': prec, 'Recall': rec, 'Threshold': threshold,
                    'Number of models': len(model_filenames), 'PR AUC': pr_auc}
    for metric in metrics_dict:
        print(metric + ': ', metrics_dict[metric])

    # save results
    np.save(pathsaveres + 'metrics.npy', metrics_dict)
    print('Metrics saved to %s' % (pathsaveres + 'metrics.npy'))

    fpr, tpr, _ = roc_curve(labels, ensemble_prediction)
    preroc, recroc, _ = precision_recall_curve(labels, ensemble_prediction)
    np.save(pathsaveres + 'roc_pr_aucs.npy', [fpr, tpr, preroc, recroc])

    print('Plotting ROC and PR AUCs...')
    f = plt.figure(figsize=(9, 6))
    lw = 2
    # ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    ax = f.add_subplot(111, label='PR ROC')
    ax.plot(recroc, preroc, color='darkorange', lw=lw, label='PR ROC curve (area = %0.2f)' % pr_auc)
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
    ax2.plot(fpr, tpr, color='darkorange', lw=lw, linestyle='--', label='AUC ROC curve (area = %0.2f)' % roc_auc)
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

    f.suptitle('PR/ROC Curves')
    f.savefig(pathsaveres + 'pr_roc.png')
    plt.close()

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

    # load best config from HPO study
    res = hpres.logged_results_to_HBS_result('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                                             'hpo_configs/study_9')
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best_config = id2config[incumbent]['config']
    # best_config = id2config[(13, 0, 7)]['config']
    # load Shallue's best config
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': 1e-4,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    config = best_config  # CHANGE TO THE CONFIG YOU WANT TO LOAD!!!
    print('Configuration loaded:', config)

    # path to trained models' weights on the best config
    models_path = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/study_9/models'
    # models_path = '/home/msaragoc/Kepler_planet_finder/models/run_shallues_bestconfig'
    model_filenames = [models_path + '/' + file for file in os.listdir(models_path)]

    # load test data
    tfrecord_par_path = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_kepler'
    tfrecord_filenames = [tfrecord_par_path + '/' + file for file in os.listdir(tfrecord_par_path) if 'test' in file]
    if not tfrecord_filenames:
        raise ValueError("Found no input tfrecord files")

    # path to save results
    pathsaveres = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/study_9/'
    # pathsaveres = '/home/msaragoc/Kepler_planet_finder/results/run_shallues_bestconfig/'
    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    # threshold on binary classification
    threshold = 0.5

    main(Config(None, **config), model_filenames, tfrecord_filenames, pathsaveres, threshold=threshold)

"""
.
"""

import os
import csv
import hpbandster.core.result as hpres
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

if 'nobackup' in os.path.dirname(__file__):
    from src.estimator_util import ModelFn, CNN1dModel
    from src.config import Config
else:
    from src.estimator_util import ModelFn, CNN1dModel
    from src.config import Config


def main(config, model_filenames, tfrecord_filenames, pathsaveres, tfrbatchsize=None):

    ensemble_prediction = []
    lc_ite = 1
    kepid_vec, glob_vec, loc_vec, ephem_vec, labels = [], [], [], [], []
    # kepid_vec, glob_vec, loc_vec, ephem_vec, glob_centrvec, loc_centrvec, mes_vec = [], [], [], [], [], [], []
    for file in tfrecord_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            kepid = example.features.feature['kepid'].int64_list.value[0]
            # tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
            # label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
            label = 0
            period = example.features.feature['tce_period'].float_list.value[0]
            duration = example.features.feature['tce_duration'].float_list.value[0]
            epoch = example.features.feature['tce_time0bk'].float_list.value[0]
            # MES = example.features.feature['mes'].float_list.value[0]
            ephem_vec += [{'period': period, 'duration': duration, 'epoch': epoch}]

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

        if lc_ite % tfrbatchsize == 0 or file == tfrecord_filenames[-1]:

            # print('number of AFP, ATP, PC: ', num_afps, num_ntps, num_pcs)
            # features = [np.array(loc_vec), np.array(glob_vec), np.array(glob_centrvec), np.array(loc_centrvec)]
            print('shapes', len(loc_vec), len(glob_vec))
            features = [np.array(loc_vec), np.array(glob_vec)]
            features = tuple([np.reshape(i, (i.shape[0], 1, i.shape[-1])) for i in features])

            glob_vec, loc_vec = [], []

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

            # aaaaa
            prediction_matrix = []
            for i, item in enumerate(model_filenames):
                print('Testing for model %i in %s' % (i + 1, item))

                # DO WE NEED THIS?
                config.model_dir_custom = item

                estimator = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                                   config=tf.estimator.RunConfig(keep_checkpoint_max=1),
                                                   model_dir=config.model_dir_custom)

                prediction_lst = []

                for predictions in estimator.predict(input_fn):
                    assert len(predictions) == 1
                    print('predictions', predictions)
                    prediction_lst.append(predictions[0])

                print(len(prediction_lst))
                prediction_matrix.append(prediction_lst)

                prediction_matrix = np.array(prediction_matrix)

                # average across models
                ensemble_prediction.extend(np.mean(prediction_matrix, axis=0))

            del features

        lc_ite += 1

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

    # get MES values
    filepath = pathsaveres + '/home/msaragoc/Kepler_planet_finder/filt_input_tb.csv'
    tce_table = pd.read_csv(filepath, index_col="loc_rowid", comment="#")
    mes_dict = {}
    for index, row in tce_table.iterrows():
        mes_dict[row['kepid']] = row['mes']
    p = pickle.Pickler(open('mes_dict.pkl', "wb+"))
    p.fast = True
    p.dump(mes_dict)

    print('MES dict', mes_dict)
    # with open(pathsaveres + 'mes_dict.pkl', 'rb+') as fp:
    #     mes_dict = pickle.load(fp)

    output_list = [{'kepid': kepid, 'prediction': pred_i, 'ephemeris': ephem_i, 'mes': mes_dict[kepid]}
                   for kepid, pred_i, ephem_i in zip(kepid_vec, ensemble_prediction, ephem_vec)]

    # rank predictions based on the score values
    ranked_predictions = sorted(output_list, key=lambda x: x['prediction'], reverse=True)

    # print_n_entries_thres = 10000000
    # ranked_predictions = ranked_predictions[:print_n_entries_thres] if len(ranked_predictions) > print_n_entries_thres else ranked_predictions

    # save results to csv file
    count_row = 1
    with open(pathsaveres + 'cand_list.csv', mode='w') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([str('Total datapoints:' + str(len(ranked_predictions)))])
        f_writer.writerow(['rowid', 'kepid', 'prediction', 'MES'])
        for entry in ranked_predictions:
            f_writer.writerow([count_row, entry['kepid'], entry['prediction'], entry['mes']])
            count_row += 1


if __name__ == "__main__":

    # tf.logging.set_verbosity(tf.logging.INFO)

    # load best config from HPO study
    res = hpres.logged_results_to_HBS_result('/home/msaragoc/Kepler_planet_finder/configs/study_1')
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best_config = id2config[incumbent]['config']
    # load Shallue's best config
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 16, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': 1e-2,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    config = shallues_best_config  # CHANGE TO THE CONFIG YOU WANT TO LOAD!!!
    print('Configuration loaded:', config)

    # path to trained models' weights on the best config
    # models_path = '/home/msaragoc/Kepler_planet_finder/models/run_2'
    models_path = '/home/msaragoc/Kepler_planet_finder/models/run_pred200k'
    model_filenames = [models_path + '/' + file for file in os.listdir(models_path)]
    model_filenames = [model_filenames[0]]

    # load test data
    tfrecord_par_path = '/home/msaragoc/Kepler_planet_finder/tf_rec_all'
    print('Loading tfrecords data from %s' % tfrecord_par_path)
    tfrecord_filenames = [tfrecord_par_path + '/' + file for file in os.listdir(tfrecord_par_path) if 'predict' in file]
    if not tfrecord_filenames:
        raise ValueError("Found no input tfrecord files.")
    else:
        print('Data loaded successfully.')

    # path to save results
    pathsaveres = '/home/msaragoc/Kepler_planet_finder/results/run_pred200k/'
    # pathsaveres = '/home/msaragoc/Kepler_planet_finder/results/run_shallues_bestconfig/'
    if not os.path.isdir(pathsaveres):
        os.mkdir(pathsaveres)

    # main(Config(**best_config), model_filenames, tfrecord_filenames, pathsaveres, threshold=threshold)
    main(Config(**config), model_filenames, tfrecord_filenames, pathsaveres, tfrbatchsize=600)

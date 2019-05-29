import tensorflow as tf
import os

from src.estimator_util import InputFn, ModelFn, CNN1dModel
from src.config import Config


def get_numpy_dataset(config):
    filenames = []
    for tfrec_dir in [config.tfrecord_dir_tpsrejects, config.tfrec_dir]:
        filenames += [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir)]

    tfrec_dict = {'features': {'global_view': [], 'local_view': []},
                  'kepids': []}

    if config.centr_flag:
        tfrec_dict['features'] = {'global_view_centr': [], 'local_view_centr': [], **tfrec_dict['features']}

    for file in filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            kepid = example.features.feature['kepid'].int64_list.value[0]
            tce_n = example.features.feature['tce_plnt_num'].int64_list.value[0]
            # label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")

            tfrec_dict['kepids'] += [(kepid, tce_n)]

            for timeseries_id in tfrec_dict['features'].keys():
                tfrec_dict[timeseries_id] += [example.features.feature[timeseries_id].float_list.value]

    return tfrec_dict


# def get_input_function(mode, config, tfrec_dict, labels=None):
#
#     # features = [np.array(loc_vec), np.array(glob_vec)]
#     # features = tuple([np.reshape(i, (i.shape[0], 1, i.shape[-1])) for i in features])
#     #
#     # def input_fn():
#     #     # batch_size = 64  # commented out because batching not implemented correctly
#     #
#     #     dataset = tf.data.Dataset.from_tensor_slices(features)
#     #     dataset.repeat(1)
#     #     dataset = dataset.map(parser)
#     #     # dataset = dataset.batch(batch_size)
#     #     # dataset = dataset.prefetch(max(1, int(256 / batch_size)))
#     #
#     #     return dataset
#     #
#     # def parser(localview, globalview):
#     #     # data_fields = {feature_name: tf.FixedLenFeature([length], tf.float32)
#     #     #                for feature_name, length in {'local_view': 201, 'global_view': 2001}.items()}
#     #     #
#     #     # parsed_features = tf.parse_single_example(serialized_example, features=data_fields)
#     #
#     #     output = {"time_series_features": {'local_view': tf.to_float(localview),
#     #                                        'global_view': tf.to_float(globalview)}}
#     #
#     #     return output
#
#     return tf.estimator.inputs.numpy_input_fn(tfrec_dict['features'], batch_size=config.batch_size)


def get_predictions(classifier, predict_input_fn):

    prediction_lst = []
    for prediction in classifier.predict(predict_input_fn):
        assert len(prediction) == 1
        prediction_lst.append(prediction[0])
        # print(str(predictions[0]))

    return prediction_lst
    # {kepid: pred_i for kepid, pred_i in zip(kepids, prediction_lst)}


def get_difference(prev_pred, new_pred):

    if prev_pred is None:
        return 10000

    count = 0
    for i, prediction in enumerate(new_pred):
        if prediction != prev_pred[i]:
            count += 1

    return count


def run_main(config):

    convergence_thres = 0

    classifier = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                        config=tf.estimator.RunConfig(keep_checkpoint_max=1),
                                        model_dir=config.model_dir_custom)

    train_input_fn = InputFn(file_pattern=config.tfrec_dir + '/train*', batch_size=config.batch_size,
                             mode=tf.estimator.ModeKeys.TRAIN, label_map=config.label_map, centr_flag=config.centr_flag)

    for epoch_i in range(config.n_epochs):
        print('\n\x1b[0;33;33m' + "Starting training of initial model" + '\x1b[0m\n')
        _ = classifier.train(train_input_fn)

    tfrec_dict = get_numpy_dataset(config)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(tfrec_dict['features'], batch_size=config.batch_size)

    iteration = 0
    previous_predictions = None
    while True:
        iteration += 1

        new_predictions = get_predictions(classifier, predict_input_fn)

        difference_count = get_difference(previous_predictions, new_predictions)

        if difference_count <= convergence_thres:
            break

        # print('\n\x1b[0;33;33m' + "Difference count: %s" % str(count) + '\x1b[0m\n')
        print("iteration %s, difference count: %s" % (str(iteration), str(difference_count)))

        train_input_fn = tf.estimator.inputs.numpy_input_fn(tfrec_dict['features'], batch_size=config.batch_size,
                                                            y=new_predictions, shuffle=True)

        for epoch_i in range(config.n_epochs):
            train_str = "Iteration %s, starting epoch %d of %d" % (str(iteration), epoch_i + 1, config.n_epochs)
            print('\n\x1b[0;33;33m' + train_str + '\x1b[0m\n')
            _ = classifier.train(train_input_fn)

        previous_predictions = new_predictions


if __name__ == '__main__':
    run_main(Config())

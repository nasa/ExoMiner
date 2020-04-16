"""
Perform inference using a single TensorFlow Estimator model.
"""

# 3rd party
import sys
# sys.path.append('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/')
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf

# local
from src.estimator_util import InputFn, ModelFn, CNN1dModel, CNN1dPlanetFinderv1, Exonet, Exonet_XS, \
    get_data_from_tfrecord, get_data_from_tfrecord_kepler
# needed for backward compatibility for models created before upgrading the model building function CNN1dModel in
# estimator_util to use tf.keras layers and different names for the graph nodes
# from src.estimator_util_bc import InputFn, ModelFn, CNN1dModel, get_data_from_tfrecord
# import src.config
# import src_hpo.utils_hpo as utils_hpo
# import paths
# import baseline_configs
# import src.utils_data as utils_data


def predict(id_proc, model_fp, data_dir, base_model, datasets, sess_config=None, proc_to_gpu_mapping=None):
    """ Test single model/ensemble of models.

    :param model_dir: str, directory with saved models
    :param data_dir: str, data directory with tfrecords
    :param datasets: list, datasets in which the model(s) is(are) applied to. The datasets names should be strings that
    match a part of the tfrecord filename - 'train', 'val', 'test', 'predict'
    :param id_proc: int, id of the process
    :param sess_config:
    :param proc_to_gpu_mapping: list, mapping of processes to GPUs
    :return:
        predictions_data, dict with key/value pairs for each dataset; values are a list of scores
    """

    if proc_to_gpu_mapping is not None:
        sess_config.gpu_options.visible_device_list = proc_to_gpu_mapping[id_proc]

    # predict on given datasets
    predictions_dataset = {dataset: [] for dataset in datasets}
    for dataset in predictions_dataset:

        print('Predicting in dataset %s for model %s' % (dataset, model_fp))

        config = np.load('{}/config.npy'.format(model_fp), allow_pickle=True).item()
        features_set = np.load('{}/features_set.npy'.format(model_fp), allow_pickle=True).item()

        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*', batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT, label_map=config['label_map'],
                                   features_set=features_set)

        estimator = tf.estimator.Estimator(ModelFn(base_model, config),
                                           config=tf.estimator.RunConfig(session_config=sess_config,
                                                                         tf_random_seed=None),
                                           model_dir=model_fp)

        prediction_lst = []
        for predictions in estimator.predict(predict_input_fn, yield_single_examples=True):
            assert len(predictions) == 1
            prediction_lst.append(predictions[0])

        predictions_dataset[dataset].append(prediction_lst)

    return predictions_dataset


if __name__ == "__main__":

    pass

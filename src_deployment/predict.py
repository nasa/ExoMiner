"""
Author: Miguel Martinho, miguel.martinho@nasa.gov

Main script of the Deep Learning (DL) pipeline. Receives TCE data from the TPS module [1] and outputs a score for the
respective TCE.

[1] Jenkins, Jon M., et al. "Transiting planet search in the Kepler pipeline." Software and Cyberinfrastructure for
Astronomy. Vol. 7740. International Society for Optics and Photonics, 2010.

TODO: remove Python modules only needed to test!
"""

# 3rd party
import os
# control OMP logging
os.environ['KMP_WARNINGS'] = '0'
import logging
# control TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import numpy as np
import multiprocessing
import pandas as pd
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))
import matplotlib; matplotlib.use('agg')
from mpi4py import MPI

# local
from src_deployment.runconfig_params import Config
from src_deployment.preprocess import _process_tce, read_light_curve
from src_deployment.estimator_util import InputFn, ModelFn, CNN1dPlanetFinderv1, CNN1dModel


def single_model_inference(modelFilepath, runConfig, tceFeatures):
    """ Performs inference for a single model.

    :param modelFilepath: str, model filepath
    :param runConfig: instance from Config, preprocessing and inference parameters
    :param tceFeatures: dictionary of features for the TCE. Check preprocess.py generate_example_for_tce function to
    know more about the structure of the dictionary
    :return:
        scores: list of scores output by the model
    """

    # load model configuration and feature set dimensions and data types
    modelConfig = np.load('{}/config.npy'.format(modelFilepath), allow_pickle=True).item()
    features_set = np.load('{}/features_set.npy'.format(modelFilepath), allow_pickle=True).item()

    # if using GPUs
    if runConfig.multiGPU:
        runConfig.config_tfsess.gpu_options.visible_device_list = str(MPI.COMM_WORLD.rank % runConfig.numGPUsPerNode)

    # create predict input function for the estimator
    predict_input_fn = InputFn(features=tceFeatures,
                               batch_size=modelConfig['batch_size'],
                               mode=tf.estimator.ModeKeys.PREDICT,
                               features_set=features_set)

    # instantiate the estimator
    estimator = tf.estimator.Estimator(ModelFn(CNN1dModel, modelConfig),
                                       config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                     session_config=runConfig.config_tfsess),
                                       model_dir=modelFilepath)

    # inference
    scores = []
    for score in estimator.predict(predict_input_fn):
        assert len(score) == 1
        scores.append(score[0])

    return scores


def sequential_inference(modelsFilepath, runConfig, tceFeatures):
    """ Performs inference for an ensemble of models sequentially.

    :param modelsFilepath: list of filepaths for the ensemble of models
    :param runConfig: instance from Config, preprocessing and inference parameters
    :param tceFeatures: dictionary of features for the TCE. Check preprocess.py generate_example_for_tce function to
    know more about the structure of the dictionary
    :return:
        scoresSet: list of scores output by the ensemble of models [[model1Score], [model2Score], ...]
    """

    # initialize list of scores for the ensemble
    scoresSet = []

    # run inference sequentially for each model
    for i, modelFilepath in enumerate(modelsFilepath):
        print('Predicting for model %i (out of %i)' % (i + 1, len(modelsFilepath)))

        # run single model inference
        scores = single_model_inference(modelFilepath, runConfig, tceFeatures)

        # append scores output for the model
        scoresSet.append(scores)

    return scoresSet


def parallel_inference(modelsFilepath, runConfig, tceFeatures):
    """ Performs inference for an ensemble of models in parallel (CPU or GPU).

    :param modelsFilepath: list of filepaths for the ensemble of models
    :param runConfig: instance from Config, preprocessing and inference parameters
    :param tceFeatures: dictionary of features for the TCE. Check preprocess.py generate_example_for_tce function to
    know more about the structure of the dictionary
    :return:
        scoresSet: list of scores output by the ensemble of models [[model1Score], [model2Score], ...]
    """

    tf.logging.info("Launching {} processes".format(runConfig.numProcesses))

    # create a pool of runConfig.numProcesses child processes
    pool = multiprocessing.Pool(processes=runConfig.numProcesses)
    # run each one in parallel in a different CPU
    async_results = [pool.apply_async(single_model_inference, (modelFilepath, runConfig, tceFeatures))
                     for modelFilepath in modelsFilepath]
    pool.close()

    # get output from each child process
    # instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
    scoresSet = [async_result.get() for async_result in async_results]

    return scoresSet


def dl_inference_pipeline(time, flux, ephemeris):
    """ Run Deep Learning Inference Pipeline for a TCE whose data come from the TPS module.

    :param time: list of numpy arrays, cadences (time in JD)
    :param flux: list of numpy arrays, PDC-SAP flux time-series
    :param ephemeris: pandas Series, TCE ephemeris
    :return:
        scoreEnsemble: list with the score output by the ensemble of models for the TCE
    """

    # get run parameters (preprocessing and inference)
    runConfig = Config()

    # process the TCE
    tceFeatures = _process_tce(ephemeris, flux, time, runConfig)

    # get models' filepaths
    modelsFilepath = [os.path.join(runConfig.modelDir, modelFilename)
                      for modelFilename in os.listdir(runConfig.modelDir)]

    # run sequential/parallel inference on the models for the TCE
    # scoresSet = sequential_inference(modelsFilepath, runConfig, tceFeatures)
    scoresSet = parallel_inference(modelsFilepath, runConfig, tceFeatures)

    # average scores across models in the ensemble
    scoreEnsemble = np.mean(scoresSet, axis=0)

    # # threshold the predicted output to get a classification
    # predClassEnsemble = scoreEnsemble[scoreEnsemble >= runConfig.classificationThreshold] = 1

    return scoreEnsemble


if __name__ == '__main__':

    # keplerID, tceID = 1433980, 1
    tessID, tceID, sector = 150100106, 1, 7

    # tceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/q1_q17_dr25_tce_2019.03.12_updt_tcert_'
    #                        'extendedtceparams_updt_normstellarparamswitherrors.csv')
    tceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/'
                           'toi_list_ssectors_dvephemeris_ephmatchnoepochthr0,25.csv')

    # TODO: change plot function so that it only takes the ephemeris into account after testing
    # ephemeris = tceTable.loc[(tceTable['kepid'] == keplerID) & (tceTable['tce_plnt_num'] == tceID)][['kepid',
    #                                                                                                  'tce_plnt_num',
    #                                                                                                  'av_training_set',
    #                                                                                                  'tce_period',
    #                                                                                                  'tce_duration',
    #                                                                                                  'tce_time0bk']]
    ephemeris = tceTable.loc[(tceTable['tic'] == tessID) & (tceTable['tce_plnt_num'] == tceID) &
                             (tceTable['sector'] == sector)][['tic',
                                                              'tce_plnt_num',
                                                              'disposition',
                                                              'sector',
                                                              'orbitalPeriodDays',
                                                              'transitDurationHours',
                                                              'transitEpochBtjd']]

    ephemeris = ephemeris.iloc[0]
    # ephemeris['tce_duration'] /= 24.0
    ephemeris['transitDurationHours'] /= 24.0

    runConfig = Config()

    # runConfig.lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'
    runConfig.lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

    if not os.path.isdir(runConfig.output_dir):
        os.makedirs(os.path.join(runConfig.output_dir, 'plots'))

    time, flux = read_light_curve(ephemeris, runConfig)

    scoreEnsemble = dl_inference_pipeline(time, flux, ephemeris)

    # print('Ensemble score: {}'.format(scoreEnsemble))

    # with open(os.path.join(runConfig.output_dir, "{}-{}.txt".format(keplerID, tceID)), "a") as res_file:
    with open(os.path.join(runConfig.output_dir, "{}-{}-s{}.txt".format(tessID, tceID, sector)), "a") as res_file:
        res_file.write('Score ensemble = {}'.format(scoreEnsemble))

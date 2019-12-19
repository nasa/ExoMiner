"""

"""

# 3rd party
import tensorflow as tf
import numpy as np
import os

# local
from src_deployment.runconfig_params import Config
from src_deployment.preprocess import _process_tce
from src_deployment.estimator_util import InputFn, ModelFn, CNN1dPlanetFinderv1, CNN1dModel


def dl_inference_pipeline(time, flux, ephemeris):

    runConfig = Config()

    with tf.python_io.TFRecordWriter(tfrecFilename) as writer:
        example = _process_tce(time, flux, ephemeris, runConfig)
        if example is not None:
            writer.write(example.SerializeToString())

    # get models' filepaths
    modelsFilepath = [os.path.join(runConfig.modelDir, modelFilename)
                      for modelFilename in os.listdir(runConfig.modelDir)]
    # classificationThreshold = 0.5

    scoresSet = []
    for i, modelFilepath in enumerate(modelsFilepath):
        print('Predicting for model %i (out of %i)' % (i + 1, len(modelsFilepath)))

        modelConfig = np.load('{}/config.npy'.format(modelFilepath))
        features_set = np.load('{}/features_set.npy'.format(modelFilepath))

        predict_input_fn = InputFn(file_pattern=tfrecFilepath, batch_size=modelConfig['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT, label_map=modelConfig['label_map'],
                                   features_set=features_set)

        estimator = tf.estimator.Estimator(ModelFn(CNN1dPlanetFinderv1, modelConfig),
                                           config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                         session_config=runConfig.config_tfsess),
                                           model_dir=modelFilepath)

        scores = []
        for score in estimator.predict(predict_input_fn):
            assert len(score) == 1
            scores.append(score[0])

        scoresSet.append(scores)

    # average scores across models in the ensemble
    scoreEnsemble = np.mean(scoresSet, axis=0)

    # # threshold the predicted output to get a classification
    # predClassEnsemble = scoreEnsemble[scoreEnsemble >= classificationThreshold] = 1

    return scoreEnsemble

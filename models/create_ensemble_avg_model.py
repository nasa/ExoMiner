"""
Create ensemble average model.
"""

# 3rd party
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from pathlib import Path
from tensorflow.keras.models import load_model
import yaml
import argparse

# local
from models.utils_models import create_ensemble
from models.models_keras import Time2Vec, SplitLayer
from src.utils.utils_dataio import set_tf_data_type_for_features


def create_avg_ensemble_model(models_fps, features_set, ensemble_fp):
    """ Create average ensemble of models.

    Args:
        models_fps: list, filepaths to models to be part of the ensemble
        features_set: dict, features set
        ensemble_fp: str, filepath to save the average ensemble model

    Returns:

    """

    # load models into a list
    models = []
    custom_objects = {"Time2Vec": Time2Vec, "SplitLayer": SplitLayer}
    with custom_object_scope(custom_objects):
        for model_i, model_fp in enumerate(models_fps):
            model = load_model(filepath=model_fp, compile=False)
            if tf.__version__ < '2.14.0':
                model._name = f'model{model_i}'
            else:
                model.name = f'model{model_i}'
            models.append(model)

    # create ensemble average model
    ensemble_avg_model = create_ensemble(features_set, models)
    ensemble_avg_model.save(ensemble_fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--ensemble_fp', type=str, help='File path to saved ensemble model.', default=None)
    parser.add_argument('--models_dir', type=str, help='Models directory', default=None)
    args = parser.parse_args()

    models_dir_fp = Path(args.models_dir)
    ensemble_model_fp = Path(args.ensemble_fp)

    # get models file paths
    models_fps_lst = list(models_dir_fp.rglob('model.keras'))
    print(f'Found {len(models_fps_lst)} models for average ensemble.')

    # get features set
    config_fp = Path(args.config_fp)
    with open(config_fp, 'r') as file:
        run_params = yaml.unsafe_load(file)
    features_set_dict = run_params['features_set']

    # convert features to appropriate tf data type
    features_set_dict = set_tf_data_type_for_features(features_set_dict)

    create_avg_ensemble_model(models_fps_lst, features_set_dict, ensemble_model_fp)

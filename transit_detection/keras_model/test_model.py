"""
Used to test building models.
"""

# 3rd party
import yaml
from pathlib import Path

# local
from model import ExoMiner_TESS_Transit_Detection
from src.utils_train_eval_predict import set_tf_data_type_for_features

# load file with features and model config
yaml_config_fp = Path('/Users/jochoa4/Desktop/ExoMiner/exoplanet_dl/transit_detection/keras_model/config_train.yaml')

with(open(yaml_config_fp, 'r')) as file:
    config = yaml.unsafe_load(file)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

model = ExoMiner_TESS_Transit_Detection(config, config['features_set']).kerasModel
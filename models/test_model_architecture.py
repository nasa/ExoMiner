"""
Used to test building models.
"""

# 3rd party
import yaml
from pathlib import Path

# local
from models.models_keras import ExoMiner_JointLocalFlux
from src.utils_train_eval_predict import set_tf_data_type_for_features
from src_hpo.utils_hpo import load_hpo_config

# load file with features and model config
yaml_config_fp = '/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src/config_train.yaml'

with(open(yaml_config_fp, 'r')) as file:
    config = yaml.unsafe_load(file)

# load model hyperparameters from HPO run; overwrites the one in the yaml file
hpo_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/hpo_configs/hpo_merged_unfolded_7-5-2023')
config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
config['config'].update(config_hpo_chosen)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

model = ExoMiner_JointLocalFlux(config, config['features_set']).kerasModel

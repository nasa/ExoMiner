"""
Used to test building models.
"""

# 3rd party
import yaml
from tensorflow.keras.utils import plot_model


# local
from models.models_keras import ExoMiner_JointLocalFlux
from src.utils.utils import set_tf_data_type_for_features

# load file with features and model config
yaml_config_fp = '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_cv/config_cv_train.yaml'

with open(yaml_config_fp, 'r') as file:
    config = yaml.unsafe_load(file)

# load model hyperparameters from HPO run; overwrites the one in the yaml file
# hpo_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/hpo_configs/hpo_merged_unfolded_7-5-2023')
# config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
# config['config'].update(config_hpo_chosen)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

model = ExoMiner_JointLocalFlux(config, config['features_set']).kerasModel

plot_model(model,
           to_file='/Users/msaragoc/Downloads/test_exominer_architecture_11-19-2024_0939.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=False,
           dpi=48)

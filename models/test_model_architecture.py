"""
Used to test building models.
"""

# 3rd party
import yaml
from tensorflow.keras.utils import plot_model


# local
from models.models_keras import ExoMinerPlusPlus, ExoMinerJointLocalFlux, ExoMinerMLP
from src.utils.utils_dataio import set_tf_data_type_for_features

# load file with features and model config
yaml_config_fp = '/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_cv/train/config_cv_train.yaml'

with open(yaml_config_fp, 'r') as file:
    config = yaml.unsafe_load(file)

# load model hyperparameters from HPO run; overwrites the one in the yaml file
# hpo_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/hpo_configs/hpo_merged_unfolded_7-5-2023')
# config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
# config['config'].update(config_hpo_chosen)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

model = ExoMinerPlusPlus(config, config['features_set']).kerasModel

plot_model(model,
           to_file='/home6/msaragoc/model_tf_learning.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=False,
           dpi=48)

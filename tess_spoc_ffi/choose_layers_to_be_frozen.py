"""
Choose layers in model to be frozen
"""

# 3rd party
import yaml
from tensorflow.keras.utils import plot_model

# local
from models.models_keras import ExoMinerPlusPlus
from src.utils.utils_dataio import set_tf_data_type_for_features

# load file with features and model config
yaml_config_fp = '/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_cv/train/config_cv_train.yaml'

with open(yaml_config_fp, 'r') as file:
    config = yaml.unsafe_load(file)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

model = ExoMinerPlusPlus(config, config['features_set']).kerasModel

plot_model(model,
           to_file='/home6/msaragoc/model_tf_learning.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=False,
           dpi=48)

#%%%

frozen_layers_lst, trainable_layers_lst, noweights_layers_lst = [], [], []
for layer in model.layers:
    print(layer.name)
    if not layer.trainable_weights:
        noweights_layers_lst.append(layer.name)
    elif 'conv' in layer.name:
        frozen_layers_lst.append(layer.name)
    elif 'pooling' in layer.name:
        frozen_layers_lst.append(layer.name)
    elif 'prelu' in layer.name:
        frozen_layers_lst.append(layer.name)
    else:
        trainable_layers_lst.append(layer.name)

with open('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/tess_spoc_ffi/trainable_layers.yaml', 'w') as file:
    yaml.dump(trainable_layers_lst, file)

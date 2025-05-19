"""
Used to test building models.
"""

# 3rd party
import yaml
from tensorflow.keras.utils import plot_model

# local
from src.utils.utils_dataio import set_tf_data_type_for_features
from models import models_keras

# load file with features and model config
yaml_config_fp = 'exominer_plus_plus.yaml'
model_fn = ''
model_plot_fp = f'.png'
model_summary_fp = f'summary.txt'

with open(yaml_config_fp, 'r') as file:
    config = yaml.unsafe_load(file)

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

base_model = getattr(models_keras, config['model_architecture'])

model = base_model(config, config['features_set']).kerasModel

# get model summary
with open(model_summary_fp, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# plot model architecture
plot_model(model,
   to_file=model_plot_fp,
   show_shapes=True,
   show_layer_names=True,
   rankdir='TB',
   expand_nested=False,
   dpi=48)

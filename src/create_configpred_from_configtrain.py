"""
Create YAML configuration files for prediction runs based on YAML configuration files from training runs.
"""

# 3rd party
from pathlib import Path
import yaml
import tensorflow as tf

# local
from utils.utils_dataio import is_yamlble

configpred_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/experiments/run_configs/explainability_shap_1-10-2022')
configpred_dir.mkdir(exist_ok=True)
default_configpred_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/experiments/run_configs/config_predict.yaml')

# get training experiments directories
train_experiments_dirs = [el for el in Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/experiments/trained_models/').iterdir() if el.is_dir()]

# get training config files for the different experiments
train_config_fps = []
for train_experiments_dir in train_experiments_dirs:
    train_config_fp = [el for el in train_experiments_dir.iterdir() if el.suffix == '.yaml'][0]
    train_config_fps.append(train_config_fp)

for train_config_fp in train_config_fps:

    with(open(train_config_fp, 'r')) as file:  # read YAML configuration train file
        train_config = yaml.load(file, Loader=yaml.Loader)
        # train_config = yaml.safe_load(file)
        # train_config = yaml.danger_load(file)

    with(open(default_configpred_fp, 'r')) as file:  # read default YAML configuration pred file
        pred_config = yaml.safe_load(file)

    # set experiment directory
    pred_config['paths']['experiment_dir'] = f'{pred_config["paths"]["experiment_dir"]}{train_config_fp.parent.name}'

    # get models directory
    pred_config['paths']['models_dir'] = f'{pred_config["paths"]["models_dir"]}{train_config["paths"]["experiment_dir"].name}'

    # get features
    pred_config['features_set'] = {}
    for feat, feat_info in train_config['features_set'].items():
        if train_config['features_set'][feat]['dtype'] == tf.float32:
            feat_dtype = 'float32'
        elif train_config['features_set'][feat]['dtype'] == tf.int64:
            feat_dtype = 'int'
        else:
            raise TypeError('Data type not considered')

        pred_config['features_set'][feat] = {'dim': train_config['features_set'][feat]['dim'], 'dtype': feat_dtype}
    # pred_config['features_set'] = train_config['features_set']

    # save the YAML config pred file
    yaml_dict = {key: val for key, val in pred_config.items() if is_yamlble(val)}  # check if parameters are YAMLble
    with open(configpred_dir / f'{train_config_fp.parent.name}_pred.yaml', 'w') as pred_config_file:
        yaml.dump(yaml_dict, pred_config_file)

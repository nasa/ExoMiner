import argparse
from pathlib import Path

import yaml
"""
Change the dataset_fps YAML File in config_train to change fold for n fold cross_validation. Takes in an argument of the new dataset_fps.yaml filepath.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset_fp', type=str, help='File path to YAML configuration file.', default=None)
args = parser.parse_args()
with(open('/Users/agiri1/Desktop/ExoPlanet/src/config_train_2.yaml', 'r')) as file:
    train_config = yaml.unsafe_load(file)
train_config['paths']['datasets_fps_yaml'] = args.current_dataset_fp
with(open('/Users/agiri1/Desktop/ExoPlanet/src/config_train.yaml', 'w+')) as file:
    yaml.dump(train_config, file, default_flow_style=False, sort_keys=False)
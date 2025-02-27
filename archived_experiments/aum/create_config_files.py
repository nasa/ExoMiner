""" Create config files for AUM runs. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import yaml

# local
from old.utils.utils_dataio import is_yamlble

# experiment directory
experiment_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-11-2022_1144')

# file path to default train configuration file
default_train_config_fp = experiment_dir / 'config_train.yaml'

configs_dir = experiment_dir / 'configs'
configs_dir.mkdir(exist_ok=True)

runs_dir = experiment_dir / 'runs'
runs_dir.mkdir(exist_ok=True)

tfrec_root_dir = experiment_dir / 'tfrecords'

config_runs_fp = experiment_dir / f'config_runs.txt'
if config_runs_fp.is_file():
    config_runs_fp.unlink()

for tfrec_run_dir in sorted(tfrec_root_dir.iterdir()):
    print(f'Create config file for run {tfrec_run_dir.name}...')

    with(open(default_train_config_fp, 'r')) as file:  # read default YAML configuration file
        train_config = yaml.safe_load(file)

    # update path to TFRecord directory
    train_config['paths']['tfrec_dir'] = str(tfrec_run_dir)

    # set experiment directory
    train_config['paths']['experiment_dir'] = str(runs_dir / tfrec_run_dir.name)

    # save the YAML config pred file
    yaml_dict = {key: val for key, val in train_config.items() if is_yamlble(val)}  # check if parameters are YAMLble
    config_fn = f'config_aum_{tfrec_run_dir.name}.yaml'
    with open(configs_dir / config_fn, 'w') as config_file:
        yaml.dump(yaml_dict, config_file)

    # add configuration to list
    with open(config_runs_fp, 'a') as list_configs_file:
        list_configs_file.write(f'{config_fn}\n')

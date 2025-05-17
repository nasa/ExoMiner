"""
Run setup for training iteration.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
import numpy as np

# 3rd party
from src_hpo.utils_hpo import load_hpo_config


def run_setup_for_train_iter(run_dir, config):
    """ Setup configuration parameters for training model.

    :param run_dir: str, experiment directory
    :param config: dict, configuration parameters

    :return:
    """

    # load model hyperparameters from HPO run; overwrites the one in the yaml file
    if config['paths']['hpo_dir']:
        hpo_dir = Path(config['paths']['hpo_dir'])
        config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
        config['config'].update(config_hpo_chosen)

    # load data sets file paths from yaml file
    with open(config['paths']['datasets_fps_yaml'], 'r') as file:
        config['datasets_fps'] = yaml.unsafe_load(file)

    with open(run_dir / 'config_run.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    # save configuration used as a NumPy file to preserve everything that is cannot go into a YAML
    np.save(run_dir / 'config.npy', config)

    # save model's architecture and hyperparameters used
    with open(run_dir / 'model_config.yaml', 'w') as file:
        yaml.dump(config['config'], file, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    output_dir_fp = Path(args.output_dir)
    config_fp = Path(args.config_fp)

    with open(args.config_fp, 'r') as config_file:
        run_config = yaml.safe_load(config_file)

    run_setup_for_train_iter(output_dir_fp, run_config)

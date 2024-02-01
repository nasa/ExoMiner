"""
Run setup for training iteration.
- Load model's hyperparameters from an HPO run.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
# import logging
import numpy as np

# 3rd party
from src_cv.create_cv_dataset.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file
from src_hpo.utils_hpo import load_hpo_config


def run_setup_for_train_iter(run_dir, config):

    # load model hyperparameters from HPO run; overwrites the one in the yaml file
    hpo_dir = Path(config['paths']['hpo_dir'])
    config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
    config['config'].update(config_hpo_chosen)

    with open(run_dir / 'config_run.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    # save configuration used as a NumPy file to preserve everything that is cannot go into a YAML
    np.save(run_dir / 'config.npy', config)

    # save the YAML file with the HPO configuration that was used
    with open(run_dir / 'hpo_config.yaml', 'w') as hpo_config_file:
        yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)
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

    with(open(args.config_fp, 'r')) as config_file:
        run_config = yaml.safe_load(config_file)

    run_setup_for_train_iter(output_dir_fp, run_config)

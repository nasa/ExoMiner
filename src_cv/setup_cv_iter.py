"""
Run setup for CV iteration.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
# import logging
import numpy as np

# 3rd party
from src_cv.cv_dataset.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file
from src_hpo.utils_hpo import load_hpo_config


def run_setup_for_cv_iter(cv_iter, cv_iter_dir, config):

    # add TFRecord data set file paths for this CV iteration to config yaml file
    config = add_tfrec_dataset_fps_to_config_file(cv_iter, config)

    # load model hyperparameters from HPO run; overwrites the one in the yaml file
    hpo_dir = Path(config['paths']['hpo_dir'])
    config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
    config['config'].update(config_hpo_chosen)

    with open(cv_iter_dir / 'config_cv.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    # save configuration used as a NumPy file to preserve everything that is cannot go into a YAML
    np.save(cv_iter_dir / 'config.npy', config)

    # save the YAML file with the HPO configuration that was used
    with open(cv_iter_dir / 'hpo_config.yaml', 'w') as hpo_config_file:
        yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)
    # save model's architecture and hyperparameters used
    with open(cv_iter_dir / 'model_config.yaml', 'w') as file:
        yaml.dump(config['config'], file, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    cv_i = args.cv_iter
    output_dir_fp = Path(args.output_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    # # set up logger
    # cv_iter_config['logger'] = logging.getLogger(name=f'create_config_cv_iter')
    # logger_handler = logging.FileHandler(filename=output_dir_fp / 'create_config_cv_iter.log', mode='w')
    # logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    # cv_iter_config['logger'].setLevel(logging.INFO)
    # logger_handler.setFormatter(logger_formatter)
    # cv_iter_config['logger'].addHandler(logger_handler)
    # cv_iter_config['logger'].info(f'Creating config YAML file for CV iteration in {output_dir_fp}')

    run_setup_for_cv_iter(cv_i, output_dir_fp, cv_iter_config)

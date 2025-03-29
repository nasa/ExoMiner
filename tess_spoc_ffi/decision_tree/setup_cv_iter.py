"""
Run setup for CV iteration.
- Add file paths to the TFRecord files for a given CV iteration.
- Load model's hyperparameters from an HPO run.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
# import logging
import numpy as np

# 3rd party
from src_cv.preprocessing.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file
from src_hpo.utils_hpo import load_hpo_config


def run_setup_for_cv_iter(cv_iter, cv_iter_dir, config, model_i=0):
    """ Run setup for a CV iteration. This involves things such as adding the filepaths for this given iteration to the
    general config yaml file for the CV experiment; loading model hyperparameters from an HPO run

    :param cv_iter: int, CV iteration number
    :param cv_iter_dir: Path, CV iteration directory
    :param config: dict, CV run parameters
    :param model_i: int, model ID

    :return:
    """

    # add TFRecord data set file paths for this CV iteration to config yaml file
    config = add_tfrec_dataset_fps_to_config_file(cv_iter, config, model_i)

    # # load model hyperparameters from HPO run; overwrites the one in the yaml file
    # if 'hpo_dir' in config['paths']:
    #     hpo_dir = Path(config['paths']['hpo_dir'])
    #     config_hpo_chosen, config['hpo_config_id'] = load_hpo_config(hpo_dir)
    #     print(f'Loading hyperparameters from HPO run and overwriting existing ones:\n {config_hpo_chosen}')
    #     config['config'].update(config_hpo_chosen)
    #
    #     # save the YAML file with the HPO configuration that was used
    #     with open(cv_iter_dir / 'hpo_config.yaml', 'w') as hpo_config_file:
    #         yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)

    with open(cv_iter_dir / 'config_cv.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    # save configuration used as a NumPy file to preserve everything that is cannot go into a YAML
    np.save(cv_iter_dir / 'config.npy', config)

    # save model's architecture and hyperparameters used
    with open(cv_iter_dir / 'model_config.yaml', 'w') as file:
        yaml.dump(config['config'], file, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--model_i', type=int, help='Model ID', default=0)

    args = parser.parse_args()

    cv_i = args.cv_iter
    output_dir_fp = Path(args.output_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    # print(f'Creating config YAML file for CV iteration in {output_dir_fp}')

    run_setup_for_cv_iter(cv_i, output_dir_fp, cv_iter_config, model_i=args.model_i)

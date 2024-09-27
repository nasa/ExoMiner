"""
Create config yaml file for a CV iteration based on a template config file for the CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
import numpy as np


def add_tfrec_dataset_fps_to_config_file(cv_iter, config, model_i):
    """ Add the dictionary with a list of the filepaths to the TFRecord shards to be used for training, validation, and
        test.

    :param cv_iter: int, CV iteration number
    :param config: dict, CV run parameters
    :param model_i: int, model ID
    :return: config, dict with CV run parameters updated with the filepaths for the TFRecord shards to be used for
        training, validation, and test
    """

    with(open(config['paths']['cv_folds'], 'r')) as file:
        config['datasets_fps'] = yaml.unsafe_load(file)[cv_iter]

    # randomly pick cv fold as validation set from the training set
    if 'val_from_train' in config:
        if config['val_from_train']:
            rng = np.random.default_rng(seed=config['rnd_seed'] + model_i)  # set different validation set based on model id
            val_fold_idx = rng.choice(len(config['datasets_fps']['train']))
            config['datasets_fps']['val'] = [config['datasets_fps']['train'][val_fold_idx]]
            del config['datasets_fps']['train'][val_fold_idx]

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--model_i', type=int, help='Model ID', default=0)
    args = parser.parse_args()

    cv_i = args.cv_iter
    output_dir_fp = Path(args.model_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    print(f'Creating config YAML file for CV iteration in {output_dir_fp}...')

    add_tfrec_dataset_fps_to_config_file(cv_i, cv_iter_config, model_i=args.model_i)

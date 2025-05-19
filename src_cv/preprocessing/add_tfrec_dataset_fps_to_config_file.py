"""
Create config yaml file for a CV iteration based on a template config file for the CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
import numpy as np


def add_tfrec_dataset_fps_to_config_file(cv_iter, config, model_i, num_val_folds=1):
    """ Add the dictionary with a list of the filepaths to the TFRecord shards to be used for training, validation, and
        test.

    :param cv_iter: int, CV iteration number
    :param config: dict, CV run parameters
    :param model_i: int, model ID
    :param num_val_folds: int, number of training folds to use for validation

    :return: config, dict with CV run parameters updated with the filepaths for the TFRecord shards to be used for
        training, validation, and test
    """

    with(open(config['paths']['cv_folds'], 'r')) as file:
        config['datasets_fps'] = yaml.unsafe_load(file)[cv_iter]

    # randomly pick validation set from the training set
    if 'val_from_train' in config:
        if config['val_from_train']:

            if 'num_val_folds' in config:
                num_val_folds = config['num_val_folds']

            num_train_folds = len(config['datasets_fps']['train'])

            # set validation set based on CV iteration
            rng = np.random.default_rng(seed=config['rnd_seed'] + cv_iter)
            num_val_folds_actual = min(num_val_folds, num_train_folds)
            if num_val_folds_actual < num_val_folds:
                print(f'Number of validation folds requested {num_val_folds} is different than number of '
                      f'validation folds set: {num_val_folds_actual}')
            if num_val_folds_actual == num_train_folds:
                print(f'Number of validation folds was set to number of training folds ({num_train_folds}). '
                      f'Setting number of validation folds to {num_val_folds_actual - 1}')
                num_val_folds_actual -= 1
                num_val_folds_actual = max(0, num_val_folds_actual)

            config['datasets_fps']['val'] = list(rng.choice(config['datasets_fps']['train'],
                                                            num_val_folds_actual,
                                                            replace=False))

            # exclude validation folds from training set
            config['datasets_fps']['train'] = [fp for fp in config['datasets_fps']['train'] if
                                               fp not in config['datasets_fps']['val']]

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

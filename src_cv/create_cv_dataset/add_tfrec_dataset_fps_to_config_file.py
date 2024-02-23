"""
Create config yaml file for a CV iteration based on a template config file for the CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
# import logging
# import numpy as np


def add_tfrec_dataset_fps_to_config_file(cv_iter, config):

    with(open(config['paths']['cv_folds'], 'r')) as file:
        config['datasets_fps'] = yaml.unsafe_load(file)[cv_iter]

    # # cv iterations dictionary
    # data_shards_fns = np.load(config['paths']['cv_folds'], allow_pickle=True)
    # config['datasets_fps'] = [{dataset: [str(Path(config['paths']['tfrec_dir']) / f'cv_iter_{cv_iter_i}' / 'norm_data' / fold) for fold in cv_iter[dataset]]
    #                           for dataset in cv_iter} for cv_iter_i, cv_iter in enumerate(data_shards_fns)][cv_iter]

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    cv_i = args.cv_iter
    output_dir_fp = Path(args.model_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    print(f'Creating config YAML file for CV iteration in {output_dir_fp}...')

    add_tfrec_dataset_fps_to_config_file(cv_i, cv_iter_config)

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
from src_cv.create_cv_dataset.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file
from src_hpo.utils_hpo import load_hpo_config


def create_cv_folds_yaml_from_dir(data_dir, config, cv_iter):
    """ Run setup for a CV iteration. This involves things such as adding the filepaths for this given iteration to the
    general config yaml file for the CV experiment; loading model hyperparameters from an HPO run

    :param data_dir: Path, data directory
    :param config: dict, CV run parameters
    :param cv_iter: int, CV iteration number

    :return:
    """

    dataset_fps = [{'predict': None} for cv_i in range(cv_iter + 1)]
    dataset_fps[cv_iter]['predict'] = list(data_dir.glob('shard-*'))

    # save cv folds filepaths
    dataset_fps_config_fp = data_dir / 'cv_folds.yaml'
    with open(dataset_fps_config_fp, 'w') as file:
        yaml.dump(dataset_fps, file, sort_keys=False)

    # add dataset path to CV iteration run yaml file
    config['paths']['cv_folds'] = str(dataset_fps_config_fp)
    with open(data_dir / 'config_cv.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--data_dir', type=str, help='Data directory.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)

    args = parser.parse_args()

    data_dir_fp = Path(args.data_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    print(f'Creating CV data folds YAML file for CV iteration {args.cv_iter} using data in {data_dir_fp}')

    create_cv_folds_yaml_from_dir(data_dir_fp, cv_iter_config, args.cv_iter)

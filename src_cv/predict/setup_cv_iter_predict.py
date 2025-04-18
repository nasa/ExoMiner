"""
Run setup for CV iteration for prediction.
- Add file paths to the TFRecord files for a given CV iteration.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse
# import logging
import numpy as np

# 3rd party
from src_cv.preprocessing.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file


def run_setup_for_cv_iter_predict(cv_iter, cv_iter_dir, config):
    """ Set configuration yaml file for CV iteration `cv_iter`.

    :param cv_iter: int, CV iteration ID
    :param cv_iter_dir: CV iteration directory
    :param config: dict, configuration parameters

    :return:
    """

    # get TFRecord files for the given CV iteration
    config = add_tfrec_dataset_fps_to_config_file(cv_iter, config, -1)

    with open(cv_iter_dir / 'config_cv.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    # save configuration used as a NumPy file to preserve everything that is cannot go into a YAML
    np.save(cv_iter_dir / 'config.npy', config)


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

    run_setup_for_cv_iter_predict(cv_i, output_dir_fp, cv_iter_config)

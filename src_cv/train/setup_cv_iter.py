"""
Run setup for CV iteration.
- Add file paths to the TFRecord files for a given CV iteration.
- Load model's hyperparameters from an HPO run.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse

# 3rd party
from src_cv.preprocessing.add_tfrec_dataset_fps_to_config_file import add_tfrec_dataset_fps_to_config_file


def run_setup_for_cv_iter(cv_iter, cv_iter_dir, config):
    """ Run setup for a CV iteration. This involves things such as adding the filepaths for this given iteration to the
    general config yaml file for the CV experiment; loading model hyperparameters from an HPO run

    :param cv_iter: int, CV iteration number
    :param cv_iter_dir: Path, CV iteration directory
    :param config: dict, CV run parameters

    :return:
    """

    # add TFRecord data set file paths for this CV iteration to config yaml file
    config = add_tfrec_dataset_fps_to_config_file(cv_iter, config)

    with open(config['paths']['model_config_fp'], 'r') as model_config_f:
        model_config = yaml.unsafe_load(model_config_f)
        config.update(model_config)

    with open(cv_iter_dir / 'config_cv.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    # save model's architecture and hyperparameters used
    with open(cv_iter_dir / 'model_config.yaml', 'w') as file:
        yaml.dump(config['config'], file, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_iter', type=int, help='CV Iteration index/rank.', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)

    args = parser.parse_args()

    cv_i = args.cv_iter
    output_dir_fp = Path(args.output_dir)
    config_fp = Path(args.config_fp)

    with open(args.config_fp, 'r') as config_file:
        cv_iter_config = yaml.safe_load(config_file)

    run_setup_for_cv_iter(cv_i, output_dir_fp, cv_iter_config)

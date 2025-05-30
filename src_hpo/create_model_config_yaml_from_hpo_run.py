"""
Create a YAML file for a model configuration from an HPO run.
"""

# 3rd party
from pathlib import Path
import yaml

# local
from src_hpo.utils_hpo import logged_results_to_HBS_result


import subprocess

def get_git_commit_id():
    """
        Get commit ID for git repository.
    """

    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit_id.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"


def write_config_from_hpo_run_to_yaml_file(config_id, config_fp, hpo_run_fp):
    """ Get configuration ID sampled from HPO run and write it to a YAML file.

    :param config_id: tuple, configuration ID of requested configuration sampled in HPO run
    :param config_fp: Path, YAML filepath used to save configuration
    :param hpo_run_fp: Path, HPO run filepath

    :return:
    """

    config = {}

    # load results json for HPO run
    res = logged_results_to_HBS_result(hpo_run_fp)

    # load HPO run parameters (including fixed hyperparameters)
    with open(hpo_run_fp / 'hpo_run_config.yaml', "r") as f:
        hpo_run_config = yaml.unsafe_load(f)
        # get fixed hyperparameters
        fixed_config = hpo_run_config['config']
        # get model architecture name
        model_architecture = hpo_run_config['model_architecture']
        # get features set
        features_set = hpo_run_config['features_set']

    # get commit id for the repository
    config['commit_id'] = get_git_commit_id()

    # add model architecture class name
    config['model_architecture'] = model_architecture

    # add features set
    config['features_set'] = features_set

    # add fixed hyperparameters
    config['config'] = fixed_config

    # get all configs
    id2config = res.get_id2config_mapping()

    # get HPO sampled configuration based on config ID
    config['config'].update(id2config[config_id]['config'])

    with open(config_fp, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


if __name__ == "__main__":

    # HPO run directory
    hpo_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/hpo_configs/hporun_newexominer_tessspoc_2mins1-s88_notsharedtargetsffi_5-22-2025_1544')
    # requested config id from HPO run
    config_id_hpo_run = (28, 0, 3)
    # destination filepath
    save_config_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/models/exominer_new.yaml')

    write_config_from_hpo_run_to_yaml_file(config_id_hpo_run, save_config_fp, hpo_dir)

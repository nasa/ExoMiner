"""
Create configuration files for SHAP runs.
"""

# 3rd party
from pathlib import Path
import itertools
import yaml
import numpy as np
from datetime import datetime

# local
from utils.utils_dataio import is_yamlble

# load configuration for the explainability run
path_to_yaml = Path('/home/msaragoc/Projects/exoplnt_dl/codebase/explainability/shap/config_create_run.yaml')
with(open(path_to_yaml, 'r')) as file:
    run_config = yaml.safe_load(file)

# create experiment directory
config_dir = Path(run_config['exp_dir']) / f'shap_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
config_dir.mkdir(exist_ok=True)
runs_dir = config_dir / 'runs_configs'
runs_dir.mkdir(exist_ok=True)

trained_models_dir = config_dir / 'trained_models'
trained_models_dir.mkdir(exist_ok=True)
results_ensemble_dir = config_dir / 'results_ensemble'
results_ensemble_dir.mkdir(exist_ok=True)

# create combinations of branches
n_branches = len(run_config['branches'])
branches_names = list(run_config['branches'].keys())
branches_nicknames = list(run_config['branches'].values())
branches_range = np.arange(n_branches)

runs = {}
for L in range(1, n_branches + 1):
    for subset in itertools.combinations(branches_range, L):
        # print(subset)
        # print(np.array(branches_names)[np.array(subset)])
        branches_run = [branches_names[idx] for idx in range(n_branches) if idx in subset]
        features_run = []
        for branch in branches_run:
            # features_run.extend(run_config['branches_features'][branch])
            if run_config['branches_features'][branch]['type'] == 'scalar':
                features_run.extend(run_config['branches_features'][branch]['features'])
            else:
                features_run.extend(run_config['branches_features'][branch]['features']['views'])
                if run_config['branches_features'][branch]['features']['scalars'] is not None:
                    features_run.extend(run_config['branches_features'][branch]['features']['scalars'])

        runs['-'.join(np.array(branches_nicknames)[np.array(subset)])] = {'branches': branches_run,
                                                                          'features': features_run}

# remove runs for which more than one feature group was excluded
runs = {run_name: run_data for run_name, run_data in runs.items()
        if len(run_name.split('-')) >= n_branches - 1}
# # remove runs for which global and local centroid are not both present
# runs = {run_name: run_data for run_name, run_data in runs.items()
#         if not (('global_centr_view_std_noclip' in run_data['branches'] and 'local_centr_view_std_noclip' not in
#                  run_data['branches']) or
#                 ('global_centr_view_std_noclip' not in run_data['branches'] and 'local_centr_view_std_noclip' in
#                  run_data['branches']))}
# # remove runs that do not include global and local centroid features
# runs = {run_name: run_data for run_name, run_data in runs.items()
#         if (('global_centr_view_std_noclip' in run_data['branches'] and 'local_centr_view_std_noclip' in run_data['branches']))}
# # remove runs  that do not include wks features
# runs = {run_name: run_data for run_name, run_data in runs.items()
#         if 'local_weak_secondary_view_max_flux-wks_norm' in run_data['branches']}

config_runs_fp = config_dir / f'list_config_runs.txt'
if config_runs_fp.is_file():
    config_runs_fp.unlink()
# aa
for run_name, run_data in runs.items():

    with(open(run_config['default_train_config_fp'], 'r')) as file:  # read default YAML configuration pred file
        train_config = yaml.safe_load(file)

    with(open(run_config['default_pred_config_fp'], 'r')) as file:  # read default YAML configuration pred file
        pred_config = yaml.safe_load(file)

    # set experiment directory for training
    # train_config['paths']['experiment_dir'] = f'{train_config["paths"]["experiment_dir"]}/{run_name}'
    train_config['paths']['experiment_dir'] = f'{str(trained_models_dir / run_name)}'

    # assign branches
    # train_config['config']['branches'] = run_data['branches']
    # aaaa
    for branch_name in run_data['branches']:
        # aa
        branch_to_add = {branch_name: run_config['branches_features'][branch_name]['features']}
        if run_config['branches_features'][branch_name]['type'] == 'convolutional':
            if train_config['config']['conv_branches'] is None:
                train_config['config']['conv_branches'] = branch_to_add
            else:
                train_config['config']['conv_branches'].update(branch_to_add)
        if run_config['branches_features'][branch_name]['type'] == 'scalar':
            if train_config['config']['scalar_branches'] is None:
                train_config['config']['scalar_branches'] = branch_to_add
            else:
                train_config['config']['scalar_branches'].update(branch_to_add)
        if run_config['branches_features'][branch_name]['type'] == 'transformer':
            if train_config['config']['transformer_branches'] is None:
                train_config['config']['transformer_branches'] = branch_to_add
            else:
                train_config['config']['transformer_branches'].update(branch_to_add)

    # assign features
    train_config['features_set'] = {feature_name: feature_info for feature_name, feature_info
                                    in train_config['features_set'].items() if feature_name in run_data['features']}

    # save the YAML config train file
    yaml_dict = {key: val for key, val in train_config.items() if is_yamlble(val)}  # check if parameters are YAMLble
    with open(runs_dir / f'config_shap_{run_name}_train.yaml', 'w') as config_file:
        yaml.dump(yaml_dict, config_file, sort_keys=False)

    # set experiment directory for predicting
    # pred_config['paths']['experiment_dir'] = f'{pred_config["paths"]["experiment_dir"]}/{run_name}'
    pred_config['paths']['experiment_dir'] = f'{results_ensemble_dir / run_name}'
    # get models directory
    pred_config['paths']['models_dir'] = f'{train_config["paths"]["experiment_dir"]}'

    # assign features
    pred_config['features_set'] = {feature_name: feature_info for feature_name, feature_info
                                   in train_config['features_set'].items() if feature_name in run_data['features']}

    # save the YAML config pred file
    yaml_dict = {key: val for key, val in pred_config.items() if is_yamlble(val)}  # check if parameters are YAMLble
    with open(runs_dir / f'config_shap_{run_name}_predict.yaml', 'w') as config_file:
        yaml.dump(yaml_dict, config_file, sort_keys=False)

    with open(config_dir / f'list_config_runs.txt', 'a') as list_configs_file:
        list_configs_file.write(f'config_shap_{run_name}\n')

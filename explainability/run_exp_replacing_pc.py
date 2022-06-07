"""
Run explainability test that iteratively replaces sets of features (branches) for a set of representative PCs by the
corresponding features for each example in the dataset.
Representative PCs are examples ranked in the top (score > thr) by the classifier.
The sets of features that change the classification of the representative PC from PC to FP when they are replaced by the
features of the example are considered to explain the classification of that given example.
Results are averaged over trials (different sets of representative PCs) and representative PCs.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import yaml
from datetime import datetime
import logging

# local
from utils import explain_branches_pcs, form_info_table_pcs

# load configuration for the explainability run
path_to_yaml = Path('/kdd/config_replacing_pc.yaml')
with(open(path_to_yaml, 'r')) as file:
    run_config = yaml.safe_load(file)

# create experiment directory
exp_dir = Path(run_config['exp_root_dir']) / f'run_blocking_pc_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
exp_dir.mkdir(exist_ok=True)

# save run configuration file
with open(exp_dir / 'run_config.yaml', 'w') as config_file:
    yaml.dump(run_config, config_file)

# set up logger
logger = logging.getLogger(name=f'explainability run')
logger_handler = logging.FileHandler(filename=exp_dir / f'run.log',
                                     mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting explainability run {exp_dir.name}...')

data_dir = Path(run_config['data_dir'])

# iterate over the data sets to get examples additional information
logger.info('Loading additional data for examples in each dataset...')
examples_info = {dataset:
                     pd.DataFrame.from_dict(np.load(data_dir /
                                                    f'{dataset}_data_params.npy', allow_pickle=True).item()[dataset])
                 for dataset in run_config['datasets']}

# load model
logger.info(f'Loading model from {run_config["model_filepath"]}')
model = load_model(filepath=run_config['model_filepath'], compile=False)
# model.summary()

# load data for the different data sets
logger.info('Loading data for examples in each dataset...')
all_data = {f'{dataset}': np.load(data_dir / f'all_{dataset}_data.npy', allow_pickle=True).item()
            for dataset in run_config['datasets']}

train_info = np.load(data_dir / 'train_data_params.npy', allow_pickle=True).item()

# load scores for all examples in the data sets
full_dataset_scores = {dataset: model.predict(all_data[f'{dataset}']) for dataset in run_config['datasets']}

# select candidate examples to representative PCs
best_PCs = np.where(full_dataset_scores['train'] > run_config['best_pcs_score_thr'])[0]
logger.info(f'Number of candidate examples from the training set as representative PCs: {len(best_PCs)}')

for trial_num in range(run_config['num_trials']):  # iterate over trials

    logger.info(f'Trial {trial_num + 1}/{run_config["num_trials"]}')

    trial_run = f'exp_{run_config["num_PCs"]}_PCs_trial_{trial_num}'
    trial_run_dir = exp_dir / trial_run
    trial_run_dir.mkdir(exist_ok=True)

    # randomly select num_PCs out of the candidates to representative PCs
    pc_inds = np.random.choice(best_PCs, run_config['num_PCs'])
    logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}] Selected {run_config["num_PCs"]} examples from '
                f'the training set as representative PCs')

    # write which PCs were used as the best templates
    with open(trial_run_dir / "PC_list.txt", "w") as text_file:
        for best_example_ind in pc_inds:
            text_file.write(f'KIC {train_info["train"]["target_id"][best_example_ind]}.'
                            f'{train_info["train"]["tce_plnt_num"][best_example_ind]}\n')

    for dataset in run_config['datasets']:  # iterate over the datasets

        logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}] - Dataset {dataset}')

        for pc_ind_i, pc_ind in enumerate(pc_inds):  # iterate over the PC templates

            logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}, Dataset {dataset}] - '
                        f'PC {pc_ind_i + 1}/{run_config["num_PCs"]}')

            # iterate over the combinations of branches/features to be replaced
            features_grouping_scores = {feature_grouping_name: []
                                     for feature_grouping_name in run_config['features_grouping']}
            for feature_grouping_name, feature_grouping in run_config['features_grouping'].items():

                logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}, Dataset {dataset}, '
                            f'PC {pc_ind_i + 1}/{run_config["num_PCs"]}] - Feature grouping {feature_grouping_name}')

                explain_feature_grouping_scores = explain_branches_pcs([feature_grouping],
                                                                    all_data[dataset],
                                                                    all_data['train'],
                                                                    model,
                                                                    pc_ind
                                                                    )
                features_grouping_scores[feature_grouping_name] = explain_feature_grouping_scores

            # put results into table
            group = form_info_table_pcs(full_dataset_scores[dataset],
                                        full_dataset_scores['train'],
                                        features_grouping_scores,
                                        pc_ind,
                                        examples_info[dataset])

            # save results for each representative PC into a csv file
            group.to_csv(trial_run_dir / f'{dataset}_top_{pc_ind_i}.csv', index=False)

logger.info(f'Finished explainability run.')

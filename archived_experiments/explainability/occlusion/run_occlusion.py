"""
Run explainability test that iteratively zeros-out sets of features (branches) for each example in the dataset.
Representative PCs are examples ranked in the top (score > thr) by the classifier.
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
from archived_experiments.explainability.norex.norex_exclusion import explain_branches_occlusion, form_info_table_occlusion

# load configuration for the explainability run
path_to_yaml = Path('/explainability/occlusion/config_occlusion.yaml')
with(open(path_to_yaml, 'r')) as file:
    run_config = yaml.safe_load(file)

# create experiment directory
exp_dir = Path(run_config['exp_root_dir']) / f'run_occlusion_zero_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
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
examples_info = {
    dataset: pd.DataFrame.from_dict(
        np.load(data_dir / f'all_{dataset}_data.npy', allow_pickle=True).item()['example_info'])
    for dataset in run_config['datasets']}

# load model
logger.info(f'Loading model from {run_config["model_filepath"]}')
model = load_model(filepath=run_config['model_filepath'], compile=False)
# model.summary()

# load data for the different data sets
logger.info('Loading data for examples in each dataset...')
all_data = {f'{dataset}': np.load(data_dir / f'all_{dataset}_data.npy', allow_pickle=True).item()['features']
            for dataset in run_config['datasets']}

# run inference on examples in the data sets to generate the full model scores
full_dataset_scores = {dataset: model.predict(all_data[f'{dataset}']) for dataset in run_config['datasets']}

for dataset in run_config['datasets']:  # iterate over the datasets

    logger.info(f'Dataset {dataset}')

    # iterate over the combinations of branches/features to be replaced
    features_grouping_scores = {feature_grouping_name: []
                                for feature_grouping_name in run_config['features_grouping']}
    for feature_grouping_name, feature_grouping in run_config['features_grouping'].items():

        logger.info(f'[Dataset {dataset}] - Feature grouping {feature_grouping_name}')

        explain_feature_grouping_scores = explain_branches_occlusion(
            [feature_grouping],
            all_data[dataset],
            model
        )
        features_grouping_scores[feature_grouping_name] = explain_feature_grouping_scores

    # create table with scores for the different branch occlusion + full model for the data set
    group = form_info_table_occlusion(full_dataset_scores[dataset],
                                      features_grouping_scores,
                                      examples_info[dataset])

    # save results into csv files
    group.to_csv(exp_dir / f'{dataset}.csv', index=False)

logger.info(f'Finished explainability run.')

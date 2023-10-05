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
import multiprocessing
import argparse

# local
from norex_utils import run_trial


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pcs', type=int, help='Number of model PCs', default=-1)
    parser.add_argument('--thr_modelpcs', type=int, help='Threshold index for choosing model PCs based on their score', default=-1)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.',
                        default='/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/explainability/config_norex.yaml')
    args = parser.parse_args()

    # load configuration for the explainability run
    with(open(args.config_file, 'r')) as file:
        run_config = yaml.safe_load(file)

    if args.num_pcs != -1:
        run_config['num_PCs'] = args.num_pcs

    if args.thr_modelpcs != -1:
        run_config['num_PCs'] = args.num_pcs

    # thr_modelpcs_arr = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.99]
    # run_config['num_PCs'] = args.num_pcs
    # run_config['best_pcs_score_thr'] = thr_modelpcs_arr[args.thr_modelpcs]

    # create experiment directory
    exp_dir = Path(run_config['exp_root_dir']) / f'run_numPCs_{run_config["num_PCs"]}_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    # exp_dir = Path(run_config['exp_root_dir']) / f'run_num_model_pcs_{run_config["num_PCs"]}_thr_{run_config["best_pcs_score_thr"]}_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
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

    # train_info = np.load(data_dir / 'all_train_data.npy', allow_pickle=True).item()['example_info']

    # run inference on examples in the data sets to generate the full model scores
    full_dataset_scores = {dataset: model.predict(all_data[f'{dataset}']) for dataset in run_config['datasets']}

    # select indices of candidate examples to representative PCs (i.e., examples that have a model score > thr)
    best_PCs = np.where(full_dataset_scores['train'] > run_config['best_pcs_score_thr'])[0]
    n_best_pcs = len(best_PCs)
    logger.info(f'Number of candidate examples from the training set as representative PCs: {len(best_PCs)}')
    if n_best_pcs < run_config['num_PCs']:
        raise ValueError(f'The number of candidate model PCs ({n_best_pcs}) in training set for thr '
                         f'{run_config["best_pcs_score_thr"]} is less than the number of requested model PCs '
                         f'({run_config["num_PCs"]})')

    for trial_num in range(run_config['num_trials']):  # iterate over trials
        logger.info(f'Running trial {trial_num + 1} out of {run_config["num_trials"]}...')
        run_trial(trial_num, run_config, exp_dir, best_PCs, examples_info, all_data, full_dataset_scores)  # , logger)

    # nprocesses = 2
    # logger.info(f'Starting {nprocesses} processes for running {run_config["num_trials"]} trials...')
    # pool = multiprocessing.Pool(processes=nprocesses)
    # jobs = [(trial_num, run_config, exp_dir, best_PCs, examples_info, all_data, full_dataset_scores)
    #         for trial_num in range(run_config['num_trials'])]
    # async_results = [pool.apply_async(run_trial, job) for job in jobs]
    # pool.close()
    # for async_result in async_results:
    #     async_result.get()

    logger.info(f'Finished explainability run.')

""" Run cross-validation experiment. """

# 3rd party
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from pathlib import Path
import numpy as np
import logging
from tensorflow.keras import callbacks
import copy
import time
import argparse
import sys
from mpi4py import MPI
# import multiprocessing
import yaml
import pandas as pd

# local
from models import models_keras
from src_hpo.utils_hpo import load_hpo_config
from utils.utils_dataio import is_yamlble
from src_cv.utils_cv import processing_data_run
from src.utils_train_eval_predict import (
    train_model, evaluate_model, predict_model, write_performance_metrics_to_txt_file, set_tf_data_type_for_features)
# from src.utils_train import PredictDuringFitCallback
from src.utils_dataio import get_data_from_tfrecords_for_predictions_table


def cv_run(data_shards_fps, run_params):
    """ Run one iteration of CV.

    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
    :param run_params: dict, configuration parameters for the CV run
    :return:
    """

    run_params['paths']['experiment_dir'] = (run_params['paths']['experiment_root_dir'] /
                                             f'cv_iter_{run_params["cv_id"]}')
    run_params['paths']['experiment_dir'].mkdir(exist_ok=True)

    # split training folds into training and validation sets by randomly selecting one of the folds as the validation
    # set
    data_shards_fps_eval = copy.deepcopy(data_shards_fps)
    if run_params['val_from_train']:
        data_shards_fps_eval['val'] = run_params['rng'].choice(data_shards_fps['train'], 1, replace=False)
        data_shards_fps_eval['train'] = np.setdiff1d(data_shards_fps['train'], data_shards_fps_eval['val'])

    # # save fold used
    # with open(run_params['paths']['experiment_dir'] / 'fold_split.json', 'w') as fold_split_file:
    #     yaml.dump(data_shards_fps_eval, fold_split_file, sort_keys=False)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Processing data for CV iteration')
    run_params['datasets_fps'] = processing_data_run(data_shards_fps_eval, run_params,
                                                     run_params['paths']['experiment_dir'])

    # get data from TFRecords files to be displayed in the table with predictions
    data = get_data_from_tfrecords_for_predictions_table(run_params['datasets'],
                                                         run_params['data_fields'],
                                                         run_params['datasets_fps'])

    # add label id
    for dataset in data:
        data[dataset]['label_id'] = (
            data[dataset][run_params['label_field_name']].apply(lambda x: run_params['label_map'][x]))

    # sequential training
    models_dir = run_params['paths']['experiment_dir'] / 'models'
    models_dir.mkdir(exist_ok=True)
    for model_id in range(run_params['training']['n_models']):  # train N models
        if run_params['logger'] is not None:
            run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Training model {model_id + 1} '
                                      f'out of {run_params["training"]["n_models"]} on '
                                      f'{run_params["training"]["n_epochs"]} epochs...')
        model_dir = models_dir / f'model{model_id}'
        model_dir.mkdir(exist_ok=True)

        # predict_fit_callback = PredictDuringFitCallback(run_params['datasets_fps'],
        #                                                 model_dir,
        #                                                 run_params['inference']['batch_size'],
        #                                                 run_params['label_map'],
        #                                                 run_params['features_set'],
        #                                                 run_params['config']['multi_class'],
        #                                                 run_params['config']['use_transformer'],
        #                                                 run_params['feature_map'],
        #                                                 data,
        #                                                 verbose=True
        #                                                 )
        #
        # if len(run_params['callbacks_list']['train']) == 1:
        #     run_params['callbacks_list']['train'].append(predict_fit_callback)
        # else:
        #     run_params['callbacks_list']['train'][1] = predict_fit_callback

        # instantiate child process for training the model; prevent GPU memory issues
        train_model(
            run_params['base_model'],
            run_params,
            model_dir,
            model_id,
            run_params['logger']
        )

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Evaluating ensemble...')
    # get the filepaths for the trained models
    run_params['paths']['models_filepaths'] = [model_dir / f'{model_dir.stem}.keras'
                                               for model_dir in models_dir.iterdir() if 'model' in model_dir.stem]

    res_eval = evaluate_model(run_params, run_params['paths']['experiment_dir'], run_params['logger'])
    np.save(run_params['paths']['experiment_dir'] / 'res_eval.npy', res_eval)

    # write results to a txt file
    write_performance_metrics_to_txt_file(run_params['paths']['experiment_dir'], run_params['datasets'], res_eval)

    # run inference on different data sets defined in run_params['datasets']
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Running inference using ensemble...')
    scores = predict_model(run_params, run_params['paths']['experiment_dir'], run_params['logger'])
    # scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8')
    #                          for dataset in run_params['datasets']}
    # for dataset in run_params['datasets']:
    #     # threshold for classification
    #     if not run_params['config']['multi_class']:  # binary classification
    #         scores_classification[dataset][scores[dataset] >= run_params['metrics']['clf_thr']] = 1
    #     else:  # multiclass
    #         scores_classification[dataset] = [scores[dataset][i].argmax() for i in range(scores[dataset].shape[0])]

    # add predictions to the data dict
    for dataset in run_params['datasets']:
        if not run_params['config']['multi_class']:
            data[dataset]['score'] = scores[dataset].ravel()
            # data[dataset]['predicted class'] = scores_classification[dataset].ravel()
        else:
            for class_label, label_id in run_params['label_map'].items():
                data[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]
            # data[dataset]['predicted class'] = scores_classification[dataset]

    # write results to a txt file
    for dataset in run_params['datasets']:

        data_df = pd.DataFrame(data[dataset])

        # sort in descending order of output
        if not run_params['config']['multi_class']:
            data_df.sort_values(by='score', ascending=False, inplace=True)
        data_df.to_csv(run_params['paths']['experiment_dir'] / f'ensemble_ranked_predictions_{dataset}set.csv',
                       index=False)

    # if run_params['logger'] is not None:
    # run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # # remove preprocessed data for this run
    # shutil.rmtree(cv_run_dir / 'norm_data')
    # # TODO: delete the models as well?

    if run_params['logger'] is not None:
        run_params['logger'].info(f'Finished CV iteration {run_params["cv_id"]}.')


def cv():
    """ Run CV experiment. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.',
                        default='/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src_cv/config_cv_train.yaml')
    args = parser.parse_args()

    with(open(args.config_file, 'r')) as file:
        config = yaml.safe_load(file)

    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    config['size'] = MPI.COMM_WORLD.size
    print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    sys.stdout.flush()
    if config['rank'] != 0:
        time.sleep(2)

    # set paths
    for path_name, path_str in config['paths'].items():
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_root_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    config['data_shards_fns'] = np.load(config['paths']['cv_folds'], allow_pickle=True)
    config['data_shards_fps'] = [{dataset: [config['paths']['tfrec_dir'] / fold for fold in cv_iter[dataset]]
                                  for dataset in cv_iter} for cv_iter in config['data_shards_fns']]

    if config["rank"] >= len(config['data_shards_fps']):
        print(f'Number of processes requested to run CV ({config["rank"]}) is higher than the number CV iterations'
              f'({len(config["data_shards_fps"])}). Ending process.')
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_root_dir'] /
                                                  f'cv_iter_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["paths"]["experiment_root_dir"].name}...')

    # setting GPU
    config['logger'].info(f'Number of GPUs selected per node = {config["ngpus_per_node"]}')
    config['gpu_id'] = config["rank"] % config['ngpus_per_node']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    config['logger'].info(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    config['logger'].info(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    # load model hyperparameters from HPO run; overwrites the one in the yaml file
    config_hpo_chosen, config['hpo_config_id'] = (
        load_hpo_config(Path(config['paths']['hpo_dir']), logger=config['logger']))
    config['config'].update(config_hpo_chosen)
    # save the YAML file with the HPO configuration that was used
    if config['rank'] == 0:
        with open(config['paths']['experiment_root_dir'] / 'hpo_config.yaml', 'w') as hpo_config_file:
            yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)

    # base model used - check models/models_keras.py to see which models are implemented
    config['base_model'] = getattr(models_keras, config['model_architecture'])

    # set tensorflow data type for features in the feature set
    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # early stopping callback
    config['callbacks_list'] = {
        'train': [
            callbacks.EarlyStopping(**config['callbacks']['early_stopping']),
                  ],
    }

    if config['rank'] == 0:

        # save configuration used
        np.save(config['paths']['experiment_root_dir'] / 'run_params.npy', config)
        # save model's architecture and hyperparameters used
        with open(config['paths']['experiment_root_dir'] / 'model_config.yaml', 'w') as file:
            yaml.dump(config['config'], file, sort_keys=False)
        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_root_dir'] / 'run_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file, sort_keys=False)

    if config['train_parallel']:
        # run each CV iteration in parallel
        cv_id = config['rank']
        if config['logger'] is not None:
            config['logger'].info(f'Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
        config['cv_id'] = cv_id
        cv_run(
            config['data_shards_fps'][cv_id],
            config,
        )
    else:
        # run each CV iteration sequentially
        for cv_id, cv_iter in enumerate(config['data_shards_fps']):
            if config['logger'] is not None:
                config['logger'].info(
                    f'[cv_iter_{cv_id}] Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
            config['cv_id'] = cv_id
            cv_run(
                cv_iter,
                config,
            )


if __name__ == '__main__':

    cv()

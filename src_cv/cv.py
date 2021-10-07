""" Run cross-validation experiment. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import logging
# from datetime import datetime
import tensorflow as tf
from tensorflow.keras import callbacks
# import itertools
import copy
# import shutil
import time
import argparse
import sys
from mpi4py import MPI
import json
import multiprocessing
import yaml

# local
import paths
from models.models_keras import CNN1dPlanetFinderv2
import models.config_keras
from src_hpo import utils_hpo
from utils.utils_dataio import is_jsonable, is_yamlble
from src_cv.utils_cv import processing_data_run, train_model, eval_ensemble


def cv_run(cv_dir, data_shards_fps, run_params):
    """ Run one iteration of CV.

    :param cv_dir: Path, directory for the CV iteration
    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
    :param run_params: dict, configuration parameters for the CV ru
    :return:
    """

    cv_run_dir = cv_dir / f'cv_iter_{run_params["cv_id"]}'
    cv_run_dir.mkdir(exist_ok=True)

    # split training folds into training and validation sets by randomly selecting one of the folds as the validation
    # set
    data_shards_fps_eval = copy.deepcopy(data_shards_fps)
    data_shards_fps_eval['val'] = run_params['rng'].choice(data_shards_fps['train'], 1, replace=False)
    data_shards_fps_eval['train'] = np.setdiff1d(data_shards_fps['train'], data_shards_fps_eval['val'])

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Split for CV iteration: {data_shards_fps_eval}')

    with open(cv_run_dir / 'fold_split.json', 'w') as cv_run_file:
        json.dump({dataset: [str(fp) for fp in data_shards_fps_eval[dataset]] for dataset in data_shards_fps_eval},
                  cv_run_file)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['config'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing data for CV iteration')
    data_shards_fps_eval = processing_data_run(data_shards_fps_eval, run_params, cv_run_dir)

    # sequential training
    for model_id in range(run_params['training']['n_models']):  # train N models
        if run_params['config'] is not None:
            run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Training model {model_id + 1} '
                                      f'out of {run_params["training"]["n_models"]} on '
                                      f'{run_params["training"]["n_epochs"]} epochs...')

        p = multiprocessing.Process(target=train_model,
                                    args=(
                                        run_params,
                                        model_id,
                                        {dataset: fps for dataset, fps in data_shards_fps_eval.items() if
                                         dataset != 'test'},
                                        cv_run_dir,
                                    ))
        p.start()
        p.join()
        # p.close()

    if run_params['config'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Evaluating ensemble')
    # get the filepaths for the trained models
    models_dir = cv_run_dir / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]

    # evaluate ensemble
    p = multiprocessing.Process(target=eval_ensemble,
                                args=(
                                    models_filepaths,
                                    run_params,
                                    data_shards_fps_eval,
                                    cv_run_dir,
                                ))
    p.start()
    p.join()

    # if run_params['config'] is not None:
    # run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # # remove preprocessed data for this run
    # shutil.rmtree(cv_run_dir / 'norm_data')
    # # TODO: delete the models as well?


def cv():

    path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/src_cv/config_cv_train.yaml')
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    config['size'] = MPI.COMM_WORLD.size
    print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    sys.stdout.flush()
    if config['rank'] != 0:
        time.sleep(2)

    try:
        print(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
        print(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except:
        print(f'[rank_{config["rank"]}] No CUDA environment variables exist.')

    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))  # number of GPUs visible to the process
    if config["rank"] == 0:
        print(f'Number of GPUs selected per node = {config["ngpus_per_node"]}')
    config['gpu_id'] = config["rank"] % config['ngpus_per_node']

    # setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])  # "0, 1"

    print(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    print(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # tf.debugging.set_log_device_placement(True)

    for path_name, path_str in config['paths'].items():
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    config['data_shards_fns'] = np.load(config['paths']['cv_folds'], allow_pickle=True)
    config['data_shards_fps'] = [{dataset: [config['paths']['tfrec_dir'] / fold for fold in cv_iter[dataset]]
                                  for dataset in cv_iter} for cv_iter in config['data_shards_fns']]

    if config["rank"] >= len(config['data_shards_fps']):
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'cv_run_{config["rank"]}.log',
                                         mode='w')
    # logger_handler_stream = logging.StreamHandler(sys.stdout)
    # logger_handler_stream.setLevel(logging.INFO)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    # logger_handler_stream.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    # logger.addHandler(logger_handler_stream)
    config['logger'].info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    if config['paths']['hpo_dir'] is not None:
        hpo_path = Path(config['paths']['hpo_dir'])
        res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_path.name}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config['config'].update(id2config[config_id_hpo]['config'])

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        config['logger'].info(f'Using configuration from HPO study {hpo_path.name}')
        config['logger'].info(f'HPO Config {config_id_hpo}: {config["config"]}')

    # base model used - check estimator_util.py to see which models are implemented
    config['base_model'] = CNN1dPlanetFinderv2

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32

    config['logger'].info(f'Feature set: {config["features_set"]}')

    # early stopping callback
    config['callbacks']['early_stopping']['obj'] = callbacks.EarlyStopping(**config['callbacks']['early_stopping'])

    config['logger'].info(f'Final configuration used: {config}')

    # save feature set used
    if config['rank'] == 0:
        np.save(config['paths']['experiment_dir'] / 'features_set.npy', config['features_set'])
        # save configuration used
        np.save(config['paths']['experiment_dir'] / 'config.npy', config['config'])

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'cv_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    # run each CV iteration sequentially
    for cv_id, cv_iter in enumerate(config['data_shards_fps']):
        config['logger'].info(
            f'[cv_iter_{cv_iter}] Running CV iteration {cv_id} (out of {len(config["data_shards_fps"])}): '
            f'{cv_iter}')
        config['cv_id'] = cv_id
        cv_run(
            config['paths']['experiment_dir'],
            cv_iter,
            config,
        )

    # # run each CV iteration in parallel
    # cv_id = config['rank']
    # if config['logger'] is not None:
    #     config['logger'].info(f'Running CV iteration {cv_id} (out of {len(config["data_shards_fps"])}): '
    #             f'{config["data_shards_fps"][cv_id]}')
    # config['cv_id'] = cv_id
    # cv_run(
    #     config['paths']['experiment_dir'],
    #     config['data_shards_fps'][cv_id],
    #     config,
    # )


if __name__ == '__main__':

    cv()
